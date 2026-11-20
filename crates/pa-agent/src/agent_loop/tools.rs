//! Tool-call batch execution: sequential and parallel dispatch, per-tool
//! preparation outcomes, batch termination, and tool-result message
//! emission.

use std::sync::Arc;

use crate::abort::AbortSignal;
use crate::types::{
    AgentContext, AgentEvent, AgentTool, AgentToolResult, AssistantMessage, NativeToolExchange,
    ToolCall, ToolExecutionMode, ToolResultMessage,
};

use super::run::clone_context;
use super::tool_call::{
    create_tool_result_message, emit_tool_execution_end, emit_tool_result_message,
    execute_prepared_tool_call, finalize_executed_tool_call, prepare_tool_call,
};
use super::{AgentEventSink, AgentLoopConfig};

pub(crate) struct ExecutedToolCallBatch {
    pub(crate) messages: Vec<ToolResultMessage>,
    pub(crate) tool_execution: ToolExecutionMode,
    pub(crate) exchanges: Vec<NativeToolExchange>,
    pub(crate) terminate: bool,
}

pub(crate) struct FinalizedToolCallOutcome {
    pub(crate) tool_call: ToolCall,
    pub(crate) result: AgentToolResult,
}

/// Tool lookup by name; exact matches win over alias-based dispatch.
pub(crate) fn find_tool<'a>(
    tools: &'a [Arc<dyn AgentTool>],
    name: &str,
) -> Option<&'a Arc<dyn AgentTool>> {
    if let Some(tool) = tools.iter().find(|t| t.name() == name) {
        return Some(tool);
    }
    tools.iter().find(|t| t.accepts_alias(name))
}

pub(crate) async fn execute_tool_calls(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    config: &AgentLoopConfig,
    signal: Option<&AbortSignal>,
    emit: &AgentEventSink,
) -> anyhow::Result<ExecutedToolCallBatch> {
    let tool_calls = assistant_message
        .tool_calls()
        .into_iter()
        .cloned()
        .collect::<Vec<ToolCall>>();
    let has_sequential_tool_call = tool_calls.iter().any(|tc| {
        find_tool(&current_context.tools, &tc.name).and_then(|tool| tool.execution_mode())
            == Some(ToolExecutionMode::Sequential)
    });
    if config.tool_execution == ToolExecutionMode::Sequential || has_sequential_tool_call {
        execute_tool_calls_sequential(
            current_context,
            assistant_message,
            &tool_calls,
            config,
            signal,
            emit,
        )
        .await
    } else {
        execute_tool_calls_parallel(
            current_context,
            assistant_message,
            &tool_calls,
            config,
            signal,
            emit,
        )
        .await
    }
}

async fn execute_tool_calls_sequential(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: Option<&AbortSignal>,
    emit: &AgentEventSink,
) -> anyhow::Result<ExecutedToolCallBatch> {
    let mut finalized_calls: Vec<FinalizedToolCallOutcome> = Vec::new();
    let mut messages: Vec<ToolResultMessage> = Vec::new();
    let mut exchanges = Vec::new();

    for (source_order, tool_call) in tool_calls.iter().enumerate() {
        if signal.is_some_and(AbortSignal::is_aborted) {
            break;
        }

        emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: tool_call.arguments.clone(),
        })
        .await?;

        let preparation = prepare_tool_call(
            current_context,
            assistant_message,
            tool_call,
            config,
            signal,
        )
        .await;
        let mut executed_input = tool_call.arguments.clone();
        let finalized = match preparation {
            Preparation::Immediate(result) => FinalizedToolCallOutcome {
                tool_call: tool_call.clone(),
                result,
            },
            Preparation::Prepared(prepared) => {
                executed_input = prepared.args.clone();
                let executed =
                    execute_prepared_tool_call(&prepared, config.tool_progress, signal, emit).await;
                finalize_executed_tool_call(
                    current_context,
                    assistant_message,
                    &prepared,
                    executed,
                    config,
                    signal,
                )
                .await?
            }
        };

        emit_tool_execution_end(&finalized, emit).await?;
        let tool_result_message = create_tool_result_message(&finalized);
        emit_tool_result_message(&tool_result_message, emit).await?;
        exchanges.push(NativeToolExchange {
            source_order,
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            original_input: tool_call.arguments.clone(),
            executed_input,
            result: tool_result_message.clone(),
        });
        messages.push(tool_result_message);
        finalized_calls.push(finalized);

        if signal.is_some_and(AbortSignal::is_aborted) {
            break;
        }
    }

    Ok(ExecutedToolCallBatch {
        messages,
        tool_execution: ToolExecutionMode::Sequential,
        exchanges,
        terminate: should_terminate_tool_batch(&finalized_calls),
    })
}

enum ParallelToolEntry {
    Task(tokio::task::JoinHandle<anyhow::Result<(FinalizedToolCallOutcome, serde_json::Value)>>),
    // Boxed: the finalized outcome holds the tool call's ordered
    // argument map (insertion-ordered for wire parity), which dwarfs
    // the join handle and would trip `large_enum_variant`.
    Outcome(Box<FinalizedToolCallOutcome>, serde_json::Value),
}

/// `tool_execution_end` is emitted in completion order (from inside the
/// concurrent tasks), while tool-result message events are emitted afterwards
/// in assistant source order, matching the TS reference.
async fn execute_tool_calls_parallel(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: Option<&AbortSignal>,
    emit: &AgentEventSink,
) -> anyhow::Result<ExecutedToolCallBatch> {
    let mut entries: Vec<ParallelToolEntry> = Vec::new();

    for tool_call in tool_calls {
        emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: tool_call.arguments.clone(),
        })
        .await?;

        let preparation = prepare_tool_call(
            current_context,
            assistant_message,
            tool_call,
            config,
            signal,
        )
        .await;
        match preparation {
            Preparation::Immediate(result) => {
                let finalized = FinalizedToolCallOutcome {
                    tool_call: tool_call.clone(),
                    result,
                };
                emit_tool_execution_end(&finalized, emit).await?;
                entries.push(ParallelToolEntry::Outcome(
                    Box::new(finalized),
                    tool_call.arguments.clone(),
                ));
            }
            Preparation::Prepared(prepared) => {
                let sink = Arc::clone(emit);
                let assistant_message = assistant_message.clone();
                let context = clone_context(current_context);
                let config = config.clone();
                let signal = signal.cloned().unwrap_or_default();
                let handle = tokio::spawn(async move {
                    let executed = execute_prepared_tool_call(
                        &prepared,
                        config.tool_progress,
                        Some(&signal),
                        &sink,
                    )
                    .await;
                    let finalized = finalize_executed_tool_call(
                        &context,
                        &assistant_message,
                        &prepared,
                        executed,
                        &config,
                        Some(&signal),
                    )
                    .await?;
                    emit_tool_execution_end(&finalized, &sink).await?;
                    Ok((finalized, prepared.args))
                });
                entries.push(ParallelToolEntry::Task(handle));
            }
        }
    }

    // Promise.all semantics: await every entry; results stay in source order.
    let mut ordered_finalized_calls: Vec<FinalizedToolCallOutcome> = Vec::new();
    let mut executed_inputs = Vec::new();
    for entry in entries {
        let (finalized, executed_input) = match entry {
            ParallelToolEntry::Outcome(finalized, executed_input) => (*finalized, executed_input),
            ParallelToolEntry::Task(handle) => handle.await.map_err(|error| {
                anyhow::anyhow!("Parallel tool execution task failed: {error}")
            })??,
        };
        ordered_finalized_calls.push(finalized);
        executed_inputs.push(executed_input);
    }

    let mut messages: Vec<ToolResultMessage> = Vec::new();
    let mut exchanges = Vec::new();
    for (source_order, (finalized, executed_input)) in ordered_finalized_calls
        .iter()
        .zip(executed_inputs)
        .enumerate()
    {
        let tool_result_message = create_tool_result_message(finalized);
        emit_tool_result_message(&tool_result_message, emit).await?;
        exchanges.push(NativeToolExchange {
            source_order,
            tool_call_id: finalized.tool_call.id.clone(),
            tool_name: finalized.tool_call.name.clone(),
            original_input: finalized.tool_call.arguments.clone(),
            executed_input,
            result: tool_result_message.clone(),
        });
        messages.push(tool_result_message);
    }

    Ok(ExecutedToolCallBatch {
        messages,
        tool_execution: ToolExecutionMode::Parallel,
        exchanges,
        terminate: should_terminate_tool_batch(&ordered_finalized_calls),
    })
}

pub(crate) enum Preparation {
    Prepared(PreparedToolCall),
    Immediate(AgentToolResult),
}

pub(crate) struct PreparedToolCall {
    pub(crate) tool_call: ToolCall,
    pub(crate) tool: Arc<dyn AgentTool>,
    pub(crate) args: serde_json::Value,
}

/// The batch terminate rule: ANY terminating result ends the turn, not
/// every one. Models emit work calls and the sanctioned `rlm.end_turn()`
/// in one message. Every call in the batch still executes before the turn
/// ends.
fn should_terminate_tool_batch(finalized_calls: &[FinalizedToolCallOutcome]) -> bool {
    !finalized_calls.is_empty()
        && finalized_calls
            .iter()
            .any(|finalized| finalized.result.terminate == Some(true))
}
