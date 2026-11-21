//! The low-level agent loop.
//!
//! The loop works with [`types::AgentMessage`] throughout and converts to LLM-bound
//! [`types::Message`] values only at the model call boundary.

use std::sync::Arc;

use crate::abort::AbortSignal;
use crate::types::{
    AfterToolCallContext, AfterToolCallResult, AgentEvent, AgentMessage, BeforeToolCallContext,
    BeforeToolCallResult, GetContinuationMessagesContext, Message, Model,
    ShouldStopAfterTurnContext, ThinkingLevel, ToolExecutionMode,
};

/// Sink receiving the loop's events. The loop awaits every emission, so
/// listeners see events strictly in order.
pub type AgentEventSink =
    Arc<dyn Fn(AgentEvent) -> crate::BoxFut<'static, anyhow::Result<()>> + Send + Sync>;

/// `convertToLlm`: converts `AgentMessage`s to LLM-compatible `Message`s
/// before each call. Must not fail; a returned `Err` interrupts the loop like
/// a `throw` in the TS reference.
pub type ConvertToLlmFn = Arc<
    dyn Fn(Vec<AgentMessage>) -> crate::BoxFut<'static, anyhow::Result<Vec<Message>>> + Send + Sync,
>;

/// AgentMessage-level transform applied before `convert_to_llm` (context
/// pruning, external injection).
pub type TransformContextFn = Arc<
    dyn Fn(
            Vec<AgentMessage>,
            AbortSignal,
        ) -> crate::BoxFut<'static, anyhow::Result<Vec<AgentMessage>>>
        + Send
        + Sync,
>;

/// Resolves the system prompt immediately before each LLM call.
pub type GetSystemPromptFn = Arc<dyn Fn() -> String + Send + Sync>;

/// Reads the live tool inventory after the provider context hook completes.
pub type GetToolsFn = Arc<
    dyn Fn() -> crate::BoxFut<'static, anyhow::Result<Vec<Arc<dyn crate::types::AgentTool>>>>
        + Send
        + Sync,
>;

/// Resolves an API key dynamically for each LLM call.
pub type GetApiKeyFn =
    Arc<dyn Fn(String) -> crate::BoxFut<'static, anyhow::Result<Option<String>>> + Send + Sync>;

/// Called after each turn fully completes and `turn_end` was emitted; return
/// true to stop the run before polling steering/follow-up queues.
pub type ShouldStopAfterTurnFn = Arc<
    dyn Fn(ShouldStopAfterTurnContext) -> crate::BoxFut<'static, anyhow::Result<bool>>
        + Send
        + Sync,
>;

/// Called synchronously after a completed turn and before polling for another
/// turn; never checked before the initial assistant turn.
pub type ShouldStopBeforeTurnFn = Arc<dyn Fn() -> bool + Send + Sync>;

/// Returns steering messages to inject mid-run.
pub type PollMessagesFn =
    Arc<dyn Fn() -> crate::BoxFut<'static, anyhow::Result<Vec<AgentMessage>>> + Send + Sync>;

/// Returns continuation messages when the agent would otherwise stop.
pub type GetContinuationMessagesFn = Arc<
    dyn Fn(
            GetContinuationMessagesContext,
            AbortSignal,
        ) -> crate::BoxFut<'static, anyhow::Result<Vec<AgentMessage>>>
        + Send
        + Sync,
>;

/// Return `{ block: true }` to prevent execution.
pub type BeforeToolCallFn = Arc<
    dyn Fn(
            BeforeToolCallContext,
            AbortSignal,
        ) -> crate::BoxFut<'static, anyhow::Result<Option<BeforeToolCallResult>>>
        + Send
        + Sync,
>;

/// Partial override of the executed tool result.
pub type AfterToolCallFn = Arc<
    dyn Fn(
            AfterToolCallContext,
            AbortSignal,
        ) -> crate::BoxFut<'static, anyhow::Result<Option<AfterToolCallResult>>>
        + Send
        + Sync,
>;

/// A turn-local host dispatch. It handles every call, with no fallback to native tools.
pub type HostToolDispatch = Arc<
    dyn Fn(
            crate::types::ToolCall,
            AbortSignal,
        ) -> crate::BoxFut<'static, crate::types::AgentToolResult>
        + Send
        + Sync,
>;

/// Tool inventory, prompt addition and dispatch bound to a single restricted run.
#[derive(Clone)]
pub struct RestrictedRun {
    pub tools: Vec<Arc<dyn crate::types::AgentTool>>,
    pub append_system_prompt: String,
    pub dispatch: HostToolDispatch,
    pub stopped: ShouldStopBeforeTurnFn,
}

impl std::fmt::Debug for RestrictedRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestrictedRun").finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct AgentLoopConfig {
    pub model: Model,
    pub api_key: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub reasoning: ThinkingLevel,
    pub session_id: Option<String>,
    /// The requested provider transport; `None` leaves it to the provider.
    pub transport: Option<crate::types::Transport>,
    /// The requested service tier rides the loop config into every stream
    /// request; `None` until a host supplies one.
    pub service_tier: Option<crate::types::ServiceTier>,
    pub convert_to_llm: ConvertToLlmFn,
    pub transform_context: Option<TransformContextFn>,
    pub get_system_prompt: Option<GetSystemPromptFn>,
    pub get_tools: Option<GetToolsFn>,
    pub get_api_key: Option<GetApiKeyFn>,
    pub should_stop_after_turn: Option<ShouldStopAfterTurnFn>,
    pub should_stop_before_turn: Option<ShouldStopBeforeTurnFn>,
    pub get_steering_messages: Option<PollMessagesFn>,
    pub get_follow_up_messages: Option<PollMessagesFn>,
    pub get_continuation_messages: Option<GetContinuationMessagesFn>,
    pub tool_execution: ToolExecutionMode,
    /// Heartbeat cadence for in-flight tool calls (start-side trace
    /// visibility for long calls); defaults to a 60s tick.
    pub tool_progress: crate::types::ToolProgressCadence,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    pub restricted: Option<RestrictedRun>,
}

impl AgentLoopConfig {
    /// Config with a pass-through `convert_to_llm` (keeps user/assistant/
    /// toolResult messages, filters everything else) and no hooks.
    pub fn new(model: Model, convert_to_llm: ConvertToLlmFn) -> Self {
        AgentLoopConfig {
            model,
            api_key: None,
            temperature: None,
            max_tokens: None,
            reasoning: ThinkingLevel::Off,
            session_id: None,
            transport: None,
            service_tier: None,
            convert_to_llm,
            transform_context: None,
            get_system_prompt: None,
            get_tools: None,
            get_api_key: None,
            should_stop_after_turn: None,
            should_stop_before_turn: None,
            get_steering_messages: None,
            get_follow_up_messages: None,
            get_continuation_messages: None,
            tool_execution: ToolExecutionMode::Parallel,
            tool_progress: crate::types::ToolProgressCadence::default(),
            before_tool_call: None,
            after_tool_call: None,
            restricted: None,
        }
    }

    #[must_use]
    pub fn default_convert_to_llm() -> ConvertToLlmFn {
        Arc::new(|messages: Vec<AgentMessage>| {
            Box::pin(async move {
                Ok(messages
                    .into_iter()
                    .filter(|m| {
                        matches!(
                            m,
                            AgentMessage::Standard(
                                Message::User(_) | Message::Assistant(_) | Message::ToolResult(_),
                            )
                        )
                    })
                    .map(|m| match m {
                        AgentMessage::Standard(message) => message,
                        AgentMessage::Custom(_) => unreachable!("filtered above"),
                    })
                    .collect())
            })
        })
    }
}

mod abort;
mod entry;
mod response;
mod run;
mod tool_call;
mod tools;

pub use entry::{run_agent_loop, run_agent_loop_continue};
