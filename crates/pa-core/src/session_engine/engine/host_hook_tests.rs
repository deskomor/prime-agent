//! The engine's optional host hooks over one scripted tool loop:
//! `before_agent_start` appends to the system prompt, `context` projects the
//! provider request with entry ids, `turn_end` adds hidden rows the next call
//! reads, and `message_end` hears each finished message.
use super::*;
use crate::session_engine::host_hooks::{HostHookFn, SessionHostHooks};
use crate::session_engine::tool_bridge::bridge_tool;
use crate::tools::tool_definition::{ExecutionMode, ToolDefinition, ToolExecutionResult};
use pa_agent::scripted::ScriptedProvider;
use serde_json::{json, Value};

/// Each provider request: its system prompt and its messages, debug-printed.
type Requests = Arc<std::sync::Mutex<Vec<(Option<String>, String)>>>;

fn echo_definition() -> ToolDefinition {
    ToolDefinition {
        name: "echo".to_string(),
        label: "Echo".to_string(),
        description: "Echoes its input".to_string(),
        prompt_snippet: String::new(),
        parameters: json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }),
        execution_mode: Some(ExecutionMode::Sequential),
        prepare_arguments: None,
        execute: Arc::new(|_id, params, _signal, _on_update| {
            let text = params
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or_default()
                .to_string();
            Box::pin(async move { Ok(ToolExecutionResult::text(format!("echo: {text}"))) })
        }),
    }
}

/// A hook that records each payload and answers with `answer(payload)`.
fn recording_hook(
    seen: &Arc<std::sync::Mutex<Vec<Value>>>,
    answer: impl Fn(&Value) -> Option<Value> + Send + Sync + 'static,
) -> HostHookFn {
    let seen = Arc::clone(seen);
    let answer = Arc::new(answer);
    Arc::new(move |payload| {
        let seen = Arc::clone(&seen);
        let answer = Arc::clone(&answer);
        Box::pin(async move {
            let reply = answer(&payload);
            seen.lock().unwrap().push(payload);
            reply
        })
    })
}

#[tokio::test]
async fn optional_host_hooks_shape_the_tool_loop() {
    let model = pa_agent::types::Model {
        id: "m".into(),
        name: "m".into(),
        api: "test".into(),
        provider: "test".into(),
        base_url: "http://localhost".into(),
        reasoning: false,
        cost: pa_agent::types::UsageCost::default(),
        context_window: 100_000,
        max_tokens: 100,
        max_tokens_explicit: false,
    };
    let provider = Arc::new(ScriptedProvider::new(model.clone()));
    provider.push_tool_call_turn(
        Some("checking"),
        vec![("call-1", "echo", json!({ "text": "hi" }))],
    );
    provider.push_text_turn("all done");
    provider.push_text_turn("review applied");
    let requests: Requests = Arc::default();
    let capture = Arc::clone(&requests);
    let inner_stream = provider.stream_fn();
    let stream: pa_agent::stream::StreamFn = Arc::new(move |model, context, options| {
        capture.lock().unwrap().push((
            context.system_prompt.clone(),
            format!("{:?}", context.messages),
        ));
        inner_stream(model, context, options)
    });

    let starts: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
    let turns: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
    let projections: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
    let ends: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
    let hooks = SessionHostHooks {
        contract_v1: Some(Arc::new(std::sync::atomic::AtomicBool::new(true))),
        before_agent_start: Some(recording_hook(&starts, |_| {
            Some(json!({
                "systemPrompt": "HOST-REPLACED-PROMPT",
                "appendSystemPrompt": "HOST-APPENDED-RULE",
                "messages": [
                    {"customType": "start.anchor", "content": "START-ANCHOR", "display": false, "details": {"kind": "anchor"}},
                    {"customType": "start.note", "content": "START-NOTE", "display": true}
                ]
            }))
        })),
        turn_end: Some(recording_hook(&turns, |payload| {
            let after_tools = payload["toolResults"]
                .as_array()
                .is_some_and(|results| !results.is_empty());
            if after_tools {
                Some(json!({ "messages": ["HIDDEN-HOST-NOTE"] }))
            } else if payload["message"].to_string().contains("all done") {
                Some(
                    json!({ "continue": true, "messages": [{"customType":"advisor","content":"TERMINAL-REVIEW","details":{"kind":"advice"}}] }),
                )
            } else if payload["message"].to_string().contains("review applied") {
                Some(
                    json!({"messages": [{"customType": "deferred.advice", "content": "DEFERRED-REVIEW", "details": {"kind": "next-prompt"}}]}),
                )
            } else {
                None
            }
        })),
        // Keep the messages: the projection only proves it is asked.
        context: Some(recording_hook(&projections, |_| None)),
        message_end: Some(recording_hook(&ends, |_| None)),
        ..SessionHostHooks::default()
    };

    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().join("project");
    std::fs::create_dir_all(&cwd).unwrap();
    let engine = create_session(SessionEngineConfig {
        host_hooks: hooks,
        on_late_sent_agent_message: None,
        semantic_edges: None,
        cron_store: None,
        queued_steering_probe: None,
        should_stop_after_turn: None,
        image_model_router: None,
        steering_mode: None,
        follow_up_mode: None,
        cwd: cwd.clone(),
        agent_dir: tmp.path().join("agent"),
        mcp_manager: None,
        model: Some(model),
        thinking_level: None,
        stream_fn: Some(stream),
        tools: vec![bridge_tool(echo_definition())],
        custom_system_prompt: None,
        prompt_guidelines: vec![],
        generic_mcp_servers: vec![],
        allow_recursion: None,
        session_manager: None,
        extra_host_handlers: None,
        conversation_log_path: None,
        additional_skill_paths: vec![],
        additional_prompt_paths: vec![],
        extra_builtin_skill_overrides: vec![],
        rlm_subagent_host: None,
        rlm_usage_store: None,
        progress_note_emit: None,
        rlm_depth: None,
        factory_child: false,
        daemonless: true,
        telemetry: None,
        model_info: None,
        on_background_work_settled: None,
        bash_notice_wake: None,
        prewarm_ipython_kernel: None,
        queued_goal_context_purge: None,
    })
    .await
    .unwrap();

    engine
        .prompt("run the echo tool", PromptOptions::default())
        .await
        .unwrap();
    engine.session.agent().wait_for_idle().await;

    // before_agent_start: sees the prompt and the base system prompt; its
    // append reaches every request of the run.
    let starts = starts.lock().unwrap().clone();
    assert_eq!(starts.len(), 1, "{starts:?}");
    assert_eq!(starts[0]["prompt"], "run the echo tool");
    assert!(starts[0]["systemPrompt"]
        .as_str()
        .is_some_and(|prompt| prompt.starts_with("# prime-agent harness")
            && !prompt.contains("HOST-APPENDED-RULE")));
    let request_log = Arc::clone(&requests);
    let requests = requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 3, "{requests:?}");
    assert!(requests[2].1.contains("TERMINAL-REVIEW"));
    // The LLM conversion carries content; custom metadata remains in native history.
    for (system_prompt, _) in &requests {
        assert_eq!(system_prompt.as_deref(), Some("HOST-REPLACED-PROMPT"));
    }
    // The base prompt itself is untouched.
    assert!(!engine.system_prompt.contains("HOST-APPENDED-RULE"));

    // before_agent_start rows ride this admission after its user row, in order.
    let first = &requests[0].1;
    let user_at = first.find("run the echo tool").unwrap();
    let anchor_at = first
        .find("START-ANCHOR")
        .expect("anchor reaches the first request");
    let note_at = first
        .find("START-NOTE")
        .expect("note reaches the first request");
    assert!(user_at < anchor_at && anchor_at < note_at, "{first}");
    let start_entries = engine.session.entries().await;
    let saved = |custom_type: &str| {
        start_entries.iter().cloned().find_map(|entry| match entry {
            pa_types::session::FileEntry::Message {
                message: pa_types::session::AgentMessage::Custom(row),
                ..
            } if row.custom_type == custom_type => Some((row.display, row.details)),
            pa_types::session::FileEntry::CustomMessage { payload, .. }
                if payload.custom_type == custom_type =>
            {
                Some((payload.display, payload.details))
            }
            _ => None,
        })
    };
    let (anchor_display, anchor_details) = saved("start.anchor").expect("anchor is saved");
    assert!(!anchor_display);
    assert_eq!(anchor_details.unwrap()["kind"], "anchor");
    assert!(
        saved("start.note").expect("note is saved").0,
        "explicit display is kept"
    );

    // turn_end: the tool batch's hidden row reaches the next request and is
    // saved as a hidden custom row.
    let turns = turns.lock().unwrap().clone();
    assert_eq!(turns.len(), 3, "{turns:?}");
    assert_eq!(turns[0]["toolResults"][0]["toolName"], "echo");
    assert!(!requests[0].1.contains("HIDDEN-HOST-NOTE"));
    assert!(
        requests[1].1.contains("HIDDEN-HOST-NOTE"),
        "{}",
        requests[1].1
    );
    let entries = engine.session.entries().await;
    assert!(entries.iter().any(|entry| matches!(entry,
        pa_types::session::FileEntry::Message { message: pa_types::session::AgentMessage::Custom(row), .. }
            if row.custom_type == "advisor" && row.details.as_ref().is_some_and(|details| details["kind"] == "advice")
    ) || matches!(entry,
        pa_types::session::FileEntry::CustomMessage { payload, .. }
            if payload.custom_type == "advisor" && payload.details.as_ref().is_some_and(|details| details["kind"] == "advice")
    )), "advisor metadata remains canonical: {entries:?}");
    assert!(
        entries.iter().any(|entry| matches!(
            entry,
            pa_types::session::FileEntry::Message {
                message: pa_types::session::AgentMessage::Custom(row),
                ..
            } if row.custom_type == "host_hook" && !row.display
        ) || matches!(
            entry,
            pa_types::session::FileEntry::CustomMessage { payload, .. }
                if payload.custom_type == "host_hook" && !payload.display
        )),
        "the hidden row is saved: {entries:?}"
    );

    // context: every provider request is offered with its entry ids.
    let projections = projections.lock().unwrap().clone();
    assert!(projections.len() >= 2, "{projections:?}");
    assert!(projections
        .iter()
        .all(|projection| projection["purpose"] == "provider"));
    assert!(projections[0]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|message| message["role"] == "user"));
    assert!(projections[0]["entryRefs"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| entry["entryId"].is_string() && entry["messageIndex"].is_number()));

    // message_end: hears the user prompt, both assistant turns and the tool result.
    let ends = ends.lock().unwrap().clone();
    let roles: Vec<&str> = ends
        .iter()
        .filter_map(|end| end["message"]["role"].as_str())
        .collect();
    assert!(
        roles.contains(&"user") && roles.contains(&"assistant"),
        "{roles:?}"
    );
    assert!(roles.contains(&"toolResult"), "{roles:?}");
    // A terminal note without continue stays queued until a new native admission.
    assert!(!entries
        .iter()
        .any(|entry| format!("{entry:?}").contains("DEFERRED-REVIEW")));
    provider.push_text_turn("later user answer");
    engine
        .prompt("a later user request", PromptOptions::default())
        .await
        .unwrap();
    engine.session.agent().wait_for_idle().await;
    let later = request_log.lock().unwrap().clone();
    assert_eq!(
        later.len(),
        4,
        "no extra model call before the later admission"
    );
    assert!(
        later[3].1.contains("DEFERRED-REVIEW"),
        "the actual next provider request gets the deferred note"
    );
    let later_entries = engine.session.entries().await;
    assert!(later_entries.iter().any(|entry| matches!(entry,
        pa_types::session::FileEntry::Message {message: pa_types::session::AgentMessage::Custom(row), ..}
            if row.custom_type == "deferred.advice" && !row.display && row.details.as_ref().is_some_and(|details| details["kind"] == "next-prompt")
    ) || matches!(entry,
        pa_types::session::FileEntry::CustomMessage {payload, ..}
            if payload.custom_type == "deferred.advice" && !payload.display && payload.details.as_ref().is_some_and(|details| details["kind"] == "next-prompt")
    )), "deferred type/details persist when admitted");
}

/// A parallel-safe echo: "slow" waits until "fast" has finished, so a batch
/// that overlaps completes in reverse source order and a sequential one stalls.
fn ordered_echo(finished: Arc<std::sync::Mutex<Vec<String>>>) -> ToolDefinition {
    let fast_done = Arc::new(tokio::sync::Notify::new());
    ToolDefinition {
        name: "ordered_echo".to_string(),
        label: "Ordered echo".to_string(),
        description: "Echoes its input".to_string(),
        prompt_snippet: String::new(),
        parameters: json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }),
        execution_mode: None,
        prepare_arguments: None,
        execute: Arc::new(move |_id, params, _signal, _on_update| {
            let finished = Arc::clone(&finished);
            let fast_done = Arc::clone(&fast_done);
            let text = params["text"].as_str().unwrap_or_default().to_string();
            Box::pin(async move {
                if text == "slow" {
                    fast_done.notified().await;
                }
                finished.lock().unwrap().push(text.clone());
                if text == "fast" {
                    fast_done.notify_one();
                }
                Ok(ToolExecutionResult::text(format!("echo: {text}")))
            })
        }),
    }
}

#[tokio::test]
async fn turn_end_receives_parallel_exchanges_in_source_order() {
    let model = pa_agent::types::Model {
        id: "m".into(),
        name: "m".into(),
        api: "test".into(),
        provider: "test".into(),
        base_url: "http://localhost".into(),
        reasoning: false,
        cost: pa_agent::types::UsageCost::default(),
        context_window: 100_000,
        max_tokens: 100,
        max_tokens_explicit: false,
    };
    let provider = Arc::new(ScriptedProvider::new(model.clone()));
    // The first call waits for the second, so the batch finishes in reverse source order.
    provider.push_tool_call_turn(
        None,
        vec![
            ("call-slow", "ordered_echo", json!({ "text": "slow" })),
            ("call-fast", "ordered_echo", json!({ "text": "fast" })),
        ],
    );
    provider.push_text_turn("done");
    let stream = provider.stream_fn();
    let finished: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
    let turns: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
    let hooks = SessionHostHooks {
        contract_v1: Some(Arc::new(std::sync::atomic::AtomicBool::new(true))),
        turn_end: Some(recording_hook(&turns, |_| None)),
        ..SessionHostHooks::default()
    };

    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().join("project");
    std::fs::create_dir_all(&cwd).unwrap();
    let engine = create_session(SessionEngineConfig {
        host_hooks: hooks,
        on_late_sent_agent_message: None,
        semantic_edges: None,
        cron_store: None,
        queued_steering_probe: None,
        should_stop_after_turn: None,
        image_model_router: None,
        steering_mode: None,
        follow_up_mode: None,
        cwd: cwd.clone(),
        agent_dir: tmp.path().join("agent"),
        mcp_manager: None,
        model: Some(model),
        thinking_level: None,
        stream_fn: Some(stream),
        tools: vec![bridge_tool(ordered_echo(Arc::clone(&finished)))],
        custom_system_prompt: None,
        prompt_guidelines: vec![],
        generic_mcp_servers: vec![],
        allow_recursion: None,
        session_manager: None,
        extra_host_handlers: None,
        conversation_log_path: None,
        additional_skill_paths: vec![],
        additional_prompt_paths: vec![],
        extra_builtin_skill_overrides: vec![],
        rlm_subagent_host: None,
        rlm_usage_store: None,
        progress_note_emit: None,
        rlm_depth: None,
        factory_child: false,
        daemonless: true,
        telemetry: None,
        model_info: None,
        on_background_work_settled: None,
        bash_notice_wake: None,
        prewarm_ipython_kernel: None,
        queued_goal_context_purge: None,
    })
    .await
    .unwrap();

    // A sequential batch would wait forever on "slow"; fail clearly instead.
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        engine
            .prompt("run both echoes", PromptOptions::default())
            .await
            .unwrap();
        engine.session.agent().wait_for_idle().await;
    })
    .await
    .expect("the batch ran its calls concurrently");

    assert_eq!(*finished.lock().unwrap(), ["fast", "slow"]);
    let turns = turns.lock().unwrap().clone();
    let batch = turns
        .iter()
        .find(|turn| {
            turn["exchanges"]
                .as_array()
                .is_some_and(|items| !items.is_empty())
        })
        .expect("turn_end for the tool batch");
    assert_eq!(batch["toolExecution"], "parallel");
    let exchanges = batch["exchanges"].as_array().unwrap();
    let order: Vec<(u64, &str, &str)> = exchanges
        .iter()
        .map(|exchange| {
            (
                exchange["sourceOrder"].as_u64().unwrap(),
                exchange["toolCallId"].as_str().unwrap(),
                exchange["result"]["toolCallId"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        order,
        [(0, "call-slow", "call-slow"), (1, "call-fast", "call-fast")]
    );
    assert_eq!(exchanges[0]["originalInput"]["text"], "slow");
    assert_eq!(exchanges[1]["executedInput"]["text"], "fast");
    assert!(exchanges[0]["result"]["content"]
        .to_string()
        .contains("echo: slow"));
    assert!(exchanges[1]["result"]["content"]
        .to_string()
        .contains("echo: fast"));
    let results: Vec<&str> = batch["toolResults"]
        .as_array()
        .unwrap()
        .iter()
        .map(|result| result["toolCallId"].as_str().unwrap())
        .collect();
    assert_eq!(
        results,
        ["call-slow", "call-fast"],
        "results keep source order"
    );
}
