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
        before_agent_start: Some(recording_hook(&starts, |_| {
            Some(json!({ "appendSystemPrompt": "HOST-APPENDED-RULE" }))
        })),
        turn_end: Some(recording_hook(&turns, |payload| {
            let after_tools = payload["toolResults"]
                .as_array()
                .is_some_and(|results| !results.is_empty());
            after_tools.then(|| json!({ "messages": ["HIDDEN-HOST-NOTE"] }))
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
    let requests = requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 2, "{requests:?}");
    for (system_prompt, _) in &requests {
        assert!(system_prompt
            .as_deref()
            .is_some_and(|prompt| prompt.ends_with("\n\nHOST-APPENDED-RULE")));
    }
    // The base prompt itself is untouched.
    assert!(!engine.system_prompt.contains("HOST-APPENDED-RULE"));

    // turn_end: the tool batch's hidden row reaches the next request and is
    // saved as a hidden custom row.
    let turns = turns.lock().unwrap().clone();
    assert_eq!(turns.len(), 2, "{turns:?}");
    assert_eq!(turns[0]["toolResults"][0]["toolName"], "echo");
    assert!(!requests[0].1.contains("HIDDEN-HOST-NOTE"));
    assert!(
        requests[1].1.contains("HIDDEN-HOST-NOTE"),
        "{}",
        requests[1].1
    );
    let entries = engine.session.entries().await;
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
        .any(|item| item["entryId"].is_string()));

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
}
