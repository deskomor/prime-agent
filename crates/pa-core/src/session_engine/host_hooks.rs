//! Hooks a host (the daemon worker, and through it an ACP client) installs
//! into a session's engine. Every field defaults to `None`, which keeps the
//! session exactly as without a host.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

/// Decides one kernel host call (`rlm.spawn`, `agent_message`, ...) before
/// its handler runs: `Ok(())` lets it run, an error refuses it and becomes
/// the error the Python caller sees. The gate enforces a rule, so a host
/// that cannot answer must return an error.
pub type HostCallGateFn = Arc<
    dyn Fn(String, Value) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>> + Send + Sync,
>;

/// One optional host hook: the engine sends the hook point's payload and
/// reads the answer, or `None` when the host gave no usable answer (the hook
/// point then continues as if no hook were installed).
pub type HostHookFn =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Option<Value>> + Send>> + Send + Sync>;

/// The host hooks one session build installs.
#[derive(Clone, Default)]
pub struct SessionHostHooks {
    /// Awaited after a tool call's arguments validate and before it runs;
    /// a `block` answer turns the call into an error tool result carrying
    /// the reason.
    pub before_tool_call: Option<pa_agent::agent_loop::BeforeToolCallFn>,
    /// Awaited after a tool call runs; a returned override replaces the
    /// result's fields.
    pub after_tool_call: Option<pa_agent::agent_loop::AfterToolCallFn>,
    /// Awaited before every kernel host call's handler.
    pub host_call_gate: Option<HostCallGateFn>,
    /// Before a prompt starts a run (and before its row is saved): sees
    /// `{prompt, systemPrompt, imageCount}`, may answer
    /// `{appendSystemPrompt}` for that run.
    pub before_agent_start: Option<HostHookFn>,
    /// After each turn: sees `{message, toolResults}`, may answer
    /// `{messages: [...]}` (hidden custom rows saved and carried into the
    /// next turn).
    pub turn_end: Option<HostHookFn>,
    /// Before a compaction summarizes: sees the attempt (messages to
    /// summarize, tokens, first kept entry), may answer `{cancel: true}` or
    /// `{summary}` (used in place of the summarizer calls).
    pub session_before_compact: Option<HostHookFn>,
    /// Projects the messages one consumer reads: sees `{purpose, messages:
    /// [{entryId, message}]}` (`purpose` is `provider`, `budget`,
    /// `compaction`, `branch_summary` or `refine`;
    /// `entryId` is `null` where no single entry backs a message), may
    /// answer `{messages: [...]}` to replace them.
    pub context: Option<HostHookFn>,
    /// After each message ends: sees `{message}`; the answer is ignored.
    pub message_end: Option<HostHookFn>,
    /// While set, the session runs no auto-refinement of its own.
    pub auto_refine_paused: Option<Arc<std::sync::atomic::AtomicBool>>,
}

/// The hidden custom rows a `turn_end` answer carries (`{messages: [...]}`;
/// each item a string or `{content, customType?, details?}`). A malformed
/// item is skipped: the hook is optional.
#[must_use]
pub fn hidden_messages(answer: &Value) -> Vec<pa_types::session::CustomMessage> {
    let Some(items) = answer.get("messages").and_then(Value::as_array) else {
        return Vec::new();
    };
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        });
    items
        .iter()
        .filter_map(|item| {
            let (content, custom_type, details) = match item {
                Value::String(text) => (Value::String(text.clone()), None, None),
                Value::Object(object) => (
                    object.get("content")?.clone(),
                    object.get("customType").and_then(Value::as_str),
                    object
                        .get("details")
                        .filter(|details| !details.is_null())
                        .cloned(),
                ),
                _ => return None,
            };
            serde_json::from_value::<pa_types::session::CustomMessage>(serde_json::json!({
                "customType": custom_type.unwrap_or(HOST_HOOK_CUSTOM_TYPE),
                "content": content,
                "display": false,
                "details": details,
                "timestamp": timestamp,
            }))
            .ok()
        })
        .collect()
}

/// The custom type of a hidden row a host hook added without naming one.
pub const HOST_HOOK_CUSTOM_TYPE: &str = "host_hook";

/// Ask the `context` hook to project `messages` for `purpose`. `entry_ids`
/// pair with the messages index for index when their counts match. `None`
/// keeps the messages (no answer, or an answer that does not parse).
pub async fn project_messages<M>(
    hook: &HostHookFn,
    purpose: &str,
    messages: &[M],
    entry_ids: &[Option<String>],
) -> Option<Vec<M>>
where
    M: serde::Serialize + serde::de::DeserializeOwned,
{
    let paired = entry_ids.len() == messages.len();
    let items: Vec<Value> = messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            serde_json::json!({
                "entryId": if paired { entry_ids[index].clone() } else { None },
                "message": message,
            })
        })
        .collect();
    let answer = hook(serde_json::json!({ "purpose": purpose, "messages": items })).await?;
    serde_json::from_value(answer.get("messages")?.clone()).ok()
}

/// The provider-request projection over the `context` hook: the session's
/// active context supplies the entry ids.
#[must_use]
pub fn provider_projection(
    hook: HostHookFn,
    session: Arc<tokio::sync::Mutex<crate::session::manager::SessionManager>>,
) -> pa_agent::agent_loop::TransformContextFn {
    Arc::new(move |messages, _signal| {
        let hook = Arc::clone(&hook);
        let session = Arc::clone(&session);
        Box::pin(async move {
            let entry_ids = session.lock().await.active_context().entry_ids;
            Ok(project_messages(&hook, "provider", &messages, &entry_ids)
                .await
                .unwrap_or(messages))
        })
    })
}
