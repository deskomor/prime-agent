//! Hooks a host (the daemon worker, and through it an ACP client) installs
//! into a session's engine. Every field defaults to `None`, which keeps the
//! session exactly as without a host.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

/// Decides one kernel host call (`rlm.spawn`, `agent_message`, ...) before
/// its handler runs: `Ok(None)` lets it run without a scope restriction;
/// `Ok(Some(ids))` restricts family requests to those durable session IDs.
/// An error refuses the request. A host that cannot answer must return an error.
pub type HostCallGateFn = Arc<
    dyn Fn(
            String,
            Value,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<Option<Vec<String>>>> + Send>>
        + Send
        + Sync,
>;

/// One optional host hook: the engine sends the hook point's payload and
/// reads the answer, or `None` when the host gave no usable answer (the hook
/// point then continues as if no hook were installed).
pub type HostHookFn =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Option<Value>> + Send>> + Send + Sync>;

/// The host hooks one session build installs.
#[derive(Clone, Default)]
pub struct SessionHostHooks {
    /// Negotiated host contract v1; unset keeps the original hook semantics.
    pub contract_v1: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Changes when the attached optional host is replaced or detached.
    pub contract_epoch: Arc<std::sync::atomic::AtomicU64>,
    /// Applied context projections, shared by provider, budget and snapshot readers.
    pub context_projection: Arc<std::sync::Mutex<ContextProjectionState>>,
    /// Awaited native lifecycle event; payload.type names the event.
    pub lifecycle: Option<HostHookFn>,
    /// Before idle prompt normalization: `{cwd}` may return `{skillPaths: [...]}`.
    /// Paths are loaded by native discovery only while contract v1 is active.
    pub resources_discover: Option<HostHookFn>,
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
    /// Projects one consumer's messages. Legacy requests wrap each message as
    /// `{entryId, message}`. Contract v1 sends raw `messages` plus `entryRefs`
    /// (`{messageIndex, entryId}` for backed messages) and accepts replacement
    /// messages, refs and an optional string `projectionIdentity`. Purposes are
    /// `provider`, `budget`, `compaction`, `branch-summary` and `refine` (legacy
    /// spells branch summary `branch_summary`).
    pub context: Option<HostHookFn>,
    /// After each message ends: sees `{message}`; the answer is ignored.
    pub message_end: Option<HostHookFn>,
    /// While set, the session runs no auto-refinement of its own.
    pub auto_refine_paused: Option<Arc<std::sync::atomic::AtomicBool>>,
}

/// A durable entry backing one message in a projected context.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextEntryRef {
    pub message_index: usize,
    pub entry_id: String,
}

/// The exact representation applied to one context consumer.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextProjection<M> {
    pub messages: Vec<M>,
    pub entry_refs: Vec<ContextEntryRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projection_identity: Option<String>,
}

/// Latest applied representations for the five supported context purposes.
#[derive(Default)]
pub struct ContextProjectionState {
    snapshots: std::collections::BTreeMap<String, Value>,
    provider_source: Option<Value>,
}

impl SessionHostHooks {
    /// Whether the saved provider projection used exactly these messages and refs.
    #[must_use]
    pub fn projection_source_matches(&self, source: &Value) -> bool {
        self.context_projection
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .provider_source
            .as_ref()
            == Some(source)
    }

    /// The latest exact applied projection for `purpose`, including entry refs and identity.
    #[must_use]
    pub fn projection_snapshot(&self, purpose: &str) -> Option<Value> {
        self.context_projection
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .snapshots
            .get(purpose)
            .cloned()
    }

    /// Apply the negotiated context hook without changing the stored transcript.
    pub async fn project_context<M>(
        &self,
        purpose: &str,
        messages: &[M],
        entry_ids: &[Option<String>],
    ) -> Option<ContextProjection<M>>
    where
        M: serde::Serialize + serde::de::DeserializeOwned,
    {
        let hook = self.context.as_ref()?;
        let v1 = self
            .contract_v1
            .as_ref()
            .is_some_and(|enabled| enabled.load(std::sync::atomic::Ordering::SeqCst));
        if !v1 {
            return project_messages(hook, purpose, messages, entry_ids)
                .await
                .map(|messages| ContextProjection {
                    messages,
                    entry_refs: Vec::new(),
                    projection_identity: None,
                });
        }

        let purpose = if purpose == "branch_summary" {
            "branch-summary"
        } else {
            purpose
        };
        let original = serde_json::to_value(messages).ok()?;
        let entry_refs: Vec<ContextEntryRef> = if entry_ids.len() == messages.len() {
            entry_ids
                .iter()
                .enumerate()
                .filter_map(|(message_index, id)| {
                    id.as_ref().map(|entry_id| ContextEntryRef {
                        message_index,
                        entry_id: entry_id.clone(),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        let source = serde_json::json!({ "messages": original, "entryRefs": entry_refs });
        if purpose == "budget" {
            let cached = {
                let state = self
                    .context_projection
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                (state.provider_source.as_ref() == Some(&source))
                    .then(|| state.snapshots.get("provider").cloned())
                    .flatten()
            };
            let cached = cached.and_then(|cached| {
                let messages = serde_json::from_value(cached["messages"].clone()).ok()?;
                let entry_refs = serde_json::from_value(cached["entryRefs"].clone()).ok()?;
                Some((cached, messages, entry_refs))
            });
            if let Some((cached, messages, entry_refs)) = cached {
                let projection = ContextProjection {
                    messages,
                    entry_refs,
                    projection_identity: cached
                        .get("projectionIdentity")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                };
                self.context_projection
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .snapshots
                    .insert(purpose.to_owned(), cached);
                return Some(projection);
            }
        }
        let answer = hook(serde_json::json!({
            "purpose": purpose,
            "messages": original,
            "entryRefs": entry_refs,
        }))
        .await;
        let projected = answer
            .as_ref()
            .and_then(|answer| answer.get("messages"))
            .unwrap_or(&original);
        let parsed = serde_json::from_value::<Vec<M>>(projected.clone());
        let (projected, projected_messages, answer) = match parsed {
            Ok(messages) => (projected, messages, answer.as_ref()),
            Err(_) => (
                &original,
                serde_json::from_value(original.clone()).ok()?,
                None,
            ),
        };
        let refs = answer
            .and_then(|answer| answer.get("entryRefs"))
            .and_then(|refs| serde_json::from_value::<Vec<ContextEntryRef>>(refs.clone()).ok())
            .filter(|refs| {
                refs.iter()
                    .all(|entry| entry.message_index < projected_messages.len())
            });
        let projection = ContextProjection {
            messages: projected_messages,
            entry_refs: refs.unwrap_or_else(|| {
                if projected == &original {
                    entry_refs
                } else {
                    Vec::new()
                }
            }),
            projection_identity: answer
                .and_then(|answer| answer.get("projectionIdentity"))
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        if [
            "provider",
            "budget",
            "compaction",
            "branch-summary",
            "refine",
        ]
        .contains(&purpose)
        {
            let mut state = self
                .context_projection
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state
                .snapshots
                .insert(purpose.to_owned(), serde_json::to_value(&projection).ok()?);
            if purpose == "provider" {
                state.provider_source = Some(source);
            }
        }
        Some(projection)
    }
}

/// The hidden custom rows a `turn_end` answer carries (`{messages: [...]}`;
/// each item a string or `{content, customType?, details?}`). A malformed
/// item is skipped: the hook is optional.
#[must_use]
pub fn hidden_messages(answer: &Value) -> Vec<pa_types::session::CustomMessage> {
    custom_rows(answer, false)
}

/// The custom rows a `before_agent_start` answer adds to its own admission
/// (`{messages: [...]}`, same item shapes as [`hidden_messages`]). Each item
/// keeps an explicit boolean `display`; hidden is the default.
#[must_use]
pub fn admission_messages(answer: &Value) -> Vec<pa_types::session::CustomMessage> {
    custom_rows(answer, true)
}

fn custom_rows(answer: &Value, keep_display: bool) -> Vec<pa_types::session::CustomMessage> {
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
            let (content, custom_type, details, display) = match item {
                Value::String(text) => (Value::String(text.clone()), None, None, false),
                Value::Object(object) => (
                    object.get("content")?.clone(),
                    object.get("customType").and_then(Value::as_str),
                    object
                        .get("details")
                        .filter(|details| !details.is_null())
                        .cloned(),
                    keep_display && object.get("display").and_then(Value::as_bool) == Some(true),
                ),
                _ => return None,
            };
            serde_json::from_value::<pa_types::session::CustomMessage>(serde_json::json!({
                "customType": custom_type.unwrap_or(HOST_HOOK_CUSTOM_TYPE),
                "content": content,
                "display": display,
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
    hooks: SessionHostHooks,
    session: Arc<tokio::sync::Mutex<crate::session::manager::SessionManager>>,
) -> pa_agent::agent_loop::TransformContextFn {
    Arc::new(move |messages, _signal| {
        let hooks = hooks.clone();
        let session = Arc::clone(&session);
        Box::pin(async move {
            let entry_ids = session.lock().await.active_context().entry_ids;
            Ok(hooks
                .project_context("provider", &messages, &entry_ids)
                .await
                .map_or(messages, |projection| projection.messages))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn context_v1_keeps_applied_custom_and_bash_refs_and_identity() {
        let messages: Vec<pa_types::session::AgentMessage> = serde_json::from_value(json!([
            {"role":"user", "content":"prompt", "timestamp":0},
            {"role":"custom", "customType":"advisor", "content":"note", "display":false, "timestamp":1},
            {"role":"bashExecution", "command":"pwd", "output":"/project", "exitCode":0,
             "cancelled":false, "truncated":false, "timestamp":2}
        ])).unwrap();
        let raw = serde_json::to_value(&messages).unwrap();
        let applied = json!({
            "messages":[raw[2].clone(),raw[1].clone()],
            "entryRefs":[{"messageIndex":0,"entryId":"bash"},{"messageIndex":1,"entryId":"note"}],
            "projectionIdentity":"projection-1"
        });
        let hook: HostHookFn = Arc::new({
            let raw = raw.clone();
            let applied = applied.clone();
            move |payload| {
                let raw = raw.clone();
                let applied = applied.clone();
                Box::pin(async move {
                    assert_eq!(
                        payload,
                        json!({
                            "purpose":"provider", "messages":raw,
                            "entryRefs":[{"messageIndex":1,"entryId":"note"},{"messageIndex":2,"entryId":"bash"}]
                        })
                    );
                    Some(applied)
                })
            }
        });
        let hooks = SessionHostHooks {
            contract_v1: Some(Arc::new(std::sync::atomic::AtomicBool::new(true))),
            context: Some(hook),
            ..Default::default()
        };
        let entry_ids = vec![None, Some("note".into()), Some("bash".into())];
        for purpose in ["provider", "budget"] {
            let projection = hooks
                .project_context(purpose, &messages, &entry_ids)
                .await
                .unwrap();
            assert_eq!(serde_json::to_value(&projection).unwrap(), applied);
            assert_eq!(hooks.projection_snapshot(purpose), Some(applied.clone()));
        }
    }

    #[tokio::test]
    async fn invalid_context_replacement_does_not_apply_its_refs_or_identity() {
        let messages = vec![pa_agent::types::AgentMessage::user("original")];
        let hooks = SessionHostHooks {
            contract_v1: Some(Arc::new(std::sync::atomic::AtomicBool::new(true))),
            context: Some(Arc::new(|payload| {
                Box::pin(async move {
                    assert_eq!(payload["purpose"], "branch-summary");
                    Some(json!({
                        "messages": "not a message list",
                        "entryRefs":[{"messageIndex":0,"entryId":"wrong"}],
                        "projectionIdentity":"unapplied"
                    }))
                })
            })),
            ..Default::default()
        };
        let projection = hooks
            .project_context("branch_summary", &messages, &[Some("original".into())])
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(projection).unwrap(),
            json!({
                "messages":messages,
                "entryRefs":[{"messageIndex":0,"entryId":"original"}]
            })
        );
    }
}
