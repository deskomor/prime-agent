//! Awaited host hooks: the worker side of the ACP
//! `_ai.primeintellect.prime-agent/hook` request.
//!
//! A client turns hooks on per session with `set_host_hooks`. From then on
//! each enabled hook point emits a `host_hook_request` session event and waits
//! for `respond_host_hook` with a `result` or an `error`. Hooks come in two
//! kinds, and the kind decides what happens without a usable answer (an
//! error, the timeout, an abort, the hooks switching off, or the worker
//! going away):
//!
//! - Enforcement hooks (`host_call`) guard a rule, so they refuse the call.
//! - Optional hooks (`after_tool_call`, `before_agent_start`, `turn_end`,
//!   `session_before_compact`, `session_before_tree`, `context`,
//!   `message_end`, `user_bash_end`) add context or listen, so the session
//!   continues as if the hook were not there.
//!
//! With no hooks enabled (the default) every hook point answers at once and
//! the session behaves exactly as without a host.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::anyhow;
use pa_agent::abort::AbortSignal;
use pa_agent::agent_loop::AfterToolCallFn;
use pa_agent::types::{AfterToolCallResult, ToolResultContent};
use pa_core::session_engine::host_hooks::HostCallGateFn;
use serde_json::{json, Value};
use tokio::sync::oneshot;

/// The tool-result override hook (optional).
pub(crate) const AFTER_TOOL_CALL: &str = "after_tool_call";
/// The kernel host-call gate (enforcement).
pub(crate) const HOST_CALL: &str = "host_call";
/// The system-prompt addition before a prompt's run (optional).
pub(crate) const BEFORE_AGENT_START: &str = "before_agent_start";
/// Hidden rows after each turn (optional).
pub(crate) const TURN_END: &str = "turn_end";
/// Cancel or replace a compaction (optional).
pub(crate) const SESSION_BEFORE_COMPACT: &str = "session_before_compact";
/// Cancel or replace a tree navigation's branch summary (optional).
pub(crate) const SESSION_BEFORE_TREE: &str = "session_before_tree";
/// Project the messages a consumer reads, by purpose (optional).
pub(crate) const CONTEXT: &str = "context";
/// Hear each finished message (optional, notification).
pub(crate) const MESSAGE_END: &str = "message_end";
/// Hear each finished user `!` bash run (optional, notification).
pub(crate) const USER_BASH_END: &str = "user_bash_end";
/// Every hook a client may enable.
pub(crate) const KNOWN_HOOKS: [&str; 9] = [
    AFTER_TOOL_CALL,
    HOST_CALL,
    BEFORE_AGENT_START,
    TURN_END,
    SESSION_BEFORE_COMPACT,
    SESSION_BEFORE_TREE,
    CONTEXT,
    MESSAGE_END,
    USER_BASH_END,
];

/// How long one hook waits for its answer unless the client sets another.
pub(crate) const HOST_HOOK_DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Emits one session event to the worker's attached clients.
pub(crate) type HostHookEmit = Arc<dyn Fn(Value) + Send + Sync>;

/// A client's answer: the hook's result, or why there is none.
type HookAnswer = Result<Value, String>;

/// The hooks one session has on.
#[derive(Clone)]
struct HookConfig {
    enabled: HashSet<String>,
    /// The host-call types the gate asks about (`rlm.*` matches a prefix);
    /// `None` asks about every type.
    host_call_types: Option<Vec<String>>,
    timeout: Duration,
}

impl Default for HookConfig {
    fn default() -> Self {
        Self {
            enabled: HashSet::new(),
            host_call_types: None,
            timeout: HOST_HOOK_DEFAULT_TIMEOUT,
        }
    }
}

#[derive(Default)]
struct BrokerState {
    config: HookConfig,
    pending: HashMap<String, oneshot::Sender<HookAnswer>>,
}

/// The session's hook broker: the engine's hook points await it, the worker
/// commands answer it.
pub(crate) struct HostHookBroker {
    state: Mutex<BrokerState>,
    emit: HostHookEmit,
    /// `set_auto_refine`: while set, the session runs no auto-refinement.
    auto_refine_paused: Arc<std::sync::atomic::AtomicBool>,
}

impl HostHookBroker {
    pub(crate) fn new(emit: HostHookEmit) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(BrokerState::default()),
            emit,
            auto_refine_paused: Arc::default(),
        })
    }

    /// The shared pause flag the session's auto-refine trigger reads.
    pub(crate) fn auto_refine_paused(&self) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::clone(&self.auto_refine_paused)
    }

    /// Pause or resume the session's own auto-refinement.
    pub(crate) fn set_auto_refine_paused(&self, paused: bool) {
        self.auto_refine_paused
            .store(paused, std::sync::atomic::Ordering::SeqCst);
    }

    /// Replace the session's hooks. An unknown hook name fails the whole
    /// call; waiting hooks of the old set get no answer (each settles by its
    /// kind's rule).
    ///
    /// # Errors
    ///
    /// Returns the message for an unknown hook name.
    ///
    /// # Panics
    ///
    /// Panics when the broker lock is poisoned.
    pub(crate) fn set_hooks(
        &self,
        hooks: &[String],
        host_call_types: Option<Vec<String>>,
        timeout: Option<Duration>,
    ) -> Result<(), String> {
        if let Some(unknown) = hooks
            .iter()
            .find(|hook| !KNOWN_HOOKS.contains(&hook.as_str()))
        {
            return Err(format!(
                "Unknown host hook \"{unknown}\". Known hooks: {}",
                KNOWN_HOOKS.join(", ")
            ));
        }
        let mut state = self.state.lock().expect("host hook lock");
        state.config = HookConfig {
            enabled: hooks.iter().cloned().collect(),
            host_call_types,
            timeout: timeout.unwrap_or(HOST_HOOK_DEFAULT_TIMEOUT),
        };
        for (_, answer) in state.pending.drain() {
            let _ = answer.send(Err("the host hooks changed".to_string()));
        }
        Ok(())
    }

    /// Answer one waiting hook; `false` when no hook waits with that id.
    ///
    /// # Panics
    ///
    /// Panics when the broker lock is poisoned.
    pub(crate) fn resolve(&self, request_id: &str, answer: HookAnswer) -> bool {
        let Some(sender) = self
            .state
            .lock()
            .expect("host hook lock")
            .pending
            .remove(request_id)
        else {
            return false;
        };
        let _ = sender.send(answer);
        true
    }

    /// Ask the client one hook. `None` when the hook is off; otherwise the
    /// answer, or the reason there is none.
    async fn call(
        &self,
        hook: &str,
        payload: Value,
        signal: Option<AbortSignal>,
    ) -> Option<HookAnswer> {
        let (request_id, receiver, timeout) = {
            let mut state = self.state.lock().expect("host hook lock");
            if !state.config.enabled.contains(hook) {
                return None;
            }
            let request_id = uuid::Uuid::new_v4().to_string();
            let (sender, receiver) = oneshot::channel();
            state.pending.insert(request_id.clone(), sender);
            (request_id, receiver, state.config.timeout)
        };
        (self.emit)(json!({
            "type": "host_hook_request",
            "requestId": request_id,
            "hook": hook,
            "payload": payload,
        }));
        let answer = tokio::select! {
            answer = receiver => answer.unwrap_or_else(|_| Err("the worker dropped the hook".to_string())),
            () = tokio::time::sleep(timeout) => Err(format!("no answer within {} ms", timeout.as_millis())),
            () = aborted(signal) => Err("the turn was aborted".to_string()),
        };
        self.state
            .lock()
            .expect("host hook lock")
            .pending
            .remove(&request_id);
        (self.emit)(json!({
            "type": "host_hook_resolved",
            "requestId": request_id,
            "hook": hook,
            "outcome": if answer.is_ok() { "answered" } else { "failed" },
        }));
        Some(answer)
    }

    /// Ask one optional hook: its answer, or `None` when the hook is off or
    /// gave no usable answer (the caller continues as if it were absent).
    pub(crate) async fn optional(&self, hook: &str, payload: Value) -> Option<Value> {
        self.call(hook, payload, None).await?.ok()
    }

    /// An optional hook as the engine's hook seam.
    pub(crate) fn optional_hook(
        self: &Arc<Self>,
        hook: &'static str,
    ) -> pa_core::session_engine::host_hooks::HostHookFn {
        let broker = Arc::clone(self);
        Arc::new(move |payload| {
            let broker = Arc::clone(&broker);
            Box::pin(async move { broker.optional(hook, payload).await })
        })
    }

    fn gates_host_call(&self, request_type: &str) -> bool {
        let state = self.state.lock().expect("host hook lock");
        state.config.enabled.contains(HOST_CALL)
            && state.config.host_call_types.as_ref().is_none_or(|types| {
                types.iter().any(|pattern| match pattern.strip_suffix('*') {
                    Some(prefix) => request_type.starts_with(prefix),
                    None => request_type == pattern,
                })
            })
    }

    /// The engine's `after_tool_call` hook point (optional: no usable
    /// answer keeps the executed result).
    pub(crate) fn after_tool_call(self: &Arc<Self>) -> AfterToolCallFn {
        let broker = Arc::clone(self);
        Arc::new(move |context, signal| {
            let broker = Arc::clone(&broker);
            Box::pin(async move {
                let payload = json!({
                    "toolCallId": context.tool_call.id,
                    "toolName": context.tool_call.name,
                    "args": context.args,
                    "result": context.result,
                    "isError": context.is_error,
                });
                Ok(
                    match broker.call(AFTER_TOOL_CALL, payload, Some(signal)).await {
                        Some(Ok(answer)) => tool_result_override(&answer),
                        Some(Err(_)) | None => None,
                    },
                )
            })
        })
    }

    /// The kernel host-call gate (enforcement: no usable answer refuses the
    /// call). Types outside the client's `hostCallTypes` pass unasked.
    pub(crate) fn host_call_gate(self: &Arc<Self>) -> HostCallGateFn {
        let broker = Arc::clone(self);
        Arc::new(move |request_type, data| {
            let broker = Arc::clone(&broker);
            Box::pin(async move {
                if !broker.gates_host_call(&request_type) {
                    return Ok(());
                }
                let payload = json!({ "type": request_type, "data": data });
                match broker.call(HOST_CALL, payload, None).await {
                    None => Ok(()),
                    Some(Ok(answer)) => host_call_verdict(&request_type, &answer),
                    Some(Err(reason)) => Err(anyhow!(
                        "The host could not check {request_type} ({reason}), so it was refused."
                    )),
                }
            })
        })
    }
}

/// The override an `after_tool_call` answer asks for: `null` or `{}` keeps
/// the result, and so does an answer of the wrong shape (the hook is
/// optional). Each present field replaces the executed one.
fn tool_result_override(answer: &Value) -> Option<AfterToolCallResult> {
    let object = answer.as_object()?;
    let content = match object.get("content") {
        None | Some(Value::Null) => None,
        Some(content) => {
            Some(serde_json::from_value::<Vec<ToolResultContent>>(content.clone()).ok()?)
        }
    };
    let flag = |key: &str| object.get(key).and_then(Value::as_bool);
    let result = AfterToolCallResult {
        content,
        details: object
            .get("details")
            .filter(|details| !details.is_null())
            .cloned(),
        is_error: flag("isError"),
        terminate: flag("terminate"),
    };
    let changes = result.content.is_some()
        || result.details.is_some()
        || result.is_error.is_some()
        || result.terminate.is_some();
    changes.then_some(result)
}

/// A host-call answer: only `{"allow": true}` lets the call run; anything
/// else refuses it, with the answer's `reason` when it gives one.
fn host_call_verdict(request_type: &str, answer: &Value) -> anyhow::Result<()> {
    if answer.get("allow").and_then(Value::as_bool) == Some(true) {
        return Ok(());
    }
    match answer.get("reason").and_then(Value::as_str) {
        Some(reason) if !reason.is_empty() => Err(anyhow!("{reason}")),
        _ => Err(anyhow!("The host refused {request_type}.")),
    }
}

/// Resolves when the turn aborts; never without a signal.
async fn aborted(signal: Option<AbortSignal>) {
    match signal {
        Some(signal) => signal.aborted().await,
        None => std::future::pending().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pa_agent::types::{
        AfterToolCallContext, AgentContext, AgentToolResult, AssistantMessage, ToolCall,
    };

    fn recording_broker() -> (Arc<HostHookBroker>, Arc<Mutex<Vec<Value>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let broker = HostHookBroker::new(Arc::new(move |event| sink.lock().unwrap().push(event)));
        (broker, events)
    }

    async fn requested_id(events: &Arc<Mutex<Vec<Value>>>) -> (String, Value) {
        loop {
            if let Some(found) = events.lock().unwrap().iter().find_map(|event| {
                (event["type"] == "host_hook_request").then(|| {
                    (
                        event["requestId"].as_str().unwrap().to_string(),
                        event.clone(),
                    )
                })
            }) {
                return found;
            }
            tokio::task::yield_now().await;
        }
    }

    fn after_context() -> AfterToolCallContext {
        AfterToolCallContext {
            assistant_message: serde_json::from_value::<AssistantMessage>(json!({
                "content": [],
                "usage": {},
                "stopReason": "toolUse",
            }))
            .unwrap(),
            tool_call: ToolCall {
                id: "call-1".to_string(),
                name: "python_repl".to_string(),
                arguments: json!({ "code": "1" }),
                thought_signature: None,
            },
            args: json!({ "code": "1" }),
            result: AgentToolResult::text("1"),
            is_error: false,
            context: AgentContext::default(),
        }
    }

    #[tokio::test]
    async fn hooks_off_answer_at_once_without_an_event() {
        let (broker, events) = recording_broker();
        let after = broker.after_tool_call()(after_context(), AbortSignal::default())
            .await
            .unwrap();
        assert!(after.is_none());
        broker.host_call_gate()("rlm.spawn".to_string(), json!({}))
            .await
            .unwrap();
        assert!(events.lock().unwrap().is_empty());
        assert!(broker
            .set_hooks(&["before_everything".to_string()], None, None)
            .is_err());
    }

    #[tokio::test]
    async fn after_tool_call_replaces_the_result_and_errors_keep_it() {
        let (broker, events) = recording_broker();
        broker
            .set_hooks(&[AFTER_TOOL_CALL.to_string()], None, None)
            .unwrap();
        let hook = broker.after_tool_call();
        let call = tokio::spawn(hook(after_context(), AbortSignal::default()));
        let (id, event) = requested_id(&events).await;
        assert_eq!(event["hook"], AFTER_TOOL_CALL);
        assert_eq!(event["payload"]["toolName"], "python_repl");
        assert_eq!(event["payload"]["result"]["content"][0]["text"], "1");
        assert!(broker.resolve(
            &id,
            Ok(json!({ "content": [{ "type": "text", "text": "rewritten" }] }))
        ));
        let replaced = call.await.unwrap().unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(replaced.content.unwrap()).unwrap(),
            json!([{ "type": "text", "text": "rewritten" }])
        );

        events.lock().unwrap().clear();
        let call = tokio::spawn(broker.after_tool_call()(
            after_context(),
            AbortSignal::default(),
        ));
        let (id, _) = requested_id(&events).await;
        broker.resolve(&id, Err("studio crashed".to_string()));
        assert!(
            call.await.unwrap().unwrap().is_none(),
            "an error keeps the result"
        );
    }

    #[tokio::test]
    async fn the_host_call_gate_denies_without_an_allow() {
        let (broker, events) = recording_broker();
        broker
            .set_hooks(
                &[HOST_CALL.to_string()],
                Some(vec!["rlm.*".to_string()]),
                Some(Duration::from_millis(20)),
            )
            .unwrap();
        let gate = broker.host_call_gate();
        // Outside the filter: runs unasked.
        gate("goal.complete".to_string(), json!({})).await.unwrap();
        assert!(events.lock().unwrap().is_empty());

        let call = tokio::spawn(gate("rlm.create_session".to_string(), json!({ "x": 1 })));
        let (id, event) = requested_id(&events).await;
        assert_eq!(event["payload"]["type"], "rlm.create_session");
        broker.resolve(&id, Ok(json!({ "allow": true })));
        call.await.unwrap().unwrap();

        events.lock().unwrap().clear();
        let call = tokio::spawn(gate("rlm.spawn".to_string(), json!({})));
        let (id, _) = requested_id(&events).await;
        broker.resolve(
            &id,
            Ok(json!({ "allow": false, "reason": "Leads delegate." })),
        );
        assert_eq!(
            call.await.unwrap().unwrap_err().to_string(),
            "Leads delegate."
        );

        events.lock().unwrap().clear();
        let call = tokio::spawn(gate("rlm.spawn".to_string(), json!({})));
        let (id, _) = requested_id(&events).await;
        broker.resolve(&id, Ok(json!("yes")));
        assert!(call.await.unwrap().is_err(), "an unexpected shape refuses");

        // The timeout refuses too.
        let error = gate("rlm.spawn".to_string(), json!({})).await.unwrap_err();
        assert!(error.to_string().contains("so it was refused"), "{error}");
    }
    #[tokio::test]
    async fn optional_hooks_continue_without_an_answer() {
        let (broker, events) = recording_broker();
        // Off: no event, no answer.
        assert!(broker.optional(TURN_END, json!({})).await.is_none());
        assert!(events.lock().unwrap().is_empty());
        broker
            .set_hooks(
                &[TURN_END.to_string(), BEFORE_AGENT_START.to_string()],
                None,
                Some(Duration::from_millis(20)),
            )
            .unwrap();
        // Only listed hooks ask.
        assert!(broker.optional(CONTEXT, json!({})).await.is_none());
        assert!(events.lock().unwrap().is_empty());

        let hook = broker.optional_hook(TURN_END);
        let call = tokio::spawn(hook(json!({ "toolResults": [] })));
        let (id, event) = requested_id(&events).await;
        assert_eq!(event["hook"], TURN_END);
        broker.resolve(&id, Ok(json!({ "messages": ["note"] })));
        assert_eq!(call.await.unwrap(), Some(json!({ "messages": ["note"] })));

        events.lock().unwrap().clear();
        let call = tokio::spawn(broker.optional_hook(BEFORE_AGENT_START)(json!({})));
        let (id, _) = requested_id(&events).await;
        broker.resolve(&id, Err("studio crashed".to_string()));
        assert!(call.await.unwrap().is_none(), "an error is no answer");

        // A timeout is no answer either.
        assert!(broker.optional(TURN_END, json!({})).await.is_none());
    }

    #[test]
    fn auto_refine_pause_is_shared() {
        let (broker, _) = recording_broker();
        let flag = broker.auto_refine_paused();
        assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));
        broker.set_auto_refine_paused(true);
        assert!(flag.load(std::sync::atomic::Ordering::SeqCst));
    }
}
