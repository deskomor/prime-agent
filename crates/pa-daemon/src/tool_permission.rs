//! Host-approved tool calls: the worker side of ACP `session/request_permission`.
//!
//! A client opts a session in with `set_tool_permission_mode` (`ask`). From
//! then on every model tool call (`bash`, `edit`, `python_repl`) waits after
//! its arguments validate: the broker emits a `tool_permission_request`
//! session event and the call runs only when a client answers with
//! `respond_tool_permission` and an allow outcome. Every other ending fails
//! closed: a reject, a cancelled or unknown outcome, the timeout, the turn's
//! abort, the mode switching back to `off`, and the worker going away (the
//! dropped answer channel) all block
//! the call with an error tool result the model reads. `allow_always` and
//! `reject_always` are remembered per tool name for the life of the worker.
//! With the mode `off` (the default) the gate answers at once and the session
//! behaves exactly as without a host.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pa_agent::agent_loop::BeforeToolCallFn;
use pa_agent::types::BeforeToolCallResult;
use serde_json::{json, Value};
use tokio::sync::oneshot;

/// How long one request waits for an answer before it is denied.
pub(crate) const TOOL_PERMISSION_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// The reason the model reads when the host denies a call.
pub(crate) const TOOL_PERMISSION_DENIED: &str = "The user denied this tool call.";
/// The reason the model reads when nobody answered in time.
pub(crate) const TOOL_PERMISSION_TIMED_OUT: &str =
    "No approval arrived for this tool call, so it was not run.";

/// Whether the session's tool calls wait for a host decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolPermissionMode {
    /// Every call runs ungated.
    Off,
    /// Every call waits for `respond_tool_permission`.
    Ask,
}

impl ToolPermissionMode {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "off" => Some(Self::Off),
            "ask" => Some(Self::Ask),
            _ => None,
        }
    }
}

/// A client's answer to one request (the ACP permission option kinds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolPermissionOutcome {
    AllowOnce,
    AllowAlways,
    RejectOnce,
    RejectAlways,
    Cancelled,
}

impl ToolPermissionOutcome {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "allow_once" => Some(Self::AllowOnce),
            "allow_always" => Some(Self::AllowAlways),
            "reject_once" => Some(Self::RejectOnce),
            "reject_always" => Some(Self::RejectAlways),
            "cancelled" => Some(Self::Cancelled),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::AllowOnce => "allow_once",
            Self::AllowAlways => "allow_always",
            Self::RejectOnce => "reject_once",
            Self::RejectAlways => "reject_always",
            Self::Cancelled => "cancelled",
        }
    }

    fn allows(self) -> bool {
        match self {
            Self::AllowOnce | Self::AllowAlways => true,
            Self::RejectOnce | Self::RejectAlways | Self::Cancelled => false,
        }
    }
}

/// Emits one session event to the worker's attached clients.
pub(crate) type ToolPermissionEmit = Arc<dyn Fn(Value) + Send + Sync>;

struct PendingRequest {
    tool_name: String,
    answer: oneshot::Sender<ToolPermissionOutcome>,
}

#[derive(Default)]
struct BrokerState {
    mode: Option<ToolPermissionMode>,
    pending: HashMap<String, PendingRequest>,
    always_allow: HashSet<String>,
    always_reject: HashSet<String>,
}

/// The session's permission broker: the engine gate awaits it, the worker
/// commands answer it.
pub(crate) struct ToolPermissionBroker {
    state: Mutex<BrokerState>,
    emit: ToolPermissionEmit,
    timeout: Duration,
}

impl ToolPermissionBroker {
    pub(crate) fn new(emit: ToolPermissionEmit, timeout: Duration) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(BrokerState::default()),
            emit,
            timeout,
        })
    }

    /// Switch the gate on or off. Switching off denies every waiting call:
    /// a host that leaves must not leave approved-by-default calls behind.
    ///
    /// # Panics
    ///
    /// Panics when the broker lock is poisoned.
    pub(crate) fn set_mode(&self, mode: ToolPermissionMode) {
        let mut state = self.state.lock().expect("tool permission lock");
        state.mode = Some(mode);
        if mode == ToolPermissionMode::Off {
            for (_, request) in state.pending.drain() {
                let _ = request.answer.send(ToolPermissionOutcome::Cancelled);
            }
        }
    }

    /// Answer one waiting request; `false` when no request has that id.
    ///
    /// # Panics
    ///
    /// Panics when the broker lock is poisoned.
    pub(crate) fn resolve(&self, request_id: &str, outcome: ToolPermissionOutcome) -> bool {
        let mut state = self.state.lock().expect("tool permission lock");
        let Some(request) = state.pending.remove(request_id) else {
            return false;
        };
        match outcome {
            ToolPermissionOutcome::AllowAlways => {
                state.always_reject.remove(&request.tool_name);
                state.always_allow.insert(request.tool_name);
            }
            ToolPermissionOutcome::RejectAlways => {
                state.always_allow.remove(&request.tool_name);
                state.always_reject.insert(request.tool_name);
            }
            ToolPermissionOutcome::AllowOnce
            | ToolPermissionOutcome::RejectOnce
            | ToolPermissionOutcome::Cancelled => {}
        }
        let _ = request.answer.send(outcome);
        true
    }

    /// The engine's `before_tool_call` seam over this broker.
    pub(crate) fn gate(self: &Arc<Self>) -> BeforeToolCallFn {
        let broker = Arc::clone(self);
        Arc::new(move |context, signal| {
            let broker = Arc::clone(&broker);
            Box::pin(async move {
                let tool_call_id = context.tool_call.id.clone();
                let tool_name = context.tool_call.name.clone();
                let Some(answer) = broker.open_request(&tool_call_id, &tool_name, &context.args)
                else {
                    return Ok(None);
                };
                let request_id = answer.request_id.clone();
                let outcome = tokio::select! {
                    outcome = answer.receiver => outcome.ok(),
                    () = tokio::time::sleep(broker.timeout) => None,
                    () = signal.aborted() => Some(ToolPermissionOutcome::Cancelled),
                };
                Ok(broker.settle(&request_id, outcome))
            })
        })
    }

    fn open_request(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        args: &Value,
    ) -> Option<OpenRequest> {
        let mut state = self.state.lock().expect("tool permission lock");
        if state.mode != Some(ToolPermissionMode::Ask) || state.always_allow.contains(tool_name) {
            return None;
        }
        let request_id = uuid::Uuid::new_v4().to_string();
        let (answer, receiver) = oneshot::channel();
        if state.always_reject.contains(tool_name) {
            let _ = answer.send(ToolPermissionOutcome::RejectAlways);
            return Some(OpenRequest {
                request_id,
                receiver,
            });
        }
        state.pending.insert(
            request_id.clone(),
            PendingRequest {
                tool_name: tool_name.to_string(),
                answer,
            },
        );
        drop(state);
        (self.emit)(json!({
            "type": "tool_permission_request",
            "requestId": request_id,
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "args": args,
        }));
        Some(OpenRequest {
            request_id,
            receiver,
        })
    }

    /// Close the request and turn its outcome into the gate's answer. `None`
    /// is the timeout.
    fn settle(
        &self,
        request_id: &str,
        outcome: Option<ToolPermissionOutcome>,
    ) -> Option<BeforeToolCallResult> {
        let waited = self
            .state
            .lock()
            .expect("tool permission lock")
            .pending
            .remove(request_id)
            .is_some();
        let resolved = outcome.map_or("timed_out", ToolPermissionOutcome::as_str);
        if waited || outcome.is_some() {
            (self.emit)(json!({
                "type": "tool_permission_resolved",
                "requestId": request_id,
                "outcome": resolved,
            }));
        }
        match outcome {
            Some(outcome) if outcome.allows() => None,
            Some(_) => Some(BeforeToolCallResult {
                block: true,
                reason: Some(TOOL_PERMISSION_DENIED.to_string()),
            }),
            None => Some(BeforeToolCallResult {
                block: true,
                reason: Some(TOOL_PERMISSION_TIMED_OUT.to_string()),
            }),
        }
    }
}

struct OpenRequest {
    request_id: String,
    receiver: oneshot::Receiver<ToolPermissionOutcome>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use pa_agent::abort::AbortSignal;
    use pa_agent::types::{AgentContext, AssistantMessage, BeforeToolCallContext, ToolCall};

    fn context(tool: &str) -> BeforeToolCallContext {
        BeforeToolCallContext {
            assistant_message: serde_json::from_value::<AssistantMessage>(json!({
                "content": [],
                "usage": {},
                "stopReason": "toolUse",
            }))
            .unwrap(),
            tool_call: ToolCall {
                id: "call-1".to_string(),
                name: tool.to_string(),
                arguments: json!({ "command": "ls" }),
                thought_signature: None,
            },
            args: json!({ "command": "ls" }),
            context: AgentContext::default(),
        }
    }

    fn recording_broker(timeout: Duration) -> (Arc<ToolPermissionBroker>, Arc<Mutex<Vec<Value>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let broker = ToolPermissionBroker::new(
            Arc::new(move |event| sink.lock().unwrap().push(event)),
            timeout,
        );
        (broker, events)
    }

    async fn requested_id(events: &Arc<Mutex<Vec<Value>>>) -> String {
        loop {
            if let Some(id) = events.lock().unwrap().iter().find_map(|event| {
                (event["type"] == "tool_permission_request")
                    .then(|| event["requestId"].as_str().unwrap().to_string())
            }) {
                return id;
            }
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test]
    async fn off_runs_every_call_without_an_event() {
        let (broker, events) = recording_broker(TOOL_PERMISSION_TIMEOUT);
        let answer = broker.gate()(context("bash"), AbortSignal::default())
            .await
            .unwrap();
        assert!(answer.is_none());
        assert!(events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ask_waits_for_the_answer_and_remembers_allow_always() {
        let (broker, events) = recording_broker(TOOL_PERMISSION_TIMEOUT);
        broker.set_mode(ToolPermissionMode::Ask);
        let gate = broker.gate();
        let call = tokio::spawn(gate(context("bash"), AbortSignal::default()));
        let id = requested_id(&events).await;
        assert!(broker.resolve(&id, ToolPermissionOutcome::AllowAlways));
        assert!(call.await.unwrap().unwrap().is_none());
        events.lock().unwrap().clear();
        let again = broker.gate()(context("bash"), AbortSignal::default())
            .await
            .unwrap();
        assert!(again.is_none());
        assert!(events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn reject_timeout_and_switching_off_all_block() {
        let (broker, events) = recording_broker(TOOL_PERMISSION_TIMEOUT);
        broker.set_mode(ToolPermissionMode::Ask);
        let call = tokio::spawn(broker.gate()(context("edit"), AbortSignal::default()));
        let id = requested_id(&events).await;
        broker.resolve(&id, ToolPermissionOutcome::RejectOnce);
        let denied = BeforeToolCallResult {
            block: true,
            reason: Some(TOOL_PERMISSION_DENIED.to_string()),
        };
        let answer = call.await.unwrap().unwrap().unwrap();
        assert_eq!(
            (answer.block, answer.reason),
            (denied.block, denied.reason.clone())
        );

        events.lock().unwrap().clear();
        let call = tokio::spawn(broker.gate()(context("edit"), AbortSignal::default()));
        requested_id(&events).await;
        broker.set_mode(ToolPermissionMode::Off);
        let answer = call.await.unwrap().unwrap().unwrap();
        assert_eq!((answer.block, answer.reason), (denied.block, denied.reason));

        let (quiet, _) = recording_broker(Duration::from_millis(1));
        quiet.set_mode(ToolPermissionMode::Ask);
        let answer = quiet.gate()(context("python_repl"), AbortSignal::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            (answer.block, answer.reason),
            (true, Some(TOOL_PERMISSION_TIMED_OUT.to_string()))
        );
    }
}
