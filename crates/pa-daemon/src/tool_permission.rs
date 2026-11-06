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
//!
//! An RLM child spawned while its parent asks inherits the gate with the
//! parent as its upstream: each request goes to the parent worker as
//! `request_tool_permission`, which decides it like its own call (its rules,
//! its clients, or its own upstream) and answers with the outcome. The
//! request carries the child's identity as `origin`.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pa_agent::abort::AbortSignal;
use pa_agent::agent_loop::BeforeToolCallFn;
use pa_agent::types::BeforeToolCallResult;
use serde_json::{json, Map, Value};
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

    pub(crate) fn as_str(self) -> &'static str {
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

/// Forwards one request (its wire shape) to the session that decides for
/// this one: an RLM child asks its parent. Resolves to the outcome, `None`
/// when no answer arrived.
pub(crate) type ToolPermissionUpstream = Arc<
    dyn Fn(Value) -> Pin<Box<dyn Future<Output = Option<ToolPermissionOutcome>> + Send>>
        + Send
        + Sync,
>;

/// One tool call waiting for a decision.
pub(crate) struct ToolPermissionRequest {
    pub(crate) tool_call_id: String,
    pub(crate) tool_name: String,
    pub(crate) args: Value,
    /// The RLM child session the call runs in; `None` for this session's own calls.
    pub(crate) origin: Option<Value>,
}

impl ToolPermissionRequest {
    /// The request fields as events and the delegation command carry them.
    fn wire(&self) -> Map<String, Value> {
        let mut wire = Map::new();
        wire.insert("toolCallId".to_string(), json!(self.tool_call_id));
        wire.insert("toolName".to_string(), json!(self.tool_name));
        wire.insert("args".to_string(), self.args.clone());
        if let Some(origin) = &self.origin {
            wire.insert("origin".to_string(), origin.clone());
        }
        wire
    }
}

struct PendingRequest {
    tool_name: String,
    answer: oneshot::Sender<ToolPermissionOutcome>,
}

#[derive(Default)]
struct BrokerState {
    mode: Option<ToolPermissionMode>,
    upstream: Option<ToolPermissionUpstream>,
    pending: HashMap<String, PendingRequest>,
    always_allow: HashSet<String>,
    always_reject: HashSet<String>,
}

/// Where one request goes, decided under the lock.
enum Route {
    Decided(ToolPermissionOutcome),
    Local(OpenRequest),
    Upstream(ToolPermissionUpstream),
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
            state.upstream = None;
            for (_, request) in state.pending.drain() {
                let _ = request.answer.send(ToolPermissionOutcome::Cancelled);
            }
        }
    }

    /// Ask, and send every request to `upstream` instead of this session's
    /// clients (an RLM child whose parent's host approves its calls).
    ///
    /// # Panics
    ///
    /// Panics when the broker lock is poisoned.
    pub(crate) fn set_upstream(&self, upstream: ToolPermissionUpstream) {
        let mut state = self.state.lock().expect("tool permission lock");
        state.mode = Some(ToolPermissionMode::Ask);
        state.upstream = Some(upstream);
    }

    /// Whether calls wait for a decision (children spawned now inherit it).
    ///
    /// # Panics
    ///
    /// Panics when the broker lock is poisoned.
    pub(crate) fn asks(&self) -> bool {
        self.state.lock().expect("tool permission lock").mode == Some(ToolPermissionMode::Ask)
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
        remember(&mut state, &request.tool_name, outcome);
        let _ = request.answer.send(outcome);
        true
    }

    /// The engine's `before_tool_call` seam over this broker.
    pub(crate) fn gate(self: &Arc<Self>) -> BeforeToolCallFn {
        let broker = Arc::clone(self);
        Arc::new(move |context, signal| {
            let broker = Arc::clone(&broker);
            Box::pin(async move {
                let request = ToolPermissionRequest {
                    tool_call_id: context.tool_call.id.clone(),
                    tool_name: context.tool_call.name.clone(),
                    args: context.args.clone(),
                    origin: None,
                };
                let outcome = broker.decide(request, Some(signal)).await;
                Ok(gate_answer(outcome))
            })
        })
    }

    /// Decide one call: at once when the gate is off or a rule covers the
    /// tool, else by this session's clients or the upstream session. `None`
    /// is the timeout (or an upstream that never answered); an abort is
    /// `Cancelled`.
    pub(crate) async fn decide(
        &self,
        request: ToolPermissionRequest,
        signal: Option<AbortSignal>,
    ) -> Option<ToolPermissionOutcome> {
        match self.route(&request) {
            Route::Decided(outcome) => Some(outcome),
            Route::Local(open) => {
                let outcome = tokio::select! {
                    outcome = open.receiver => outcome.ok(),
                    () = tokio::time::sleep(self.timeout) => None,
                    () = aborted(signal) => Some(ToolPermissionOutcome::Cancelled),
                };
                self.settle(&open.request_id, outcome);
                outcome
            }
            Route::Upstream(upstream) => {
                let outcome = tokio::select! {
                    outcome = upstream(Value::Object(request.wire())) => outcome,
                    () = tokio::time::sleep(self.timeout) => None,
                    () = aborted(signal) => Some(ToolPermissionOutcome::Cancelled),
                };
                if let Some(outcome) = outcome {
                    let mut state = self.state.lock().expect("tool permission lock");
                    remember(&mut state, &request.tool_name, outcome);
                }
                outcome
            }
        }
    }

    fn route(&self, request: &ToolPermissionRequest) -> Route {
        let mut state = self.state.lock().expect("tool permission lock");
        if state.mode != Some(ToolPermissionMode::Ask) {
            return Route::Decided(ToolPermissionOutcome::AllowOnce);
        }
        if state.always_allow.contains(&request.tool_name) {
            return Route::Decided(ToolPermissionOutcome::AllowAlways);
        }
        if state.always_reject.contains(&request.tool_name) {
            return Route::Decided(ToolPermissionOutcome::RejectAlways);
        }
        if let Some(upstream) = &state.upstream {
            return Route::Upstream(Arc::clone(upstream));
        }
        let request_id = uuid::Uuid::new_v4().to_string();
        let (answer, receiver) = oneshot::channel();
        state.pending.insert(
            request_id.clone(),
            PendingRequest {
                tool_name: request.tool_name.clone(),
                answer,
            },
        );
        drop(state);
        let mut event = request.wire();
        event.insert("type".to_string(), json!("tool_permission_request"));
        event.insert("requestId".to_string(), json!(request_id));
        (self.emit)(Value::Object(event));
        Route::Local(OpenRequest {
            request_id,
            receiver,
        })
    }

    /// Close a local request and tell the clients how it ended. `None` is
    /// the timeout.
    fn settle(&self, request_id: &str, outcome: Option<ToolPermissionOutcome>) {
        self.state
            .lock()
            .expect("tool permission lock")
            .pending
            .remove(request_id);
        (self.emit)(json!({
            "type": "tool_permission_resolved",
            "requestId": request_id,
            "outcome": outcome.map_or("timed_out", ToolPermissionOutcome::as_str),
        }));
    }
}

/// Record an `always` answer for the tool.
fn remember(state: &mut BrokerState, tool_name: &str, outcome: ToolPermissionOutcome) {
    match outcome {
        ToolPermissionOutcome::AllowAlways => {
            state.always_reject.remove(tool_name);
            state.always_allow.insert(tool_name.to_string());
        }
        ToolPermissionOutcome::RejectAlways => {
            state.always_allow.remove(tool_name);
            state.always_reject.insert(tool_name.to_string());
        }
        ToolPermissionOutcome::AllowOnce
        | ToolPermissionOutcome::RejectOnce
        | ToolPermissionOutcome::Cancelled => {}
    }
}

/// The gate's answer for an outcome (`None` is the timeout).
fn gate_answer(outcome: Option<ToolPermissionOutcome>) -> Option<BeforeToolCallResult> {
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

/// Resolves when the turn aborts; never without a signal.
async fn aborted(signal: Option<AbortSignal>) {
    match signal {
        Some(signal) => signal.aborted().await,
        None => std::future::pending().await,
    }
}

struct OpenRequest {
    request_id: String,
    receiver: oneshot::Receiver<ToolPermissionOutcome>,
}

#[cfg(test)]
mod tests {
    use super::*;
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
    #[tokio::test]
    async fn a_child_asks_its_upstream_and_keeps_always_answers() {
        let (broker, events) = recording_broker(TOOL_PERMISSION_TIMEOUT);
        let forwarded = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&forwarded);
        broker.set_upstream(Arc::new(move |request| {
            sink.lock().unwrap().push(request);
            Box::pin(async { Some(ToolPermissionOutcome::RejectAlways) })
        }));
        assert!(broker.asks());
        for _ in 0..2 {
            let answer = broker.gate()(context("bash"), AbortSignal::default())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                (answer.block, answer.reason.as_deref()),
                (true, Some(TOOL_PERMISSION_DENIED))
            );
        }
        let forwarded = forwarded.lock().unwrap();
        assert_eq!(
            forwarded.len(),
            1,
            "the remembered rule answers the second call"
        );
        assert_eq!(forwarded[0]["toolName"], "bash");
        assert_eq!(forwarded[0]["args"], json!({ "command": "ls" }));
        assert!(
            events.lock().unwrap().is_empty(),
            "no local client is asked"
        );
    }

    #[tokio::test]
    async fn a_delegated_request_carries_its_origin_and_off_allows_it() {
        let (broker, events) = recording_broker(TOOL_PERMISSION_TIMEOUT);
        broker.set_mode(ToolPermissionMode::Ask);
        let request = || ToolPermissionRequest {
            tool_call_id: "call-9".to_string(),
            tool_name: "python_repl".to_string(),
            args: json!({ "code": "1" }),
            origin: Some(json!({ "rlmChildId": "kid" })),
        };
        let asking = Arc::clone(&broker);
        let call = tokio::spawn(async move { asking.decide(request(), None).await });
        let id = requested_id(&events).await;
        assert_eq!(events.lock().unwrap()[0]["origin"]["rlmChildId"], "kid");
        assert!(broker.resolve(&id, ToolPermissionOutcome::AllowOnce));
        assert_eq!(call.await.unwrap(), Some(ToolPermissionOutcome::AllowOnce));

        broker.set_mode(ToolPermissionMode::Off);
        assert!(!broker.asks());
        assert_eq!(
            broker.decide(request(), None).await,
            Some(ToolPermissionOutcome::AllowOnce)
        );
    }
}
