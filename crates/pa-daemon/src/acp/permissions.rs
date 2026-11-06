//! ACP `session/request_permission` over the daemon worker's tool-permission
//! broker (`crate::tool_permission`).
//!
//! A client opts in at `initialize` with
//! `clientCapabilities._meta["ai.primeintellect.prime-agent"].toolPermissions
//! = true`; clients that do not ask keep the ungated behavior. For an
//! opted-in connection `session/new` switches the worker gate to `ask`, each
//! `tool_permission_request` session event becomes one
//! `session/request_permission` request to the client, and the client's
//! answer goes back as `respond_tool_permission`. An error answer, an
//! unknown option id, or a `cancelled` outcome deny the call. Subagents
//! spawned by the session ask through it: their requests carry an `origin`
//! (the subagent's ids and name), shown in the title and `_meta`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use serde_json::{json, Value};

use super::meta::PRIME_AGENT_META_NAMESPACE;
use super::types::AcpToolKind;

/// The ACP permission options offered for every gated call; the option id is
/// also the outcome the worker receives.
const PERMISSION_OPTIONS: [(&str, &str); 4] = [
    ("allow_once", "Allow once"),
    ("allow_always", "Always allow"),
    ("reject_once", "Reject"),
    ("reject_always", "Always reject"),
];

/// Where a client's answer goes back to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PermissionAnswer {
    pub(crate) daemon_session_id: String,
    pub(crate) request_id: String,
    pub(crate) outcome: String,
}

struct OutstandingRequest {
    daemon_session_id: String,
    request_id: String,
}

/// The connection's permission relay: opt-in flag, outgoing request ids, and
/// the requests still waiting for the client.
#[derive(Default)]
pub(crate) struct PermissionRelay {
    opted_in: AtomicBool,
    next_id: AtomicU64,
    outstanding: Mutex<HashMap<String, OutstandingRequest>>,
}

impl PermissionRelay {
    /// Record the client's opt-in from its `initialize` params.
    pub(crate) fn observe_initialize(&self, params: &Value) {
        let opted_in = params
            .pointer("/clientCapabilities/_meta")
            .and_then(|meta| meta.get(PRIME_AGENT_META_NAMESPACE))
            .and_then(|meta| meta.get("toolPermissions"))
            .and_then(Value::as_bool)
            == Some(true);
        self.opted_in.store(opted_in, Ordering::SeqCst);
    }

    pub(crate) fn opted_in(&self) -> bool {
        self.opted_in.load(Ordering::SeqCst)
    }

    /// The `session/request_permission` request for one worker
    /// `tool_permission_request` event, or `None` for any other event.
    ///
    /// # Panics
    ///
    /// Panics when the outstanding-request lock is poisoned.
    pub(crate) fn request_frame(
        &self,
        event: &Value,
        acp_session_id: &str,
        daemon_session_id: &str,
    ) -> Option<Value> {
        if !self.opted_in()
            || event.get("type").and_then(Value::as_str) != Some("tool_permission_request")
        {
            return None;
        }
        let request_id = event.get("requestId").and_then(Value::as_str)?;
        let tool_name = event
            .get("toolName")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let tool_call_id = event
            .get("toolCallId")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let id = format!(
            "prime-permission-{}",
            self.next_id.fetch_add(1, Ordering::SeqCst) + 1
        );
        self.outstanding
            .lock()
            .expect("permission relay lock")
            .insert(
                id.clone(),
                OutstandingRequest {
                    daemon_session_id: daemon_session_id.to_string(),
                    request_id: request_id.to_string(),
                },
            );
        let options: Vec<Value> = PERMISSION_OPTIONS
            .iter()
            .map(|(kind, name)| json!({ "optionId": kind, "name": name, "kind": kind }))
            .collect();
        let mut tool_call = json!({
            "toolCallId": tool_call_id,
            "title": tool_name,
            "kind": AcpToolKind::of_tool(tool_name),
            "status": "pending",
            "rawInput": event.get("args").cloned().unwrap_or(Value::Null),
        });
        // A subagent's call names the subagent in the title and carries its
        // identity under the prime-agent `_meta` namespace.
        if let Some(origin) = event.get("origin").filter(|origin| origin.is_object()) {
            let subagent = ["sessionName", "rlmChildId", "activeSessionId"]
                .iter()
                .find_map(|key| origin.get(*key).and_then(Value::as_str))
                .unwrap_or("subagent");
            tool_call["title"] = json!(format!("{tool_name} (subagent {subagent})"));
            tool_call["_meta"] = json!({ PRIME_AGENT_META_NAMESPACE: { "origin": origin } });
        }
        Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "session/request_permission",
            "params": {
                "sessionId": acp_session_id,
                "toolCall": tool_call,
                "options": options,
            },
        }))
    }

    /// Match a client response to its request; `None` when the id is not an
    /// outstanding permission request.
    ///
    /// # Panics
    ///
    /// Panics when the outstanding-request lock is poisoned.
    pub(crate) fn take_answer(
        &self,
        id: &Value,
        result: Option<&Value>,
    ) -> Option<PermissionAnswer> {
        let request = self
            .outstanding
            .lock()
            .expect("permission relay lock")
            .remove(id.as_str()?)?;
        let selected = result
            .and_then(|result| result.get("outcome"))
            .filter(|outcome| outcome.get("outcome").and_then(Value::as_str) == Some("selected"))
            .and_then(|outcome| outcome.get("optionId"))
            .and_then(Value::as_str)
            .filter(|option| PERMISSION_OPTIONS.iter().any(|(kind, _)| kind == option));
        Some(PermissionAnswer {
            daemon_session_id: request.daemon_session_id,
            request_id: request.request_id,
            outcome: selected.unwrap_or("cancelled").to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opted_in_relay() -> PermissionRelay {
        let relay = PermissionRelay::default();
        relay.observe_initialize(&json!({
            "protocolVersion": 1,
            "clientCapabilities": { "_meta": { "ai.primeintellect.prime-agent": { "toolPermissions": true } } },
        }));
        relay
    }

    fn request_event() -> Value {
        json!({
            "type": "tool_permission_request",
            "requestId": "r1",
            "toolCallId": "call-1",
            "toolName": "bash",
            "args": { "command": "rm -rf build" },
        })
    }

    #[test]
    fn a_client_without_the_opt_in_gets_no_requests() {
        let relay = PermissionRelay::default();
        relay.observe_initialize(&json!({ "protocolVersion": 1, "clientCapabilities": {} }));
        assert_eq!(
            relay.request_frame(&request_event(), "acp-1", "daemon-1"),
            None
        );
    }

    #[test]
    fn a_request_round_trips_to_the_worker_answer() {
        let relay = opted_in_relay();
        let frame = relay
            .request_frame(&request_event(), "acp-1", "daemon-1")
            .unwrap();
        assert_eq!(
            frame,
            json!({
                "jsonrpc": "2.0",
                "id": "prime-permission-1",
                "method": "session/request_permission",
                "params": {
                    "sessionId": "acp-1",
                    "toolCall": {
                        "toolCallId": "call-1",
                        "title": "bash",
                        "kind": "execute",
                        "status": "pending",
                        "rawInput": { "command": "rm -rf build" },
                    },
                    "options": [
                        { "optionId": "allow_once", "name": "Allow once", "kind": "allow_once" },
                        { "optionId": "allow_always", "name": "Always allow", "kind": "allow_always" },
                        { "optionId": "reject_once", "name": "Reject", "kind": "reject_once" },
                        { "optionId": "reject_always", "name": "Always reject", "kind": "reject_always" },
                    ],
                },
            })
        );
        let answer = relay.take_answer(
            &json!("prime-permission-1"),
            Some(&json!({ "outcome": { "outcome": "selected", "optionId": "allow_once" } })),
        );
        assert_eq!(
            answer,
            Some(PermissionAnswer {
                daemon_session_id: "daemon-1".to_string(),
                request_id: "r1".to_string(),
                outcome: "allow_once".to_string(),
            })
        );
        assert_eq!(relay.take_answer(&json!("prime-permission-1"), None), None);
    }

    #[test]
    fn errors_cancellations_and_unknown_options_deny() {
        let relay = opted_in_relay();
        for result in [
            None,
            Some(json!({ "outcome": { "outcome": "cancelled" } })),
            Some(json!({ "outcome": { "outcome": "selected", "optionId": "sure" } })),
        ] {
            let frame = relay
                .request_frame(&request_event(), "acp-1", "daemon-1")
                .unwrap();
            let answer = relay.take_answer(&frame["id"], result.as_ref()).unwrap();
            assert_eq!(answer.outcome, "cancelled");
        }
    }

    #[test]
    fn a_subagent_request_names_the_subagent() {
        let relay = opted_in_relay();
        let mut event = request_event();
        event["origin"] =
            json!({ "activeSessionId": "w-2", "rlmChildId": "c1", "sessionName": "kid" });
        let frame = relay.request_frame(&event, "acp-1", "daemon-1").unwrap();
        let tool_call = &frame["params"]["toolCall"];
        assert_eq!(tool_call["title"], "bash (subagent kid)");
        assert_eq!(
            tool_call["_meta"]["ai.primeintellect.prime-agent"]["origin"]["rlmChildId"],
            "c1"
        );
        let answer = relay
            .take_answer(
                &frame["id"],
                Some(&json!({ "outcome": { "outcome": "selected", "optionId": "reject_once" } })),
            )
            .unwrap();
        assert_eq!(
            (answer.daemon_session_id.as_str(), answer.outcome.as_str()),
            ("daemon-1", "reject_once"),
            "the parent session decides the subagent's call"
        );
    }
}
