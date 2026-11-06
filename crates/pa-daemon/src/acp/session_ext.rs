//! The client-to-agent extension requests a hook client uses to write into
//! the hosted session: `_ai.primeintellect.prime-agent/append_session_entry`
//! (`{sessionId, customType, data?}` → `{entryId}`) and
//! `_ai.primeintellect.prime-agent/set_auto_refine` (`{sessionId, enabled}`).

use std::sync::Arc;

use pa_types::daemon::DaemonCommand;
use serde_json::{json, Map, Value};
use tokio::sync::Mutex;

use super::daemon::{DaemonAcpState, DaemonLink};
use super::jsonrpc;
use super::producer;

/// Append one custom entry to the hosted session.
pub(crate) const APPEND_SESSION_ENTRY_METHOD: &str =
    "_ai.primeintellect.prime-agent/append_session_entry";
/// Turn the session's own auto-refinement on or off.
pub(crate) const SET_AUTO_REFINE_METHOD: &str = "_ai.primeintellect.prime-agent/set_auto_refine";

/// Answer one session extension request through the worker's wire command.
pub(super) async fn handle_session_ext(
    id: Value,
    method: &str,
    params: Value,
    link: &Arc<DaemonLink>,
    state: &Arc<Mutex<DaemonAcpState>>,
    tx: producer::FrameSink,
) {
    let invalid = |reason: String| {
        jsonrpc::error_response(
            &id,
            jsonrpc::INVALID_PARAMS,
            "Invalid params",
            Some(&json!({ "reason": reason })),
        )
    };
    let session_id = params
        .get("sessionId")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let daemon_session_id = {
        let guard = state.lock().await;
        guard
            .session
            .as_ref()
            .filter(|hosted| hosted.acp_session_id == session_id)
            .map(|hosted| hosted.daemon_active_session_id.clone())
    };
    let Some(active_session_id) = daemon_session_id else {
        let _ = tx.send(invalid(format!("Unknown ACP session: {session_id}")));
        return;
    };
    let command = if method == APPEND_SESSION_ENTRY_METHOD {
        let Some(custom_type) = params
            .get("customType")
            .and_then(Value::as_str)
            .filter(|custom_type| !custom_type.is_empty())
        else {
            let _ = tx.send(invalid("customType must be a non-empty string".to_string()));
            return;
        };
        DaemonCommand::AppendSessionEntry {
            id: None,
            active_session_id,
            custom_type: custom_type.to_string(),
            data: params.get("data").filter(|data| !data.is_null()).cloned(),
            rest: Map::default(),
        }
    } else {
        let Some(enabled) = params.get("enabled").and_then(Value::as_bool) else {
            let _ = tx.send(invalid("enabled must be a boolean".to_string()));
            return;
        };
        DaemonCommand::SetAutoRefine {
            id: None,
            active_session_id,
            enabled,
            rest: Map::default(),
        }
    };
    let frame = match link.request(command).await {
        Ok(response) if response.success => {
            jsonrpc::response(&id, &response.data.unwrap_or_else(|| json!({})))
        }
        Ok(response) => jsonrpc::error_response(
            &id,
            jsonrpc::INTERNAL_ERROR,
            "Internal error",
            Some(&json!({ "details": response.error.unwrap_or_default() })),
        ),
        Err(error) => jsonrpc::error_response(
            &id,
            jsonrpc::INTERNAL_ERROR,
            "Internal error",
            Some(&json!({ "details": error.to_string() })),
        ),
    };
    let _ = tx.send(frame);
}
