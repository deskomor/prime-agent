//! ACP host access to the root session and its native descendants.
use std::collections::HashSet;
use std::sync::Arc;

use pa_types::daemon::DaemonCommand;
use serde_json::{json, Map, Value};
use tokio::sync::Mutex;

use super::daemon::{DaemonAcpState, DaemonLink};
use super::{jsonrpc, producer};

pub(crate) const APPEND_SESSION_ENTRY_METHOD: &str =
    "_ai.primeintellect.prime-agent/append_session_entry";
pub(crate) const SET_AUTO_REFINE_METHOD: &str = "_ai.primeintellect.prime-agent/set_auto_refine";
pub(crate) const SESSION_SNAPSHOT_METHOD: &str = "_ai.primeintellect.prime-agent/session_snapshot";
pub(crate) const APPEND_CUSTOM_MESSAGE_METHOD: &str =
    "_ai.primeintellect.prime-agent/append_custom_message";
pub(crate) const SET_ACTIVE_TOOLS_METHOD: &str = "_ai.primeintellect.prime-agent/set_active_tools";

// Only authoritative native parent links establish membership. An ACP selector
// cannot confer access to another root in the shared daemon.
fn is_descendant(root: &str, target: &str, sessions: &[Value]) -> bool {
    let mut current = target;
    let mut seen = HashSet::new();
    while seen.insert(current) {
        if current == root {
            return true;
        }
        let Some(parent) = sessions
            .iter()
            .find(|session| session.get("activeSessionId").and_then(Value::as_str) == Some(current))
            .and_then(|session| session.get("parentActiveSessionId"))
            .and_then(Value::as_str)
        else {
            return false;
        };
        current = parent;
    }
    false
}

pub(super) async fn handle_session_ext(
    id: Value,
    method: &str,
    params: Value,
    link: &Arc<DaemonLink>,
    state: &Arc<Mutex<DaemonAcpState>>,
    tx: producer::FrameSink,
) {
    let result: anyhow::Result<Value> = async {
        let session_id = params
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let root = {
            let guard = state.lock().await;
            if params.get("targetActiveSessionId").is_some()
                || matches!(
                    method,
                    SESSION_SNAPSHOT_METHOD
                        | APPEND_CUSTOM_MESSAGE_METHOD
                        | SET_ACTIVE_TOOLS_METHOD
                )
            {
                anyhow::ensure!(
                    guard.permissions.host_contract(),
                    "Studio host contract v1 was not negotiated"
                );
            }
            guard
                .session
                .as_ref()
                .filter(|hosted| hosted.acp_session_id == session_id)
                .map(|hosted| hosted.daemon_active_session_id.clone())
        }
        .ok_or_else(|| anyhow::anyhow!("Unknown ACP session: {session_id}"))?;
        let active_session_id = match params.get("targetActiveSessionId") {
            None => root.clone(),
            Some(target) => {
                let target = target
                    .as_str()
                    .filter(|target| !target.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!("targetActiveSessionId must be a non-empty string")
                    })?;
                if target != root {
                    let roster = link
                        .request(DaemonCommand::List {
                            id: None,
                            all: Some(false),
                            cwd: None,
                            session_dir: None,
                            include_client_owned: Some(true),
                            rest: Map::default(),
                        })
                        .await?;
                    anyhow::ensure!(roster.success, "{}", roster.error.unwrap_or_default());
                    let data = roster.data.unwrap_or(Value::Null);
                    let sessions = data
                        .get("sessions")
                        .and_then(Value::as_array)
                        .ok_or_else(|| anyhow::anyhow!("Native session roster missing"))?;
                    anyhow::ensure!(
                        is_descendant(&root, target, sessions),
                        "Target is not a descendant of the ACP session"
                    );
                }
                target.to_string()
            }
        };
        let command = match method {
            APPEND_SESSION_ENTRY_METHOD => {
                let custom_type = params
                    .get("customType")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("customType must be a non-empty string"))?;
                DaemonCommand::AppendSessionEntry {
                    id: None,
                    active_session_id,
                    custom_type: custom_type.to_string(),
                    data: params.get("data").filter(|data| !data.is_null()).cloned(),
                    rest: Map::default(),
                }
            }
            SET_AUTO_REFINE_METHOD => DaemonCommand::SetAutoRefine {
                id: None,
                active_session_id,
                enabled: params
                    .get("enabled")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| anyhow::anyhow!("enabled must be a boolean"))?,
                rest: Map::default(),
            },
            SESSION_SNAPSHOT_METHOD => {
                let mut bounds = [None, None];
                for (index, key) in ["offset", "limit"].iter().enumerate() {
                    if let Some(value) = params.get(*key) {
                        let value = value
                            .as_u64()
                            .and_then(|value| usize::try_from(value).ok())
                            .ok_or_else(|| {
                                anyhow::anyhow!("{key} must be a non-negative integer")
                            })?;
                        anyhow::ensure!(*key != "limit" || value > 0, "limit must be positive");
                        bounds[index] = Some(value);
                    }
                }
                DaemonCommand::SessionSnapshot {
                    id: None,
                    active_session_id,
                    offset: bounds[0],
                    limit: bounds[1],
                    rest: Map::default(),
                }
            }
            APPEND_CUSTOM_MESSAGE_METHOD => DaemonCommand::AppendCustomMessage {
                id: None,
                active_session_id,
                message: params
                    .get("message")
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("message is required"))?,
                rest: Map::default(),
            },
            SET_ACTIVE_TOOLS_METHOD => {
                let tools = params
                    .get("tools")
                    .and_then(Value::as_array)
                    .and_then(|tools| {
                        tools
                            .iter()
                            .map(|tool| tool.as_str().map(str::to_string))
                            .collect::<Option<Vec<_>>>()
                    })
                    .ok_or_else(|| anyhow::anyhow!("tools must be an array of names"))?;
                DaemonCommand::SetActiveTools {
                    id: None,
                    active_session_id,
                    tools,
                    rest: Map::default(),
                }
            }
            _ => anyhow::bail!("Unknown session extension: {method}"),
        };
        let response = link.request(command).await?;
        anyhow::ensure!(response.success, "{}", response.error.unwrap_or_default());
        let mut data = response.data.unwrap_or_else(|| json!({}));
        if method == SESSION_SNAPSHOT_METHOD {
            let producer = state
                .lock()
                .await
                .session
                .as_ref()
                .map(|hosted| Arc::clone(&hosted.producer));
            if let Some(producer) = producer {
                data["promptTurnId"] = json!(producer.active_prompt_turn().await);
            }
        }
        Ok(data)
    }
    .await;
    let frame = match result {
        Ok(data) => jsonrpc::response(&id, &data),
        Err(error) => jsonrpc::error_response(
            &id,
            jsonrpc::INVALID_PARAMS,
            "Session extension failed",
            Some(&json!({"reason":error.to_string()})),
        ),
    };
    let _ = tx.send(frame);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descendant_targets_do_not_admit_foreign_sessions() {
        let sessions = json!([
            {"activeSessionId":"root"},
            {"activeSessionId":"child","parentActiveSessionId":"root"},
            {"activeSessionId":"grandchild","parentActiveSessionId":"child"},
            {"activeSessionId":"foreign","parentActiveSessionId":"other-root"},
        ]);
        let sessions = sessions.as_array().unwrap();
        assert!(is_descendant("root", "grandchild", sessions));
        assert!(!is_descendant("root", "foreign", sessions));
        assert!(!is_descendant("root", "missing", sessions));
    }
}
