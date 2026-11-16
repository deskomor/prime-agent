//! Host snapshots read the worker's canonical store and the live agent tool list.
use serde_json::{json, Value};

use crate::protocol::{response_failure, response_success, DaemonResponse};
use crate::worker::Worker;

// Leave room for metadata and JSON-RPC framing below a 1 MiB wire line.
const SNAPSHOT_BYTES: usize = 768 * 1024;

fn bounded_page(
    rows: &[Value],
    offset: usize,
    limit: usize,
    budget: usize,
) -> anyhow::Result<(Vec<Value>, Option<usize>)> {
    let mut page = Vec::new();
    let mut bytes = 0;
    for row in rows.iter().skip(offset).take(limit) {
        let size = serde_json::to_vec(row)?.len() + 1;
        if bytes + size > budget {
            anyhow::ensure!(
                !page.is_empty(),
                "Snapshot row exceeds the wire page budget"
            );
            break;
        }
        bytes += size;
        page.push(row.clone());
    }
    let end = offset.saturating_add(page.len());
    Ok((page, (end < rows.len()).then_some(end)))
}

fn projected_context_usage(projections: &Value, context_window: Option<u64>) -> Option<Value> {
    use pa_types::usage::{calculate_context_tokens, estimate_tokens, valid_assistant_usage};
    let (purpose, projection) = projections
        .get("provider")
        .map(|projection| ("provider", projection))
        .or_else(|| {
            projections
                .get("budget")
                .map(|projection| ("budget", projection))
        })?;
    let messages = projection.get("messages")?.as_array()?;
    let tokens = match messages
        .iter()
        .rposition(|message| valid_assistant_usage(message).is_some())
    {
        Some(index) => {
            let usage = valid_assistant_usage(&messages[index]).expect("usage anchor was checked");
            calculate_context_tokens(&usage)
                + messages[index + 1..]
                    .iter()
                    .map(estimate_tokens)
                    .sum::<u64>()
        }
        None => messages.iter().map(estimate_tokens).sum::<u64>(),
    };
    let context_window = context_window.filter(|window| *window > 0);
    Some(json!({
        "tokens": tokens,
        "contextWindow": context_window,
        "percent": context_window.map(|window| tokens as f64 / window as f64 * 100.0),
        "projectedMessageCount": messages.len(),
        "projectionIdentity": projection.get("projectionIdentity"),
        "projectionPurpose": purpose,
    }))
}

impl Worker {
    pub(crate) async fn handle_session_snapshot(&self, payload: &Value) -> DaemonResponse {
        if let Err(response) = self.require_created("session_snapshot") {
            return response;
        }
        let offset = payload.get("offset").and_then(Value::as_u64).unwrap_or(0);
        let Ok(offset) = usize::try_from(offset) else {
            return response_failure(None, "session_snapshot", "Invalid offset", None);
        };
        let limit = payload
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(100)
            .clamp(1, 1000);
        let limit = usize::try_from(limit).expect("bounded page limit");
        let before = {
            let core = self.core.lock().unwrap();
            let Some(store) = core.store.as_ref() else {
                return response_failure(
                    None,
                    "session_snapshot",
                    "Session is still initializing",
                    None,
                );
            };
            (
                core.generation.clone(),
                core.last_event_sequence,
                store.entries().len(),
                store.leaf_id().map(str::to_string),
            )
        };
        let mut projections = self.engine.host_context_projections();
        let applied_projections = projections.clone();
        // Estimate the full applied context before slicing its wire pages.
        let projected_usage =
            projected_context_usage(&projections, self.engine.model_context_window());
        let projection_count = projections.as_object().map_or(0, serde_json::Map::len);
        let page_budget = (SNAPSHOT_BYTES - 128 * 1024) / (2 + projection_count);
        if let Some(projections) = projections.as_object_mut() {
            for projection in projections.values_mut() {
                if let Some(messages) = projection.get("messages").and_then(Value::as_array) {
                    let total = messages.len();
                    let (page, next) = match bounded_page(messages, offset, limit, page_budget) {
                        Ok(page) => page,
                        Err(error) => {
                            return response_failure(
                                None,
                                "session_snapshot",
                                &error.to_string(),
                                None,
                            )
                        }
                    };
                    let end = offset.saturating_add(page.len());
                    if let Some(refs) = projection
                        .get_mut("entryRefs")
                        .and_then(Value::as_array_mut)
                    {
                        refs.retain(|reference| {
                            reference
                                .get("messageIndex")
                                .and_then(Value::as_u64)
                                .and_then(|index| usize::try_from(index).ok())
                                .is_some_and(|index| index >= offset && index < end)
                        });
                        for reference in refs {
                            if let Some(index) =
                                reference.get("messageIndex").and_then(Value::as_u64)
                            {
                                reference["messageIndex"] =
                                    json!(index
                                        .saturating_sub(u64::try_from(offset).unwrap_or(u64::MAX)));
                            }
                        }
                    }
                    projection["messages"] = json!(page);
                    projection["offset"] = json!(offset);
                    projection["total"] = json!(total);
                    projection["nextOffset"] = json!(next);
                }
            }
        }
        let definitions = self.engine.host_tool_definitions().await;
        let tools_stable = self.engine.host_tool_definitions().await == definitions;
        let tools: Vec<Value> = definitions
            .iter()
            .filter_map(|tool| tool.get("name").cloned())
            .collect();
        let core = self.core.lock().unwrap();
        let Some(store) = core.store.as_ref() else {
            return response_failure(
                None,
                "session_snapshot",
                "Session is still initializing",
                None,
            );
        };
        let after = (
            core.generation.clone(),
            core.last_event_sequence,
            store.entries().len(),
            store.leaf_id().map(str::to_string),
        );
        if before != after
            || !tools_stable
            || self.engine.host_context_projections() != applied_projections
        {
            return response_failure(
                None,
                "session_snapshot",
                "Session changed while reading snapshot; retry",
                None,
            );
        }
        let branch = store.branch();
        let entries: Vec<Value> = branch
            .iter()
            .map(|entry| serde_json::to_value(entry).expect("session entry serialization"))
            .collect();
        let context =
            pa_core::session::build_session_context(&store.branch_file_entries(), store.leaf_id());
        let messages: Vec<Value> = context
            .messages
            .iter()
            .map(|message| serde_json::to_value(message).expect("session message serialization"))
            .collect();
        let full_refs: Vec<_> = context
            .entry_ids
            .iter()
            .enumerate()
            .filter_map(|(index, entry_id)| {
                entry_id
                    .as_ref()
                    .map(|entry_id| json!({"messageIndex":index, "entryId":entry_id}))
            })
            .collect();
        let source = json!({"messages":messages, "entryRefs":full_refs});
        let projection_current = applied_projections.get("provider").is_some()
            && self.engine.host_context_source_matches(&source);
        let canonical_usage =
            crate::session_stats::store_context_usage(store, self.engine.model_context_window());
        let current_usage = if projection_current {
            projected_usage.clone()
        } else {
            canonical_usage.clone()
        };
        let (entry_page, next_entry) = match bounded_page(&entries, offset, limit, page_budget) {
            Ok(page) => page,
            Err(error) => {
                return response_failure(None, "session_snapshot", &error.to_string(), None)
            }
        };
        let (message_page, next_message) = match bounded_page(&messages, offset, limit, page_budget)
        {
            Ok(page) => page,
            Err(error) => {
                return response_failure(None, "session_snapshot", &error.to_string(), None)
            }
        };
        let refs: Vec<_> = context
            .entry_ids
            .iter()
            .enumerate()
            .skip(offset)
            .take(message_page.len())
            .filter_map(|(index, entry_id)| {
                entry_id
                    .as_ref()
                    .map(|entry_id| json!({"messageIndex":index - offset, "entryId":entry_id}))
            })
            .collect();
        let auto_refine_enabled = !self
            .host_hooks
            .auto_refine_paused()
            .load(std::sync::atomic::Ordering::Acquire);
        let mut admission_ids: Vec<&String> = core.running_admission_ids.iter().collect();
        admission_ids.sort();
        let snapshot = json!({
            "activeSessionId": core.active_session_id,
            "generation": core.generation,
            "eventSequence": core.last_event_sequence,
            "admissionIds": admission_ids,
            "sessionId": store.session_id(),
            "sessionFile": store.path,
            "cwd": core.cwd,
            "role": core.runtime_kind,
            "runtimeKind": core.runtime_kind,
            "parentActiveSessionId": core.parent_active_session_id,
            "parentSessionId": core.parent_session_id,
            "header": store.header,
            "leafId": store.leaf_id(),
            "revision": {
                "generation": core.generation,
                "eventSequence": core.last_event_sequence,
                "entryCount": store.entries().len(),
                "leafId": store.leaf_id(),
                "tools": tools,
                "autoRefineEnabled": auto_refine_enabled,
            },
            "model": self.engine.model_metadata(),
            "tools": tools,
            "toolDefinitions": definitions,
            "contextUsage": current_usage,
            "contextUsageSource": if projection_current { "appliedProjection" } else { "canonical" },
            "canonicalContextUsage": canonical_usage,
            "lastAppliedProjectionUsage": projected_usage,
            "autoRefineEnabled": auto_refine_enabled,
            "projections": projections,
            "entries": entry_page,
            "offset": offset,
            "total": entries.len(),
            "nextOffset": next_entry,
            "context": {
                "messages": message_page,
                "entryRefs": refs,
                "offset": offset,
                "total": messages.len(),
                "nextOffset": next_message,
                "thinkingLevel": context.thinking_level,
                "serviceTier": context.service_tier,
                "model": context.model.map(|(provider, model_id)| json!({"provider":provider, "modelId":model_id})),
            },
        });
        if serde_json::to_vec(&snapshot).map_or(true, |bytes| bytes.len() > SNAPSHOT_BYTES) {
            return response_failure(
                None,
                "session_snapshot",
                "Snapshot metadata exceeds the wire budget",
                None,
            );
        }
        response_success(None, "session_snapshot", Some(snapshot))
    }

    pub(crate) async fn handle_set_active_tools(&self, payload: &Value) -> DaemonResponse {
        if let Err(response) = self.require_created("set_active_tools") {
            return response;
        }
        let Some(tools) = payload.get("tools").and_then(Value::as_array) else {
            return response_failure(
                None,
                "set_active_tools",
                "tools must be an array of names",
                None,
            );
        };
        let Some(tools) = tools
            .iter()
            .map(|tool| tool.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()
        else {
            return response_failure(
                None,
                "set_active_tools",
                "tools must be an array of names",
                None,
            );
        };
        let owner = self.host_hook_owner.lock().await;
        if let Some(token) = payload.get("hostOwnerToken") {
            if token.as_str().is_none() || token.as_str() != owner.as_deref() {
                return response_failure(
                    None,
                    "set_active_tools",
                    "Host adapter lease has expired",
                    None,
                );
            }
        }
        let result = self.engine.set_host_active_tools(tools).await;
        drop(owner);
        match result {
            Ok(()) => response_success(None, "set_active_tools", None),
            Err(error) => response_failure(None, "set_active_tools", &error.to_string(), None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_preserves_bookkeeping_and_advances_without_truncating_rows() {
        let rows = vec![
            json!({"type":"model_change","id":"m"}),
            json!({"type":"custom","id":"c","data":"kept"}),
        ];
        let budget = serde_json::to_vec(&rows[0]).unwrap().len() + 1;
        assert_eq!(
            bounded_page(&rows, 0, 100, budget).unwrap(),
            (vec![rows[0].clone()], Some(1))
        );
        assert_eq!(
            bounded_page(&rows, 1, 1, 1024).unwrap(),
            (vec![rows[1].clone()], None)
        );
        assert!(bounded_page(&rows, 0, 1, 1).is_err());
    }
    #[test]
    fn usage_reads_full_applied_budget_projection_before_paging() {
        let projection = json!({"budget": {
            "messages": [
                {"role":"assistant", "usage":{"totalTokens":40}},
                {"role":"user", "content":"abcdefgh"}
            ],
            "projectionIdentity":"budget-1"
        }});
        let usage = projected_context_usage(&projection, Some(100)).unwrap();
        assert_eq!(usage["tokens"], json!(42));
        assert_eq!(usage["projectedMessageCount"], json!(2));
        assert_eq!(usage["projectionIdentity"], json!("budget-1"));
    }
}
