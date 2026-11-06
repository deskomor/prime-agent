//! The host's compaction hooks: `context` projects the summarizer input and
//! `session_before_compact` cancels the run or supplies the summary.
use super::*;
use crate::session_engine::host_hooks::{HostHookFn, SessionHostHooks};
use crate::session_engine::AgentSession;
use serde_json::{json, Value};
use std::sync::Arc;

/// A hook that records each payload and answers `answer`.
fn recording_hook(seen: &Arc<std::sync::Mutex<Vec<Value>>>, answer: Option<Value>) -> HostHookFn {
    let seen = Arc::clone(seen);
    Arc::new(move |payload| {
        let seen = Arc::clone(&seen);
        let answer = answer.clone();
        Box::pin(async move {
            seen.lock().unwrap().push(payload);
            answer
        })
    })
}

async fn hooked_session(dir: &std::path::Path, hooks: SessionHostHooks) -> AgentSession {
    let session = AgentSession::new(
        Arc::new(pa_agent::agent::Agent::new(
            pa_agent::agent::AgentOptions::default(),
        )),
        session_with_turns(dir, 3),
        vec![],
    )
    .await
    .unwrap();
    session.set_compaction_settings(super::super::compaction::CompactionSettings {
        keep_recent_tokens: 20,
        ..Default::default()
    });
    session.set_host_hooks(hooks);
    session
}

async fn compaction_summaries(session: &AgentSession) -> Vec<String> {
    session
        .shared_persistence()
        .lock()
        .await
        .get_entries()
        .into_iter()
        .filter_map(|entry| match entry {
            FileEntry::Compaction { payload, .. } => Some(payload.summary),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn the_host_cancels_a_compaction() {
    let registration = faux_registration();
    let model = registration.get_model();
    let tmp = tempfile::tempdir().unwrap();
    let seen: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
    let session = hooked_session(
        tmp.path(),
        SessionHostHooks {
            session_before_compact: Some(recording_hook(&seen, Some(json!({ "cancel": true })))),
            ..SessionHostHooks::default()
        },
    )
    .await;
    let outcome = session
        .compact(Some("focus"), &model, None, None)
        .await
        .unwrap();
    assert_eq!(
        outcome,
        CompactOutcome::Skipped(crate::session_engine::compaction_arms::HOST_CANCELLED_COMPACTION)
    );
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0]["customInstructions"], "focus");
    assert!(seen[0]["messagesToSummarize"]
        .as_array()
        .is_some_and(|messages| !messages.is_empty()));
    assert!(seen[0]["firstKeptEntryId"].is_string(), "{}", seen[0]);
    assert!(compaction_summaries(&session).await.is_empty());
    registration.unregister();
}

#[tokio::test]
async fn the_host_supplies_the_compaction_summary() {
    let registration = faux_registration();
    let model = registration.get_model();
    let tmp = tempfile::tempdir().unwrap();
    let before: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
    let projections: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
    let session = hooked_session(
        tmp.path(),
        SessionHostHooks {
            session_before_compact: Some(recording_hook(
                &before,
                Some(json!({ "summary": "HOST SUMMARY" })),
            )),
            context: Some(recording_hook(&projections, None)),
            ..SessionHostHooks::default()
        },
    )
    .await;
    let calls_before = registration.call_count();
    let outcome = session.compact(None, &model, None, None).await.unwrap();
    assert!(
        !matches!(outcome, CompactOutcome::Skipped(_)),
        "{outcome:?}"
    );
    assert_eq!(
        registration.call_count(),
        calls_before,
        "the summarizer never ran"
    );
    let summaries = compaction_summaries(&session).await;
    assert_eq!(summaries.len(), 1, "{summaries:?}");
    assert!(summaries[0].contains("HOST SUMMARY"), "{summaries:?}");
    let projections = projections.lock().unwrap().clone();
    assert!(
        projections
            .iter()
            .any(|projection| projection["purpose"] == "compaction"),
        "{projections:?}"
    );
    registration.unregister();
}
