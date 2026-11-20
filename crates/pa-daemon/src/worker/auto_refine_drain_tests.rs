//! The background auto-refinement drain (`wait_for_auto_refine_idle`).
use super::*;

async fn created_worker() -> Arc<Worker> {
    let dir = std::env::temp_dir().join(format!("pa-worker-drain-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = WorkerConfig {
        socket_path: dir.join("worker.sock"),
        supervisor_socket_path: PathBuf::new(),
        token: "token".to_string(),
        worker_instance_id: String::new(),
        active_session_id: "drain-session".to_string(),
        agent_dir: dir.join("agent"),
        recovery_journal_path: dir.join("recovery.jsonl"),
        telemetry_disabled: None,
        script: Some(json!({ "responses": ["ack"] })),
    };
    let worker = Arc::new(Worker::new(config, None));
    let created = worker
        .dispatch(
            "create",
            &json!({ "noSession": true, "cwd": "/tmp", "name": "drain" }),
        )
        .await;
    assert!(created.success, "create failed: {created:?}");
    worker
}

fn settled(response: &DaemonResponse) -> Option<bool> {
    assert!(response.success, "{response:?}");
    response.data.as_ref()?.get("settled")?.as_bool()
}

#[tokio::test]
async fn the_drain_waits_for_a_reserved_round_and_answers_false_at_its_deadline() {
    let worker = created_worker().await;
    let drain = |timeout_ms: u64| {
        let worker = Arc::clone(&worker);
        async move {
            worker
                .dispatch(
                    "wait_for_auto_refine_idle",
                    &json!({ "timeoutMs": timeout_ms }),
                )
                .await
        }
    };
    assert_eq!(settled(&drain(1000).await), Some(true));

    worker.core.lock().unwrap().auto_refine_rounds += 1;
    assert_eq!(settled(&drain(50).await), Some(false));

    let waiting = tokio::spawn(drain(5000));
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(!waiting.is_finished(), "a reserved round holds the drain");
    worker.core.lock().unwrap().auto_refine_rounds -= 1;
    worker.idle_notify.notify_waiters();
    assert_eq!(settled(&waiting.await.unwrap()), Some(true));

    let missing = worker
        .dispatch("wait_for_auto_refine_idle", &json!({}))
        .await;
    assert!(!missing.success);
}
