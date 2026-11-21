//! Native admission for one ephemeral explorer per live root turn.
use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, Mutex},
};

use anyhow::{ensure, Context, Result};
use pa_agent::abort::{throw_if_aborted, AbortController};
use pa_types::daemon::DaemonCommand;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{worker_lifecycle::RosterFamilySeedOwner, RouteAdmission, Supervisor};
use crate::{
    plan_mode::{self, Scope},
    registry::ResidentWorker,
};

pub(crate) type ExploreRegistry = Arc<Mutex<HashMap<String, Arc<ExploreTurn>>>>;

pub(crate) struct ExploreTurn {
    scope: Scope,
    stop: AbortController,
    slot: tokio::sync::Mutex<()>,
    actor: Mutex<Option<String>>,
}

pub(crate) struct ExploreGuard(ExploreRegistry, String);
impl ExploreGuard {
    pub(crate) fn register(supervisor: &Supervisor, root: &ResidentWorker, scope: Scope) -> Self {
        let registry = Arc::clone(&supervisor.plan_explorers);
        let key = root.worker_id.clone();
        registry.lock().unwrap().insert(
            key.clone(),
            Arc::new(ExploreTurn {
                scope,
                stop: AbortController::new(),
                slot: tokio::sync::Mutex::new(()),
                actor: Mutex::new(None),
            }),
        );
        Self(registry, key)
    }
}
impl Drop for ExploreGuard {
    fn drop(&mut self) {
        if let Some(turn) = self.0.lock().unwrap().remove(&self.1) {
            turn.stop.abort();
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Start { admission: String, task: String },
    Stop { admission: String },
    Read { hook: String, payload: Value },
}

impl Supervisor {
    pub(super) async fn worker_plan_explore(
        self: &Arc<Self>,
        token: &str,
        value: Value,
    ) -> Result<Value> {
        let worker = self
            .registry
            .find_by_token(token)
            .await
            .context("Unknown native worker")?;
        let request = serde_json::from_value::<Request>(value)?;
        match request {
            Request::Read { hook, payload } => {
                ensure!(
                    matches!(hook.as_str(), plan_mode::VETO | plan_mode::TOOL)
                        && payload["toolName"] == "studio_plan_read",
                    "Invalid Plan explorer hook"
                );
                let bound = self
                    .plan_explorers
                    .lock()
                    .unwrap()
                    .iter()
                    .find_map(|(root, turn)| {
                        (turn.actor.lock().unwrap().as_deref() == Some(worker.worker_id.as_str()))
                            .then(|| (root.clone(), Arc::clone(turn)))
                    });
                let (root, turn) = bound.context("Worker is not the bound Plan explorer")?;
                throw_if_aborted(&turn.stop.signal())?;
                let root = self.registry.get(&root).await.context("Plan root ended")?;
                self.explorer_hook(&root, &turn, &worker.worker_id, &hook, payload)
                    .await
            }
            Request::Start { ref admission, .. } | Request::Stop { ref admission } => {
                let turn = self
                    .plan_explorers
                    .lock()
                    .unwrap()
                    .get(&worker.worker_id)
                    .cloned()
                    .filter(|turn| &turn.scope.admission == admission)
                    .context("Plan admission ended")?;
                match request {
                    Request::Start { task, .. } => {
                        ensure!(plan_mode::valid_task(&task), "Invalid Plan explorer task");
                        // Own admission independently of the worker link's lifetime.
                        let supervisor = Arc::clone(self);
                        tokio::spawn(async move {
                            supervisor.run_plan_explorer(&worker, &turn, &task).await
                        })
                        .await?
                    }
                    Request::Stop { .. } => {
                        turn.stop.abort();
                        Ok(Value::Null)
                    }
                    Request::Read { .. } => unreachable!(),
                }
            }
        }
    }

    async fn explorer_hook(
        &self,
        root: &Arc<ResidentWorker>,
        turn: &ExploreTurn,
        actor: &str,
        hook: &str,
        payload: Value,
    ) -> Result<Value> {
        let result = self
            .route_command_typed(
                root,
                "plan_explorer_hook",
                json!({"scope":turn.scope,"actor":actor,"hook":hook,"payload":payload}),
                20_000,
                RouteAdmission::SupervisorInternal,
            )
            .await?;
        ensure!(
            result.success,
            "{}",
            result
                .error
                .unwrap_or_else(|| "Plan explorer hook failed".into())
        );
        result.data.context("Missing Plan explorer hook response")
    }

    async fn run_plan_explorer(
        self: &Arc<Self>,
        root: &Arc<ResidentWorker>,
        turn: &ExploreTurn,
        task: &str,
    ) -> Result<Value> {
        let _slot = turn
            .slot
            .try_lock()
            .context("A Plan explorer is already running")?;
        let stop = turn.stop.signal();
        throw_if_aborted(&stop)?;
        let mut config = {
            let descriptor = root.descriptor.lock().await;
            let source = &descriptor.create_command.rest;
            let mut config = json!({"cwd":source.get("cwd").and_then(Value::as_str).context("Plan root has no cwd")?,
                "rlmDepth":1,"rlmMaxDepth":1});
            for key in ["provider", "model", "thinking", "models"] {
                if let Some(value) = source.get(key) {
                    config[key] = value.clone();
                }
            }
            if let Some(script) = source.get("childScript").and_then(Value::as_str) {
                config["script"] = json!(script);
            } else if let Some(path) = source.get("script").and_then(Value::as_str) {
                let script: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
                if let Some(child_path) = script["childScript"].as_str() {
                    config["script"] = json!(child_path);
                }
            }
            config
        };
        let live = self
            .route_command_typed(
                root,
                "get_state",
                json!({}),
                5_000,
                RouteAdmission::SupervisorInternal,
            )
            .await?;
        let state = live
            .data
            .filter(|_| live.success)
            .context("Plan root state unavailable")?;
        if let Some(provider) = state.pointer("/model/provider").and_then(Value::as_str) {
            config["provider"] = json!(provider);
        }
        if let Some(model) = state.pointer("/model/id").and_then(Value::as_str) {
            config["model"] = json!(model);
        }
        if let Some(thinking) = state["thinkingLevel"].as_str() {
            config["thinking"] = json!(thinking);
        }
        let child_id = uuid::Uuid::new_v4();
        let create = DaemonCommand::Create {
            id: None,
            session_path: None,
            continue_recent: None,
            no_session: Some(true),
            name: Some(format!("plan-explorer-{child_id}")),
            config: Some(config),
            telemetry_disabled: Some(true),
            runtime_metadata: Some(
                json!({"kind":"subagent","rlmChildId":child_id.to_string(), "rlmDepth":1,
                "parentActiveSessionId":turn.scope.actor}),
            ),
            lifecycle: None,
            env: None,
            launch_env: None,
            rest: serde_json::Map::new(),
        };
        let start = self.launch_worker(&create, None, RosterFamilySeedOwner::PlanExplorer);
        with_explorer(
            &stop,
            start,
            |(child, _)| async move {
                let ack = self
                    .explorer_hook(
                        root,
                        turn,
                        &child.worker_id,
                        plan_mode::EXPLORER,
                        json!({"action":"admitted"}),
                    )
                    .await?;
                ensure!(
                    ack["ok"] == true,
                    "Host did not acknowledge the Plan explorer"
                );
                Ok(())
            },
            |(child, _)| async move {
                *turn.actor.lock().unwrap() = Some(child.worker_id.clone());
                let reply = self
                    .route_command_typed(
                        &child,
                        "plan_explorer_prompt",
                        json!({"task":task}),
                        90_000,
                        RouteAdmission::SupervisorInternal,
                    )
                    .await?;
                ensure!(reply.success, "Plan explorer failed");
                reply.data.context("Plan explorer returned no report")
            },
            |(child, _), announced| async move {
                turn.actor.lock().unwrap().take();
                let (stopped, revoked) = tokio::join!(self.stop_worker(&child), async {
                    if announced {
                        let ack = self
                            .explorer_hook(
                                root,
                                turn,
                                &child.worker_id,
                                plan_mode::EXPLORER,
                                json!({"action":"finished"}),
                            )
                            .await?;
                        ensure!(ack["ok"] == true, "Host did not revoke the Plan explorer");
                    }
                    Ok::<_, anyhow::Error>(())
                });
                stopped?;
                revoked
            },
        )
        .await
    }
}

async fn with_explorer<C: Clone, R, A, E, F>(
    stop: &pa_agent::abort::AbortSignal,
    start: impl Future<Output = Result<C>>,
    announce: impl FnOnce(C) -> A,
    run: impl FnOnce(C) -> E,
    finish: impl FnOnce(C, bool) -> F,
) -> Result<R>
where
    A: Future<Output = Result<()>>,
    E: Future<Output = Result<R>>,
    F: Future<Output = Result<()>>,
{
    use pa_agent::abort::race_with_abort;
    throw_if_aborted(stop)?;
    // Keep ownership through startup: a process handle can arrive after Stop.
    let child = start.await?;
    let announced = !stop.is_aborted();
    let result = async {
        race_with_abort(announce(child.clone()), stop).await??;
        race_with_abort(run(child.clone()), stop).await?
    }
    .await;
    let finished = finish(child, announced).await;
    let report = result?;
    finished?;
    throw_if_aborted(stop)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    #[tokio::test(start_paused = true)]
    async fn plan_explorer_stop_during_startup_or_pending_ack_closes_the_handle() {
        for stop_during_ack in [false, true] {
            let stop = AbortController::new();
            let signal = stop.signal();
            let (starting, start_seen) = oneshot::channel();
            let (release, released) = oneshot::channel();
            let (announcing, announce_seen) = oneshot::channel();
            let (finished, finish_seen) = oneshot::channel();
            let request = tokio::spawn(async move {
                with_explorer(
                    &signal,
                    async {
                        starting.send(()).unwrap();
                        released.await.unwrap();
                        Ok("native handle")
                    },
                    async move |_| {
                        announcing.send(()).unwrap();
                        std::future::pending().await
                    },
                    async |_| -> Result<()> {
                        panic!("a cancelled or unacknowledged explorer ran")
                    },
                    async move |handle, announced| {
                        finished.send((handle, announced)).unwrap();
                        Ok(())
                    },
                )
                .await
            });
            start_seen.await.unwrap();
            if stop_during_ack {
                release.send(()).unwrap();
                announce_seen.await.unwrap();
                stop.abort();
            } else {
                stop.abort();
                release.send(()).unwrap();
                assert!(announce_seen.await.is_err());
            }
            assert_eq!(
                tokio::time::timeout(std::time::Duration::from_secs(1), finish_seen)
                    .await
                    .unwrap()
                    .unwrap(),
                ("native handle", stop_during_ack)
            );
            assert!(pa_agent::abort::is_abort_error(
                &request.await.unwrap().unwrap_err()
            ));
        }
    }
}
