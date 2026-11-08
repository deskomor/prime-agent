//! `/refine --probe` (fork-only): plan a refinement, replay one task twice
//! in throwaway git worktrees (base harness and candidate harness), and keep
//! only the edits the candidate run supports.
//!
//! The two runs are RLM children of this session, spawned through the
//! daemon's child-session host, so they inherit its host hooks and tool
//! permission like any `rlm.spawn` child. Each child's local harness store
//! starts from a seed (`harnessSeed` on its create); the global store is the
//! shared one. The worktrees keep the runs out of the user's checkout; side
//! effects outside it (network calls, writes under `~`) are not contained.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use pa_types::session::{AgentMessage, FileEntry};

use super::refine::{
    apply_prepared_refinement, billed_refiner_call, context_rows_by_ids, local_harness_state_dir,
    prepare_refinement, record_refinement_verdicts, strip_display_prefixes, RefineOptions,
    RefinementSource,
};
use super::rlm_host::{RlmSpawnRequest, RlmSubagentHost};
use super::AgentSession;
use crate::refinement::executor::{conversation_text, plan_refinement, reply_text, RefinerFn};
use crate::refinement::planner::{
    AUTO_REFINE_REVIEW_MAX_OUTPUT_TOKENS, PROBE_COMPOSE_SYSTEM_PROMPT,
};
use crate::refinement::probe::{
    compose_prompt, kept_plan, parse_compose, probe_notice, probe_task, probe_verdicts,
    seed_states, ProbeRun, DEFAULT_PROBE_TIMEOUT_MS, PROBE_EVIDENCE_CHARS,
};
use crate::refinement::{load_harness_state, HarnessScope, RefinementResult};

/// One probe run to start.
#[derive(Debug, Clone)]
pub struct ProbeChildSpec {
    /// `base` or `candidate`.
    pub label: &'static str,
    /// The run's worktree (plus the session cwd's path inside the repo).
    pub cwd: PathBuf,
    /// The harness directory the run's local store starts from.
    pub harness_seed: PathBuf,
    pub task: String,
    /// `provider/model`, the session's own model.
    pub model: String,
    pub timeout: Duration,
}

/// What one finished (or timed-out) probe run left behind.
#[derive(Debug, Clone, Default)]
pub struct ProbeChildOutput {
    pub messages: Vec<AgentMessage>,
    pub answer: Option<String>,
    pub error: Option<String>,
}

/// How often a probe re-collects a child that has not settled yet.
const PROBE_COLLECT_POLL: Duration = Duration::from_millis(250);

/// Runs one probe child to its end and retires it.
pub type ProbeRunnerFn = Arc<
    dyn Fn(ProbeChildSpec) -> Pin<Box<dyn Future<Output = anyhow::Result<ProbeChildOutput>> + Send>>
        + Send
        + Sync,
>;

/// What `/refine --probe` asks for beyond a plain `/refine`.
#[derive(Debug, Clone, Default)]
pub struct ProbeRequest {
    /// Replay this text instead of the latest task prompt.
    pub task: Option<String>,
}

/// Deletes a probe child that was not retired, e.g. when the probe future
/// is dropped mid-run.
struct ChildGuard {
    host: Arc<dyn RlmSubagentHost>,
    id: Option<String>,
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            let delete = self.host.delete_subagent(id);
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                runtime.spawn(async move {
                    let _ = delete.await;
                });
            }
        }
    }
}

/// The runner over the daemon's child-session host: spawn, collect within
/// the timeout, read the child's session, then delete the child.
#[must_use]
pub fn rlm_probe_runner(host: Arc<dyn RlmSubagentHost>) -> ProbeRunnerFn {
    Arc::new(move |spec: ProbeChildSpec| {
        let host = Arc::clone(&host);
        Box::pin(async move {
            let handle = host
                .spawn(RlmSpawnRequest {
                    prompt: spec.task.clone(),
                    name: Some(format!("refine-probe-{}-{}", spec.label, now_millis())),
                    model: Some(spec.model.clone()),
                    thinking: None,
                    cwd: Some(spec.cwd.to_string_lossy().into_owned()),
                    spawned_by_request_id: None,
                    factory: None,
                    harness_seed: Some(spec.harness_seed.to_string_lossy().into_owned()),
                })
                .await?;
            let mut guard = ChildGuard {
                host: Arc::clone(&host),
                id: Some(handle.rlm_child_id.clone()),
            };
            // A collect can return before the child's first turn even starts
            // (its task prompt is admitted asynchronously), so re-collect
            // until it settles or the budget runs out.
            let deadline = std::time::Instant::now() + spec.timeout;
            let collected = loop {
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                let remaining_ms = u64::try_from(remaining.as_millis()).unwrap_or(u64::MAX);
                match host
                    .collect(vec![handle.rlm_child_id.clone()], remaining_ms)
                    .await
                {
                    Ok(results) => match results.into_iter().next() {
                        Some(result) if !result.settled && !remaining.is_zero() => {}
                        result => break Ok(result),
                    },
                    Err(error) => break Err(error),
                }
                tokio::time::sleep(PROBE_COLLECT_POLL.min(remaining)).await;
            };
            let messages = session_messages(Path::new(&handle.session_dir));
            let (answer, error) = match collected {
                Ok(Some(result)) if result.settled => (result.answer_text, result.error),
                Ok(Some(result)) => (
                    result.answer_text,
                    Some(format!("timed out after {} s", spec.timeout.as_secs())),
                ),
                Ok(None) => (None, Some("the run reported nothing".to_string())),
                Err(error) => (None, Some(format!("{error:#}"))),
            };
            if let Some(id) = guard.id.take() {
                let _ = host.delete_subagent(id).await;
            }
            Ok(ProbeChildOutput {
                messages,
                answer,
                error,
            })
        })
    })
}

/// Every message and custom message row in the session files under `dir`.
fn session_messages(dir: &Path) -> Vec<AgentMessage> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "jsonl") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
        .iter()
        .filter_map(|path| std::fs::read_to_string(path).ok())
        .flat_map(|content| crate::session::parse_session_entries(&content))
        .filter_map(|entry| match &entry {
            FileEntry::Message { message, .. } => Some(message.clone()),
            // Custom rows carry the task kickoff and the harness digest the
            // run saw, which the compose call reads too.
            FileEntry::CustomMessage { payload, .. } => Some(AgentMessage::Custom(
                crate::session::create_custom_message(payload, &entry),
            )),
            _ => None,
        })
        .collect()
}

/// The two worktrees and harness seeds of one probe. Dropping it removes the
/// worktrees and the scratch directory, on success, error or abort.
struct ProbeWorkspace {
    repo: PathBuf,
    root: PathBuf,
    worktrees: Vec<PathBuf>,
}

impl ProbeWorkspace {
    /// `relative` is the session cwd's path inside the repository.
    fn create(cwd: &Path, id: &str) -> anyhow::Result<(Self, PathBuf)> {
        let toplevel = git(cwd, &["rev-parse", "--show-toplevel"]).map_err(|_| {
            anyhow::anyhow!(
                "/refine --probe needs a git repository: {} is not inside one",
                cwd.display()
            )
        })?;
        let repo = PathBuf::from(toplevel.trim());
        let relative = cwd
            .canonicalize()
            .ok()
            .zip(repo.canonicalize().ok())
            .and_then(|(cwd, repo)| cwd.strip_prefix(repo).ok().map(Path::to_path_buf))
            .unwrap_or_default();
        let root = std::env::temp_dir().join(format!("prime-agent-probe-{id}"));
        std::fs::create_dir_all(&root)?;
        let mut workspace = Self {
            repo,
            root,
            worktrees: Vec::new(),
        };
        for label in ["base", "candidate"] {
            let path = workspace.root.join(label);
            git(
                &workspace.repo,
                &[
                    "worktree",
                    "add",
                    "--detach",
                    &path.to_string_lossy(),
                    "HEAD",
                ],
            )?;
            workspace.worktrees.push(path);
        }
        Ok((workspace, relative))
    }

    fn worktree(&self, label: &str) -> PathBuf {
        self.root.join(label)
    }
}

impl Drop for ProbeWorkspace {
    fn drop(&mut self) {
        for worktree in &self.worktrees {
            let _ = git(
                &self.repo,
                &["worktree", "remove", "--force", &worktree.to_string_lossy()],
            );
        }
        let _ = std::fs::remove_dir_all(&self.root);
        let _ = git(&self.repo, &["worktree", "prune"]);
    }
}

fn git(dir: &Path, args: &[&str]) -> anyhow::Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()?;
    anyhow::ensure!(
        output.status.success(),
        "git {}: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// One run's evidence: its answer, trajectory, error and worktree changes.
fn probe_run(output: anyhow::Result<ProbeChildOutput>, worktree: &Path) -> ProbeRun {
    let output = output.unwrap_or_else(|error| ProbeChildOutput {
        error: Some(format!("{error:#}")),
        ..ProbeChildOutput::default()
    });
    let diff = git(worktree, &["add", "-A"])
        .and_then(|_| {
            Ok(format!(
                "{}\n{}",
                git(worktree, &["diff", "--cached", "--stat"])?,
                git(worktree, &["diff", "--cached"])?
            ))
        })
        .unwrap_or_default();
    ProbeRun {
        final_answer: output.answer.filter(|answer| !answer.trim().is_empty()),
        trajectory: conversation_text(&output.messages, PROBE_EVIDENCE_CHARS),
        diff: diff.trim().to_string(),
        error: output.error,
    }
}

/// The finished probe: the refinement applied with the kept edits, and the
/// probe line for the user.
pub struct ProbeOutcome {
    pub result: RefinementResult,
    pub notice: String,
}

impl AgentSession {
    /// `refine.probe.enabled`: every manual `/refine` probes.
    #[must_use]
    pub fn refine_probe_by_default(&self) -> bool {
        self.refine_probe_settings()
            .and_then(|probe| probe.enabled)
            .unwrap_or(false)
    }

    fn refine_probe_settings(&self) -> Option<crate::settings::RefineProbeSettings> {
        self.refinement_settings.as_ref().and_then(|settings| {
            settings
                .reopen()
                .settings()
                .refine
                .as_ref()
                .and_then(|refine| refine.probe.clone())
        })
    }

    /// `/refine --probe` with the session's own refiner and its RLM child
    /// host.
    ///
    /// # Errors
    ///
    /// Refuses a session without a daemon child host; otherwise see
    /// [`Self::refine_probed_with`].
    pub async fn refine_probed(
        &self,
        options: &RefineOptions,
        request: &ProbeRequest,
        model: &pa_types::ai::Model,
        api_key: Option<String>,
        global_harness_dir: PathBuf,
    ) -> anyhow::Result<ProbeOutcome> {
        let host = self.probe_children.clone().ok_or_else(|| {
            anyhow::anyhow!("/refine --probe needs a daemon-backed session: it runs its test runs as RLM children")
        })?;
        let refine_call = self.refiner_call(model, api_key.clone()).await?;
        let compose_call = self.refiner_call(model, api_key).await?;
        self.refine_probed_with(
            options,
            request,
            model,
            refine_call,
            compose_call,
            rlm_probe_runner(host),
            global_harness_dir,
        )
        .await
    }

    /// Plan, probe, compose, then apply the kept edits as one ordinary
    /// refinement (same audit row, history and notice, plus the probe line)
    /// and record the probe's per-edit verdicts.
    ///
    /// # Errors
    ///
    /// Refuses a rollback, a cwd outside a git repository, and a
    /// conversation with no task to replay. Also returns planning, compose,
    /// apply and persistence errors.
    ///
    /// # Panics
    ///
    /// Panics if a billed-call slot's mutex is poisoned.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub async fn refine_probed_with(
        &self,
        options: &RefineOptions,
        request: &ProbeRequest,
        model: &pa_types::ai::Model,
        refine_call: RefinerFn,
        compose_call: RefinerFn,
        runner: ProbeRunnerFn,
        global_harness_dir: PathBuf,
    ) -> anyhow::Result<ProbeOutcome> {
        anyhow::ensure!(
            options.rollback_id.is_none(),
            "/refine --probe cannot be combined with rollback"
        );
        let _flight = self.compaction_flight.lock().await;
        let parts = self.session.lock().await.refine_transcript_parts();
        let crate::session::manager::RefineTranscriptParts {
            messages,
            refinement_history,
        } = parts.await?;
        let messages = self.host_projected("refine", messages, &[]).await;
        let task = request
            .task
            .clone()
            .filter(|task| !task.trim().is_empty())
            .or_else(|| probe_task(&messages))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "/refine --probe found no task to replay: no prompt in this conversation led to a tool call. Pass one with --task \"...\""
                )
            })?;
        let agent_dir = self.agent_dir.clone();
        let (prepared, cwd, local_state) = {
            let session = self.session.lock().await;
            (
                prepare_refinement(
                    &session,
                    &refinement_history,
                    &global_harness_dir,
                    options,
                    agent_dir.as_deref(),
                )
                .await?,
                session.get_cwd().to_path_buf(),
                load_harness_state(&local_harness_state_dir(&session), HarnessScope::Local),
            )
        };
        // The git check runs before any model spend.
        let (workspace, relative) = ProbeWorkspace::create(&cwd, &now_millis().to_string())?;

        let (refine_call, billed) = billed_refiner_call(refine_call, "refine");
        let plan = plan_refinement(
            &messages,
            &prepared.planning_state,
            &prepared.history,
            model,
            &prepared.core_options,
            refine_call,
        )
        .await;
        let billed = billed.lock().expect("billed refiner call lock").take();
        self.record_model_usage(billed).await;
        let plan = strip_display_prefixes(plan?);
        let edit_count = plan.proposal.edits.len();

        let (decision, verdicts) = if edit_count == 0 {
            drop(workspace);
            (None, Vec::new())
        } else {
            let factory_enabled = match agent_dir.clone() {
                Some(dir) => {
                    tokio::task::spawn_blocking(move || crate::refinement::factory_enabled(&dir))
                        .await
                        .unwrap_or(false)
                }
                None => false,
            };
            let (base_state, candidate_state) = seed_states(&local_state, &plan, factory_enabled);
            let timeout = Duration::from_millis(
                self.refine_probe_settings()
                    .and_then(|probe| probe.timeout_ms)
                    .unwrap_or(DEFAULT_PROBE_TIMEOUT_MS),
            );
            let mut specs = Vec::new();
            for (label, state) in [("base", &base_state), ("candidate", &candidate_state)] {
                let seed = workspace.root.join(format!("harness-{label}"));
                crate::refinement::save_harness_state(&seed, state)?;
                specs.push(ProbeChildSpec {
                    label,
                    cwd: workspace.worktree(label).join(&relative),
                    harness_seed: seed,
                    task: task.clone(),
                    model: format!("{}/{}", model.provider, model.id),
                    timeout,
                });
            }
            let (base_output, candidate_output) =
                tokio::join!(runner(specs[0].clone()), runner(specs[1].clone()));
            let base = probe_run(base_output, &workspace.worktree("base"));
            let candidate = probe_run(candidate_output, &workspace.worktree("candidate"));
            drop(workspace);

            let (compose_call, billed) = billed_refiner_call(compose_call, "refine_probe");
            let mut compose_model = model.clone();
            compose_model.max_tokens = compose_model
                .max_tokens
                .min(AUTO_REFINE_REVIEW_MAX_OUTPUT_TOKENS);
            let max_tokens = compose_model.max_tokens;
            let reply = compose_call(
                compose_model,
                PROBE_COMPOSE_SYSTEM_PROMPT,
                compose_prompt(&task, &plan, &base, &candidate),
            )
            .await;
            let billed = billed.lock().expect("billed refiner call lock").take();
            self.record_model_usage(billed).await;
            let text = reply_text(&reply?, max_tokens, "Probe compose failed")?;
            let decision = parse_compose(&text, edit_count, candidate.delivered())?;
            let verdicts = probe_verdicts(
                &decision,
                &plan,
                &plan.id,
                &crate::session::manager::format_iso_now(),
            );
            (Some(decision), verdicts)
        };
        let notice = decision.as_ref().map_or_else(
            || "probe: nothing to probe, the proposal had no edits".to_string(),
            |decision| probe_notice(decision, edit_count),
        );
        let plan = match &decision {
            Some(decision) => kept_plan(plan, decision),
            None => plan,
        };
        let (result, context_row_ids) = {
            let mut session = self.session.lock().await;
            let applied = apply_prepared_refinement(
                &mut session,
                prepared,
                plan,
                &global_harness_dir,
                RefinementSource::User,
                agent_dir.as_deref(),
                Some(&notice),
            )
            .await?;
            record_refinement_verdicts(
                &mut session,
                &global_harness_dir,
                std::slice::from_ref(&applied.0),
                &verdicts,
            )?;
            applied
        };
        let rows = {
            let session = self.session.lock().await;
            context_rows_by_ids(session.retained_entries(), &context_row_ids)
        };
        let loop_rows: Vec<_> = rows
            .iter()
            .filter_map(super::session_message_to_loop)
            .collect();
        if !loop_rows.is_empty() {
            self.agent.append_messages(loop_rows).await;
        }
        Ok(ProbeOutcome { result, notice })
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
        })
}
