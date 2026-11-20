//! Session creation and reuse on the worker: the create command's
//! construction of the live session.
use super::{
    json, paths, response_failure, response_success, restore_queue_snapshot, session_file_name,
    Arc, EngineModelSelection, Result, RlmSessionIdentity, SessionEngine, SessionFile, VecDeque,
    Worker,
};

use serde::Deserialize as _;
use serde_json::Value;

use crate::agent_engine::CreateSessionResources;
use crate::protocol::DaemonResponse;

impl Worker {
    /// The upstream that sends this child's requests to its parent session
    /// as `request_tool_permission`, stamped with the child's identity. A
    /// failed or unreadable answer is no answer: the call is denied.
    fn tool_permission_upstream(
        &self,
        parent_active_session_id: &str,
        rlm_child_id: Option<&str>,
        name: Option<&str>,
    ) -> crate::tool_permission::ToolPermissionUpstream {
        let link = Arc::new(crate::supervisor_link::SupervisorLink::new(
            self.config.supervisor_socket_path.clone(),
        ));
        let parent = parent_active_session_id.to_string();
        let origin = json!({
            "activeSessionId": self.config.active_session_id,
            "rlmChildId": rlm_child_id,
            "sessionName": name,
        });
        Arc::new(move |request| {
            let link = Arc::clone(&link);
            let mut command = request;
            command["type"] = json!("request_tool_permission");
            command["activeSessionId"] = json!(parent);
            // A grandchild's request keeps the origin it started with.
            if command.get("origin").is_none_or(Value::is_null) {
                command["origin"] = origin.clone();
            }
            Box::pin(async move {
                let answer = link
                    .request_success(
                        command,
                        crate::tool_permission::TOOL_PERMISSION_TIMEOUT
                            + std::time::Duration::from_secs(60),
                    )
                    .await
                    .ok()?;
                answer
                    .get("outcome")
                    .and_then(Value::as_str)
                    .and_then(crate::tool_permission::ToolPermissionOutcome::parse)
            })
        })
    }

    /// The upstream that sends this child's host hooks to its parent session
    /// as `request_host_hook`, stamped with the child's identity. A failed
    /// or unreadable answer is an error, which each hook settles by its kind
    /// (the enforcement gate refuses).
    fn host_hook_upstream(
        &self,
        parent_active_session_id: &str,
        rlm_child_id: Option<&str>,
        name: Option<&str>,
        hook_timeout: std::time::Duration,
    ) -> crate::host_hooks::HostHookUpstream {
        let link = Arc::new(crate::supervisor_link::SupervisorLink::new(
            self.config.supervisor_socket_path.clone(),
        ));
        let parent = parent_active_session_id.to_string();
        let core = Arc::clone(&self.core);
        let active_session_id = self.config.active_session_id.clone();
        let rlm_child_id = rlm_child_id.map(str::to_string);
        let name = name.map(str::to_string);
        Arc::new(move |hook, payload, origin| {
            let link = Arc::clone(&link);
            let command = json!({
                "type": "request_host_hook",
                "activeSessionId": parent,
                "hook": hook,
                "payload": payload,
                // A grandchild's hook keeps the origin it started with.
                "origin": origin.unwrap_or_else(|| {
                    let core = core.lock().unwrap();
                    json!({
                        "activeSessionId": active_session_id,
                        "sessionId": core.store.as_ref().map(SessionFile::session_id),
                        "sessionFile": core.store.as_ref().map(|store| store.path.to_string_lossy()),
                        "cwd": core.cwd,
                        "parentSessionId": core.parent_session_id,
                        "parentActiveSessionId": core.parent_active_session_id,
                        "generation": core.generation,
                        "eventSequence": core.last_event_sequence,
                        "admissionIds": core.running_admission_ids,
                        "rlmChildId": rlm_child_id,
                        "sessionName": name,
                    })
                }),
            });
            Box::pin(async move {
                let answer = match link
                    .request_success(command, hook_timeout + std::time::Duration::from_secs(60))
                    .await
                {
                    Ok(answer) => answer,
                    Err(error) => {
                        return Some(Err(format!("the parent session did not answer ({error})")))
                    }
                };
                if answer.get("off").and_then(Value::as_bool) == Some(true) {
                    return None;
                }
                Some(match answer.get("error").and_then(Value::as_str) {
                    Some(error) => Err(error.to_string()),
                    None => Ok(answer.get("result").cloned().unwrap_or(Value::Null)),
                })
            })
        })
    }

    pub(super) async fn handle_create(&self, payload: &Value) -> DaemonResponse {
        // One create in flight at a time: a concurrent create joins this open and
        // answers with the created summary instead of racing a second init.
        let _create_gate = self.create_gate.lock().await;
        let existing_summary = {
            let (core, inputs) = self.summary_inputs();
            core.created.then(|| self.summary_locked(&core, inputs))
        };
        if let Some(summary) = existing_summary {
            // Idempotent re-create after a supervisor restart or respawn.
            // A respawned worker's re-create re-binds the pane reporter
            // (the session may carry a fresh client env on the payload)
            // and re-reports: a supervisor restart must not leave the
            // pane holding a stale pre-restart state.
            self.rebind_herdr_reporter(payload);
            return response_success(
                None,
                "create",
                Some(serde_json::to_value(&summary).unwrap_or(Value::Null)),
            );
        }
        let session_path = match payload.get("sessionPath").and_then(Value::as_str) {
            Some(path) => match paths::expand_tilde(path) {
                Ok(expanded) => Some(expanded),
                Err(error) => return response_failure(None, "create", &error.to_string(), None),
            },
            None => None,
        };
        let studio_servers: Vec<pa_core::mcp::AcpMcpServerConfig> = match payload
            .get("studioMcpServers")
            .filter(|_| {
                payload
                    .get("studioHostContractVersion")
                    .and_then(Value::as_u64)
                    == Some(1)
            })
            .cloned()
            .map(serde_json::from_value)
            .transpose()
        {
            Ok(servers) => servers.unwrap_or_default(),
            Err(error) => return response_failure(None, "create", &error.to_string(), None),
        };
        let no_session = payload
            .get("noSession")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let name = payload.get("name").and_then(Value::as_str);
        let flagged_model = payload.get("model").and_then(Value::as_str).is_some();
        // A revived session restores the model its file pins before the startup
        // chain; an explicit model flag wins. The flags are authoritative for
        // this worker's runtime config and survive every session replacement.
        let requested_thinking = match payload.get("thinking") {
            None => None,
            Some(Value::String(level)) => {
                match pa_ai::models::thinking_level_from_str(level) {
                    Some(level) => Some(level),
                    // The wire contract takes validated levels only: reject
                    // the create loudly instead of silently dropping it.
                    None => {
                        return response_failure(
                            None,
                            "create",
                            &format!(
                                "Invalid thinking level \"{level}\". Valid values: off, minimal, low, medium, high, xhigh, max"
                            ),
                            None,
                        );
                    }
                }
            }
            Some(_) => {
                return response_failure(
                    None,
                    "create",
                    "Invalid thinking level: expected a string",
                    None,
                );
            }
        };
        self.engine.configure_create_model(EngineModelSelection {
            provider: payload
                .get("provider")
                .and_then(Value::as_str)
                .map(str::to_string),
            model: payload
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string),
            api_key: payload
                .get("apiKey")
                .and_then(Value::as_str)
                .map(str::to_string),
            thinking: requested_thinking,
        });
        // Folded before the background build spawn at the create's tail,
        // so the first build sees them.
        let resources = match CreateSessionResources::deserialize(payload) {
            Ok(resources) => resources,
            Err(error) => {
                return response_failure(
                    None,
                    "create",
                    &format!("Invalid create config: {error}"),
                    None,
                );
            }
        };
        if let Some(agent_engine) = &self.agent_engine {
            if let Some(autonomous) = &resources.autonomous {
                *agent_engine.autonomous.lock().await =
                    pa_core::autonomous::create_autonomous_runtime_state(Some(autonomous), None);
            }
            *agent_engine
                .create_resources
                .write()
                .expect("create resources lock") = resources;
        }
        let cwd = payload
            .get("cwd")
            .and_then(Value::as_str)
            .unwrap_or("/")
            .to_string();
        let session_dir = match payload.get("sessionDir").and_then(Value::as_str) {
            Some(dir) => match paths::expand_tilde(dir) {
                Ok(expanded) => expanded,
                Err(error) => return response_failure(None, "create", &error.to_string(), None),
            },
            None => match paths::sessions_dir(&self.config.agent_dir) {
                Ok(dir) => dir,
                Err(error) => return response_failure(None, "create", &error.to_string(), None),
            },
        };
        // RLM recursion identity (children of an RLM parent run at depth+1):
        // the durable create replays these so a respawned child keeps them.
        let (rlm_depth, rlm_max_depth) = match create_payload_rlm_depth(payload) {
            Ok(identity) => identity,
            Err(error) => return response_failure(None, "create", &error, None),
        };
        let parent_session_path = payload
            .get("parentSessionPath")
            .and_then(Value::as_str)
            .map(str::to_string);
        // The subagent runtime identity (TS `runtimeMetadata` on the create
        // command): the child id and the parent's live/persisted ids ride
        // the session summaries so the roster can key children
        // `parentPath#childId` like TS `rosterAgentIdForSummary`. The
        // semantic spawn origin (TS `semanticParentSessionId` +
        // `semanticSpawnedByRequestId`) exists only for this arm: a
        // resumed saved subagent file is a top-level runtime and spawns
        // no edge.
        let (rlm_child_id, parent_active_session_id, parent_session_id, semantic_spawn) =
            match payload.get("runtimeMetadata") {
                Some(metadata)
                    if metadata.get("kind").and_then(Value::as_str) == Some("subagent") =>
                {
                    let parent_session_id = metadata
                        .get("parentSessionId")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    (
                        metadata
                            .get("rlmChildId")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        metadata
                            .get("parentActiveSessionId")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        parent_session_id.clone(),
                        Some(crate::engine::SemanticSpawnOrigin {
                            parent_session_id,
                            spawned_by_request_id: payload
                                .get("spawnedByRequestId")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                        }),
                    )
                }
                _ => (None, None, None, None),
            };
        let factory_child = payload
            .get("runtimeMetadata")
            .is_some_and(|metadata| metadata.get("factory").is_some());
        let thinking = payload
            .get("thinking")
            .and_then(Value::as_str)
            .map(str::to_string);
        // Verification seam (TS children inherit the parent's `sessionConfig`): a
        // scripted parent passes its children's engine file down; product creates carry `None`.
        let child_script = payload
            .get("childScript")
            .and_then(Value::as_str)
            .map(str::to_string);
        // A child spawned by an ask-mode parent asks that parent for every
        // tool call (the durable create replays the key on a respawn).
        if let Some(parent) = payload.get("toolPermissionParent").and_then(Value::as_str) {
            self.tool_permissions
                .set_upstream(self.tool_permission_upstream(parent, rlm_child_id.as_deref(), name));
        }
        // A child of a hooked session runs its parent's hooks through it (the
        // durable create replays the key on a respawn).
        if let Some(inherited) = payload.get("hostHooksParent") {
            self.host_hooks.set_contract(
                inherited.get("hostContractVersion").and_then(Value::as_u64) == Some(1),
            );
            let parent = inherited
                .get("activeSessionId")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let hooks: Vec<String> = inherited
                .get("hooks")
                .and_then(Value::as_array)
                .map(|hooks| {
                    hooks
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let host_call_types = inherited.get("hostCallTypes").and_then(|types| {
                serde_json::from_value::<Option<Vec<String>>>(types.clone())
                    .ok()
                    .flatten()
            });
            let timeout = std::time::Duration::from_millis(
                inherited
                    .get("timeoutMs")
                    .and_then(Value::as_u64)
                    .unwrap_or(30_000),
            );
            if parent.is_empty()
                || self
                    .host_hooks
                    .set_hooks(&hooks, host_call_types, Some(timeout))
                    .is_err()
            {
                // An unreadable inheritance must not leave the child unguarded.
                return response_failure(
                    None,
                    "create",
                    "The parent session's host hooks could not be inherited",
                    None,
                );
            }
            let owner = inherited
                .get("hostOwnerToken")
                .and_then(Value::as_str)
                .map(str::to_string);
            *self.host_hook_owner.lock().await = owner.clone();
            self.host_hooks.set_owner_token(owner);
            self.host_hooks.set_upstream(self.host_hook_upstream(
                parent,
                rlm_child_id.as_deref(),
                name,
                timeout,
            ));
        }

        let host_start_pause = payload
            .get("hostHooksParent")
            .filter(|_| self.host_hooks.lifecycle_enabled("session_start"))
            .map(|_| self.hold_queued_work_pause("host-session-start"));

        // Set by the fresh-path arm when the name landed in its single rewrite:
        // the shared name persist must not append a second `session_info` line.
        let mut name_persisted_by_fresh_arm = false;
        // Set by the continuing arm when it OPENED an existing session file:
        // `is_continuing` reads this arm fact, never an existence check.
        let mut opened_existing_session = false;
        let mut restored_tier: Option<Option<pa_types::ai::ServiceTier>> = None;
        // The fresh arms defer their creation prefix to after the startup scope
        // registers, so a fresh `--models` session persists the scoped startup pick.
        let mut fresh_prefix = FreshPrefixPlan::None;
        let (settings, trace_consent) = pa_core::agent_traces::ContinuousTraceUpload::load_settings(
            std::path::Path::new(&cwd),
            &self.config.agent_dir,
        );
        let traces = |path: &std::path::Path| {
            pa_core::agent_traces::ContinuousTraceUpload::install(
                std::path::Path::new(&cwd),
                &self.config.agent_dir,
                Some(path),
                trace_consent.clone(),
            )
        };
        let mut store = match (&session_path, no_session) {
            (Some(path), false) if path.exists() => {
                let loaded = {
                    let path = path.clone();
                    let agent_dir = self.config.agent_dir.clone();
                    tokio::task::spawn_blocking(move || {
                        let lease = crate::lease::acquire_runtime_session_lease(&path, &agent_dir)?;
                        pa_core::session::manager::repair_jsonl_damage(&path);
                        let mut store = SessionFile::open_windowed(&path)?;
                        store.lease = Some(Arc::new(lease));
                        Ok(store)
                    })
                    .await
                    .map_err(anyhow::Error::from)
                    .and_then(|result| result)
                };
                match loaded {
                    Ok(mut opened) => {
                        opened.trace_upload = traces(&opened.path);
                        opened_existing_session = true;
                        if opened.skipped_lines > 0 {
                            // The rows stay on disk (the append-only
                            // reopen): the skip count is the damage
                            // report the torn-tail repair can act on — a
                            // silent skip is how a torn session degraded
                            // unnoticed (the operator's 2026-10-08
                            // report).
                            eprintln!(
                                "pa-daemon: session {} skipped {} unparsable row(s) on open; they stay on disk",
                                path.display(),
                                opened.skipped_lines
                            );
                        }
                        // The session-model restore records its decision only for a path
                        // this worker opened — a failed open never leaks the binding into a
                        // later create.
                        self.engine.set_session_file(path.clone());
                        // One fold serves all consumers: the model restore takes its saved
                        // context off the store this create just opened.
                        let restored = opened.restored_settings();
                        restored_tier = opened.has_service_tier().then_some(restored.service_tier);
                        let has_thinking_level = opened.has_thinking_level();
                        let saved = crate::agent_engine::saved_session_context_from_parts(
                            &restored,
                            has_thinking_level,
                        );
                        if !flagged_model {
                            self.engine
                                .restore_session_model(path, Some(saved.clone()))
                                .await;
                        }
                        // The saved thinking level restores here when the create carries no
                        // explicit flag; the saved MODEL restores through the engine's
                        // session-model restore.
                        self.engine.configure_model(EngineModelSelection {
                            provider: None,
                            model: None,
                            api_key: None,
                            thinking: requested_thinking.or(saved.thinking),
                        });
                        let append_start = opened.entries.len();
                        append_creation_prefix(
                            &mut opened,
                            self.engine.as_ref(),
                            &self.config.agent_dir,
                            &cwd,
                            false,
                        );
                        let _ = opened.append_session_state("active");
                        // A reopen is APPEND-ONLY (the operator's 2026-10-08
                        // report: a reopened session showed none of the old
                        // messages): the full reader skips malformed rows in
                        // memory, and the legacy full-file rewrite this
                        // arm carried DELETED them from disk — a gap early
                        // in the parent chain took the whole transcript
                        // with it. Only the rows this open appended
                        // persist; the file keeps every original byte for
                        // the torn-tail repair to see.
                        let persisted = opened.persist_appended(append_start);
                        if let Err(error) = persisted {
                            return response_failure(None, "create", &error.to_string(), None);
                        }
                        // Prime the usage fold on the file's final identity,
                        // off the runtime and before the core lock:
                        // summaries under the lock fold only the appended
                        // tail.
                        let primed = path.clone();
                        let _ = tokio::task::spawn_blocking(move || {
                            crate::session_store::read_session_info(&primed)
                        })
                        .await;
                        opened
                    }
                    Err(error) => return crate::hold_refusal::create_failure_response(&error),
                }
            }
            (Some(path), false) => {
                let mut created = SessionFile::create(
                    &cwd,
                    parent_session_path.as_deref(),
                    rlm_depth.unwrap_or(0),
                );
                created.set_path(path.clone());
                created.trace_upload = traces(&created.path);
                let acquired = {
                    let path = path.clone();
                    let agent_dir = self.config.agent_dir.clone();
                    tokio::task::spawn_blocking(move || {
                        crate::lease::acquire_runtime_session_lease(&path, &agent_dir)
                    })
                    .await
                    .map_err(anyhow::Error::from)
                    .and_then(|lease| lease)
                };
                match acquired {
                    Ok(lease) => created.lease = Some(Arc::new(lease)),
                    Err(error) => return crate::hold_refusal::create_failure_response(&error),
                }
                if let Err(error) = created.rewrite() {
                    return response_failure(None, "create", &error.to_string(), None);
                }
                fresh_prefix = FreshPrefixPlan::PathBacked { fold_name: false };
                created
            }
            // In-memory session: no file, like the TS `noSession` create.
            (None, true) => {
                let created = SessionFile::create(
                    &cwd,
                    parent_session_path.as_deref(),
                    rlm_depth.unwrap_or(0),
                );
                fresh_prefix = FreshPrefixPlan::InMemory;
                created
            }
            (None, false) => {
                let mut created = SessionFile::create(
                    &cwd,
                    parent_session_path.as_deref(),
                    rlm_depth.unwrap_or(0),
                );
                let path = session_dir.join(session_file_name(created.session_id()));
                created.set_path(path.clone());
                created.trace_upload = traces(&created.path);
                let acquired = {
                    let path = path.clone();
                    let agent_dir = self.config.agent_dir.clone();
                    tokio::task::spawn_blocking(move || {
                        crate::lease::acquire_runtime_session_lease(&path, &agent_dir)
                    })
                    .await
                    .map_err(anyhow::Error::from)
                    .and_then(|lease| lease)
                };
                match acquired {
                    Ok(lease) => created.lease = Some(Arc::new(lease)),
                    Err(error) => return crate::hold_refusal::create_failure_response(&error),
                }
                // One durable write instead of three: the prefix, the `active` state,
                // and the session name land in a single rewrite; the create stays
                // pathless until it succeeds, so no reader reads the file mid-create.
                fresh_prefix = FreshPrefixPlan::PathBacked { fold_name: true };
                created
            }
            (Some(_), true) => {
                return response_failure(
                    None,
                    "create",
                    "Session cannot be both no-session and session-pathed",
                    None,
                );
            }
        };
        // The worker owns the session file; the engine reads it for the
        // system prompt's conversation-log path and the local harness dir.
        if !store.path.as_os_str().is_empty() {
            self.engine.set_session_file(store.path.clone());
        }
        // A `/refine --probe` run starts from the harness its parent seeded:
        // `harnessSeed` names a harness directory whose state becomes this
        // session's local store. The session is pinned to that store of its
        // own, so it neither reads nor writes its project's. Only a fresh
        // store is seeded, so a respawn keeps what the run itself wrote.
        if let Some(seed) = payload.get("harnessSeed").and_then(Value::as_str) {
            let local =
                crate::scheduled_jobs::session_artifact_dir(&store.path, store.session_id())
                    .and_then(|dir| pa_core::refinement::get_local_harness_state_dir(Some(&dir)));
            if let Some(local) = local {
                if !pa_core::refinement::get_harness_state_path(&local).exists() {
                    let state = pa_core::refinement::load_harness_state(
                        std::path::Path::new(seed),
                        pa_core::refinement::HarnessScope::Local,
                    );
                    let pinned =
                        pa_core::refinement::save_harness_state(&local, &state).and_then(|_| {
                            Ok(std::fs::write(
                                local.join(
                                    pa_core::refinement::project::SESSION_STORE_MARKER_FILE_NAME,
                                ),
                                "",
                            )?)
                        });
                    if let Err(error) = pinned {
                        return response_failure(None, "create", &error.to_string(), None);
                    }
                }
            }
        }

        // The `models` patterns (else settings `enabledModels`) resolve once into
        // the scoped list, AFTER the store open (the saved-model restore's wait
        // already covered the catalog fetch): resolving earlier stores an empty scope.
        let model_patterns = match payload.get("models").and_then(Value::as_array) {
            Some(patterns) => patterns
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<String>>(),
            None => settings.get_enabled_models().unwrap_or_default(),
        };
        let is_continuing = opened_existing_session;
        let scoped_models = if model_patterns.is_empty() {
            Vec::new()
        } else {
            let mut registry = crate::state_getters::worker_model_registry(&self.config.agent_dir);
            pa_core::models::resolve_model_scope_from_models_with_readiness(
                &mut registry,
                &model_patterns,
                settings
                    .get_default_provider()
                    .zip(settings.get_default_model()),
                is_continuing,
                pa_core::models::SESSION_MODEL_RESTORE_READINESS_TIMEOUT_MS,
            )
            .await
        };
        self.engine.configure_startup_scope(
            scoped_models.clone(),
            model_patterns.clone(),
            is_continuing,
        );
        // The fresh arms' deferred creation prefix runs here, against the
        // registered scope; the continuing arm wrote its own prefix above, so
        // the scope never changes a continuing file's pinned model and thinking.
        match fresh_prefix {
            FreshPrefixPlan::None => {}
            FreshPrefixPlan::InMemory => {
                append_creation_prefix(
                    &mut store,
                    self.engine.as_ref(),
                    &self.config.agent_dir,
                    &cwd,
                    true,
                );
            }
            FreshPrefixPlan::PathBacked { fold_name } => {
                append_creation_prefix(
                    &mut store,
                    self.engine.as_ref(),
                    &self.config.agent_dir,
                    &cwd,
                    true,
                );
                let _ = store.append_session_state("active");
                if fold_name {
                    if let Some(name) = name.filter(|n| !n.trim().is_empty()) {
                        store.append_session_info(name);
                        name_persisted_by_fresh_arm = true;
                    }
                }
                if let Err(error) = store.rewrite() {
                    return response_failure(None, "create", &error.to_string(), None);
                }
            }
        }
        // The wire shape `set_scoped_models` stores (the connection state
        // surface and the cycler's input): `{ model, thinkingLevel? }`.
        let scoped_entries: Vec<Value> = scoped_models
            .iter()
            .map(|scoped| {
                let mut entry = json!({ "model": scoped.model });
                // The pattern's `:thinking` suffix; the wire name is the level the
                // cycler and the state readers parse back.
                if let Some(level) = scoped.thinking_level {
                    entry["thinkingLevel"] = json!(level);
                }
                entry
            })
            .collect();

        if !name_persisted_by_fresh_arm {
            if let Some(name) = name.filter(|n| !n.trim().is_empty()) {
                if let Err(error) =
                    store.persist_entry("session_info", json!({ "name": name.trim() }))
                {
                    return response_failure(None, "create", &error.to_string(), None);
                }
            }
        }
        // Restore the persisted queue snapshot (crash/respawn recovery) from
        // the worker recovery journal.
        let (steering, follow_up, restored_suspended) = {
            let guard = self.recovery.lock().unwrap();
            match guard.as_ref() {
                Some(journal) => restore_queue_snapshot(journal, &self.config.active_session_id),
                None => (VecDeque::new(), VecDeque::new(), false),
            }
        };
        // The session's settings-seeded switches: a restarted session re-seeds
        // its auto-compaction flag from the persisted `compaction.enabled`.
        let (service_tier, steering_mode, follow_up_mode, auto_compaction_enabled) = {
            let queue_mode = |mode: pa_core::settings::QueueModeSetting| -> String {
                match mode {
                    pa_core::settings::QueueModeSetting::All => "all".to_string(),
                    pa_core::settings::QueueModeSetting::OneAtATime => "one-at-a-time".to_string(),
                }
            };
            (
                settings.get_default_service_tier(),
                queue_mode(settings.get_steering_mode()),
                queue_mode(settings.get_follow_up_mode()),
                settings.get_compaction_enabled(),
            )
        };
        // The ACTIVE tier clamps to the model's support; the stored preference
        // keeps the requested tier, so an ineligible model shows the degraded tier.
        let configured_tier = restored_tier.unwrap_or(Some(service_tier));
        let clamped_tier =
            crate::setting_switches::effective_service_tier(configured_tier, self.engine.as_ref());
        self.engine.configure_service_tier(clamped_tier);
        // The abort supervision's terminal record: the rebuilt transcript discloses
        // the abort with the same `compaction_outcome` row the auto-abort arms persist
        // (a manual run persists nothing); the dedup matches the row's fields alone.
        let interrupted_compaction_requested = payload.get("interruptedCompaction").is_some();
        let interrupted_compaction = crate::compaction::interrupted_compaction_disclosure(payload);
        // The disclosure row's landing state: the supervisor consumes the terminal
        // record only once the disclosure is durable (a requested record with no
        // disclosure row is vacuously durable; no requested record adds no key).
        let mut interrupted_compaction_persisted = interrupted_compaction_requested;
        // MCP admission precedes the created flag: a failed admission is not
        // an idempotently successful create on the next request.
        if !studio_servers.is_empty() {
            let binding = json!({
                "sessionId": store.session_id(),
                "activeSessionId": self.config.active_session_id,
                "sessionFile": store.path.to_string_lossy(),
                "cwd": cwd,
                "parentSessionId": payload.get("studioMcpParentSessionId"),
            });
            let servers = bind_studio_mcp_servers(studio_servers, &binding);
            let manager = self.session_mcp_manager();
            let owner = format!("studio-native:{}", self.config.active_session_id);
            // Only the inherited Studio sidecar is enrolled for a credential.
            let studio = if servers
                .iter()
                .any(|server| server.name() == pa_core::mcp::STUDIO_MCP_SERVER)
            {
                pa_core::mcp::StudioUpdate::Bind(pa_core::mcp::STUDIO_MCP_SERVER)
            } else {
                pa_core::mcp::StudioUpdate::Unbind
            };
            if let Err(error) = manager
                .lock()
                .unwrap()
                .replace_acp_servers_with_studio(&servers, &owner, studio)
            {
                return response_failure(None, "create", &error.to_string(), None);
            }
            if let Some(children) = self
                .agent_engine
                .as_ref()
                .and_then(|engine| engine.children.as_ref())
            {
                children.set_studio_mcp_servers(servers);
            }
        }
        // The core lock stays inside this block: everything after it may await,
        // and a std MutexGuard must never ride an await point.
        let (summary, rlm_depth) = {
            let (mut core, inputs) = self.summary_inputs();
            core.cwd = cwd;
            core.steering = steering;
            core.follow_up = follow_up;
            // The restored lanes carry their suspension: a user-aborted queue
            // revives still parked, never auto-running its prompts.
            core.queued_input_suspended = restored_suspended;
            // A resumed session's tool counter re-seeds from its file: a
            // revived worker keeps the count its roster rows carried.
            core.tool_use_count = store.tool_call_count();
            core.store = Some(store);
            if let Some(disclosure) = &interrupted_compaction {
                if let Some(store) = core.store.as_mut() {
                    let already_disclosed = store.entries().iter().any(|entry| {
                        entry.type_ == "custom_message" && entry.fields == disclosure.row
                    });
                    if !already_disclosed
                        && store
                            .persist_entry_at(
                                "custom_message",
                                disclosure.row.clone(),
                                &disclosure.declared_at,
                            )
                            .is_err()
                    {
                        interrupted_compaction_persisted = false;
                    }
                }
            }
            core.created = true;
            core.abort_requested = false;
            core.turn_cancelled = false;
            core.turn_completed = false;
            // A fresh session starts live: the previous close's
            // `session_closed` marker clears with the new session.
            if let Some(agent_engine) = &self.agent_engine {
                agent_engine.clear_session_closed();
            }
            core.auto_compaction_enabled = auto_compaction_enabled;
            core.service_tier = configured_tier;
            core.active_service_tier = clamped_tier;
            core.steering_mode.clone_from(&steering_mode);
            core.follow_up_mode.clone_from(&follow_up_mode);
            // The restored rows carry their forced-batch flag: the batch
            // gate rides the lane, so a restored forced batch keeps its
            // one-turn semantics.
            core.forced_all_steering = core.steering.iter().any(|item| item.forced_batch);
            core.scoped_models.clone_from(&scoped_entries);
            core.retry_abort_requested = false;
            // The session's depth falls back to the opened file's header: a resumed
            // subagent file carries its persisted depth, so the roster never re-nests
            // it under its original parent.
            let rlm_depth = rlm_depth
                .or_else(|| core.store.as_ref().and_then(SessionFile::rlm_depth))
                .unwrap_or(0);
            core.rlm_depth = rlm_depth;
            core.runtime_kind = if rlm_child_id.is_some() {
                "subagent".to_string()
            } else {
                "top-level".to_string()
            };
            core.rlm_child_id = rlm_child_id;
            core.parent_active_session_id = parent_active_session_id;
            core.parent_session_id = parent_session_id;
            core.child_script.clone_from(&child_script);
            (self.summary_locked(&core, inputs), rlm_depth)
        };
        // The engine's agent-level queues drain per the same modes the
        // worker lane delivers by.
        self.engine
            .set_queue_modes(Some(&steering_mode), Some(&follow_up_mode));
        // Seed the engine's RLM identity: recursion depth and bound, the session's
        // persistence ids, the children's default thinking level.
        if let Err(error) = self.engine.configure_rlm_identity(RlmSessionIdentity {
            rlm_depth,
            rlm_max_depth,
            cwd: Some(summary.cwd.clone()),
            session_id: Some(summary.session_id.clone()),
            session_file: summary.session_file.clone(),
            thinking,
            child_script: child_script.clone(),
            semantic_spawn,
            factory_child,
        }) {
            return response_failure(None, "create", &error.to_string(), None);
        }
        if opened_existing_session {
            self.reseed_rlm_children().await;
        }
        // The built-in Herdr connector binds here, per session: the pane
        // identity comes from the create payload's client env (the client
        // that owns the pane sent it), never from this process's ambient
        // environment — so sessions created in other panes report their
        // own panes regardless of where the supervisor booted (the TS
        // boot-context bug class, not reproduced). A live RLM child never
        // reports: it shares the parent's pane, and a child's turn or quit
        // must not flip or release it. The bind sits PAST the create's
        // last fallible step, so a create that fails never publishes an
        // idle claim for a session the supervisor then tears down (the
        // failed create would otherwise leave the pane ghost-claimed —
        // the force-kill path releases nothing).
        self.rebind_herdr_reporter(payload);
        // The engine renders this summary into the sender identity block
        // of worker-to-worker agent messages.
        if let Ok(summary_value) = serde_json::to_value(&summary) {
            self.engine.set_session_summary(summary_value);
        }
        // Deliberate divergence from TS's eager create build: the response stays
        // model-independent; a build failure surfaces on the first demand seam.
        if let Some(agent_engine) = &self.agent_engine {
            let engine = std::sync::Arc::clone(agent_engine);
            tokio::spawn(async move {
                let Ok(model) = engine.resolve_model() else {
                    return;
                };
                let _ = engine.ensure_core_session_async(&model).await;
            });
        }
        // Bind the schedule catalog onto the session (artifact partition,
        // job rebind, scheduler start) — TS `rebindCronJobsToState`.
        if let Err(error) = self.bind_scheduled_jobs().await {
            return response_failure(None, "create", &error.to_string(), None);
        }
        // Recovery journal writes must not happen while holding the core
        // lock: record_recovery locks the core to read the store.
        let _ = self.record_recovery(true, "create");
        let session_id = summary.session_id.clone();
        if let Some(registration) = &self.registration {
            registration.notify_session_created(session_id);
        }
        // Fire the same background `refreshAvailableModels` as TS session boot: the
        // effect is the on-disk cache file; failures fall back to the cached or
        // bundled catalog without touching the session.
        let agent_dir = self.config.agent_dir.clone();
        tokio::spawn(async move {
            let auth = pa_core::auth::AuthStorage::create(&agent_dir);
            let mut registry =
                pa_core::models::ModelRegistry::create(auth, agent_dir.join("models.json"));
            let _ = registry.refresh_available_models().await;
        });
        self.work_notify.notify_one();
        // Warm the context-tree cache at session open, so an early `/context`
        // answers from it instead of walking the artifact tree inline.
        self.poke_context_tree_refresh();
        // The delete boundary invalidates the cache's rows for the deleted child
        // immediately (the background refresh would otherwise keep its last row).
        if let Some(agent_engine) = &self.agent_engine {
            if let Some(children) = &agent_engine.children {
                let cache = std::sync::Arc::clone(&self.context_tree);
                children.set_delete_notifier(std::sync::Arc::new(move |child_id| {
                    cache.invalidate_child(child_id);
                }));
                let core = std::sync::Arc::clone(&self.core);
                let events = self.events.clone();
                children.set_child_update_sink(std::sync::Arc::new(move |mut child| {
                    if let Some(parent_id) = core.lock().unwrap().rlm_child_id.clone() {
                        child["parentId"] = json!(parent_id);
                    }
                    crate::user_bash::emit_session_event_frame(
                        &core,
                        &events,
                        json!({ "type": "rlm_child_update", "child": child }),
                    );
                }));
            }
        }
        let mut data = serde_json::to_value(&summary).unwrap_or(Value::Null);
        if interrupted_compaction_requested {
            data["interruptedCompactionPersisted"] =
                serde_json::json!(interrupted_compaction_persisted);
        }
        if let Some(start_pause) = host_start_pause {
            let hooks = Arc::clone(&self.host_hooks);
            // Create must return before callback reads can pass supervisor route readiness.
            // Queued input, including restored lanes, stays paused until the real hook settles.
            tokio::spawn(async move {
                let _pause = start_pause;
                hooks.lifecycle_hook()(json!({
                    "type": "session_start",
                    "reason": "startup",
                }))
                .await;
            });
        }
        // A resumed create just rebuilt the store from the session
        // file: return the load's freed heap to the OS.
        pa_types::memory_release::trim_freed_heap();
        response_success(None, "create", Some(data))
    }

    /// (Re)bind the pane reporter from the create payload (the TS
    /// `session_start` hook): resolve the client env the create carried,
    /// start the reporter for this session's Herdr pane when the session
    /// runs in one, and force-publish the current state with this
    /// session's reference. The previous reporter (a replaced session's,
    /// or a respawn's) is dropped here — its task goes silent without a
    /// release, exactly like the TS replacement arm, so it cannot race
    /// this session's reports on the pane.
    ///
    /// # Panics
    ///
    /// Panics on a poisoned session-core mutex (a holder panicked while
    /// holding it — the worker's standing convention).
    pub(super) fn rebind_herdr_reporter(&self, payload: &Value) {
        let client_env: std::collections::BTreeMap<String, String> = payload
            .get("env")
            .cloned()
            .and_then(|env| serde_json::from_value(env).ok())
            .map(|env| crate::herdr::filter_client_env(&env))
            .unwrap_or_default();
        // TS keys the child skip on the SPAWN OVERRIDE ONLY
        // (`sessionOptionsOverride?.rlmDepth`), never the persisted file
        // depth: a resumed subagent file opened as a top-level session
        // still reports for its own pane (the file's depth serves the
        // roster and usage attribution, not the reporter decision).
        let spawned_as_child = payload
            .get("rlmDepth")
            .and_then(Value::as_u64)
            .is_some_and(|depth| depth > 0);
        let (active, session_ref) = {
            let core = self.core.lock().unwrap();
            (core.busy, Worker::herdr_session_ref(&core))
        };
        let reporter = match crate::herdr::HerdrConfig::from_env(&client_env) {
            Some(config) if !spawned_as_child => {
                // The fresh epoch: bumping the shared counter makes the
                // replaced reporter's task drop its queued boundary
                // events instead of flushing them over this session's
                // pane state.
                let generation = self
                    .herdr_generation
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                crate::herdr::HerdrReporter::start(
                    config,
                    session_ref.clone(),
                    generation,
                    std::sync::Arc::clone(&self.herdr_generation),
                )
            }
            None if !spawned_as_child && self.herdr.lock().unwrap().enabled() => {
                // An idempotent re-create that carries no pane identity
                // (e.g. a replay from a client outside a Herdr pane)
                // must not strip the binding an earlier create or an
                // attach installed — the session keeps reporting for its
                // pane (adopt-if-absent, never rebind to nothing).
                return;
            }
            // Not inside a Herdr pane (the no-op reporter), or a live
            // RLM child: subagents share the parent's pane, so their runs
            // must not flip it and their quits must not release it.
            _ => crate::herdr::HerdrReporter::default(),
        };
        reporter.session_started(active, session_ref);
        *self.herdr.lock().unwrap() = reporter;
    }
}

/// Each native child gets a fresh Studio binding; copied parent bindings never survive.
pub(super) fn bind_studio_mcp_servers(
    mut servers: Vec<pa_core::mcp::AcpMcpServerConfig>,
    binding: &Value,
) -> Vec<pa_core::mcp::AcpMcpServerConfig> {
    for server in &mut servers {
        if let pa_core::mcp::AcpMcpServerConfig::Stdio { cwd, env, .. } = server {
            env.retain(|key, _| !key.starts_with("PRIME_STUDIO_NATIVE_"));
            env.insert(
                "PRIME_STUDIO_NATIVE_SESSION".to_string(),
                binding.to_string(),
            );
            if let Some(child_cwd) = binding.get("cwd").and_then(Value::as_str) {
                *cwd = child_cwd.to_string();
            }
        }
    }
    servers
}

/// The identity a Studio descriptor of this session carries (`PRIME_STUDIO_NATIVE_SESSION`).
pub(crate) fn studio_session_identity(core: &super::SessionCore) -> Option<Value> {
    let store = core.store.as_ref()?;
    Some(json!({
        "activeSessionId": core.active_session_id,
        "sessionId": store.session_id(),
        "sessionFile": store.path,
        "cwd": core.cwd,
        "parentSessionId": core.parent_session_id,
        "parentActiveSessionId": core.parent_active_session_id,
    }))
}

/// A session replacement inside this worker (new, switch, import, fork) rotates the Studio
/// MCP binding onto the successor: fresh credential and generation, descriptor rebound to its
/// identity and cwd. Called under the core lock right after the store swap, the lock the
/// credential check reads the binding and the identity through.
pub(crate) fn rotate_studio_mcp_for_successor(
    manager: &std::sync::Mutex<pa_core::mcp::McpManager>,
    core: &super::SessionCore,
) {
    let Some(identity) = studio_session_identity(core) else {
        return;
    };
    let rotated = manager.lock().unwrap().rotate_studio_binding(|server| {
        bind_studio_mcp_servers(vec![server.clone()], &identity)
            .pop()
            .unwrap_or_else(|| server.clone())
    });
    if let Err(error) = rotated {
        eprintln!("pa-daemon: Studio MCP binding revoked on session replacement: {error:#}");
    }
}

pub(super) fn active_session_id_of(payload: &[u8]) -> String {
    serde_json::from_slice::<Value>(payload)
        .ok()
        .and_then(|value| {
            value
                .get("activeSessionId")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_default()
}

pub(super) fn worker_server_capabilities(agent_dir: &std::path::Path) -> Vec<String> {
    // The factory lane advertises only while its opt-in gate reads
    // enabled (`factory.enabled`, default off).
    crate::factory_activity::advertised_server_capabilities(agent_dir)
}

/// RLM depth fields of a create payload: `(depth, max_depth)`. Values must
/// fit a u32; anything else fails the create instead of silently truncating.
fn create_payload_rlm_depth(payload: &Value) -> Result<(Option<u32>, Option<u32>), String> {
    fn parse(payload: &Value, key: &str) -> Result<Option<u32>, String> {
        match payload.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .map(Some)
                .ok_or_else(|| format!("create {key} must be a non-negative integer")),
        }
    }
    let depth = parse(payload, "rlmDepth")?;
    let max_depth = parse(payload, "rlmMaxDepth")?;
    Ok((depth, max_depth))
}

/// Creation prefix for a daemon-hosted session file: fresh files record all three changes;
/// a reopened session records them only when no earlier entry did. Which fresh create
/// arm still owes its creation prefix (the prefix runs after the startup scope registers).
enum FreshPrefixPlan {
    /// Nothing pending: the continuing arm wrote its own prefix before
    /// the scope.
    None,
    /// The in-memory `noSession` create: the prefix stays in memory, no
    /// rewrite.
    InMemory,
    /// A file-backed fresh session: prefix, `active` state, the optional
    /// single-write name fold, then one rewrite.
    PathBacked { fold_name: bool },
}

fn append_creation_prefix(
    store: &mut SessionFile,
    engine: &dyn SessionEngine,
    agent_dir: &std::path::Path,
    cwd: &str,
    fresh: bool,
) {
    let has_thinking_entry = store.has_thinking_level();
    let has_service_tier_entry = store.has_service_tier();
    let thinking_level = engine
        .effective_thinking_level()
        .unwrap_or_else(|| "off".to_string());
    if fresh {
        if let Some((provider, model_id)) = engine.creation_model() {
            store.append_model_change(&provider, &model_id);
        }
        store.append_thinking_level_change(&thinking_level);
    } else if !has_thinking_entry {
        store.append_thinking_level_change(&thinking_level);
    }
    if fresh || !has_service_tier_entry {
        let settings = pa_core::settings::SettingsManager::create(cwd, agent_dir);
        let service_tier = settings.get_default_service_tier();
        store.append_entry(
            "service_tier_change",
            json!({ "serviceTier": service_tier }),
        );
    }
}

#[cfg(test)]
#[path = "create_collapse_tests.rs"]
mod create_collapse_tests;

#[cfg(test)]
mod studio_mcp_tests {
    use super::{bind_studio_mcp_servers, json};
    use pa_core::mcp::AcpMcpServerConfig;
    use std::collections::HashMap;

    #[test]
    fn inherited_studio_servers_rebind_for_each_native_generation() {
        let server = AcpMcpServerConfig::Stdio {
            name: "studio".to_string(),
            command: "node".to_string(),
            args: vec!["bridge.mjs".to_string()],
            cwd: "/parent".to_string(),
            env: HashMap::from([
                (
                    "PRIME_STUDIO_NATIVE_SESSION".to_string(),
                    "parent".to_string(),
                ),
                (
                    "PRIME_STUDIO_NATIVE_WRITER".to_string(),
                    "parent-writer".to_string(),
                ),
                ("RELAY_TOKEN".to_string(), "shared-relay".to_string()),
            ]),
        };
        let child = json!({
            "sessionId": "child", "activeSessionId": "child-active",
            "sessionFile": "/child.jsonl", "cwd": "/child", "parentSessionId": "parent"
        });
        let grandchild = json!({
            "sessionId": "grandchild", "activeSessionId": "grandchild-active",
            "sessionFile": "/grandchild.jsonl", "cwd": "/grandchild", "parentSessionId": "child"
        });
        let bound = bind_studio_mcp_servers(vec![server], &child);
        let expected = |binding: &serde_json::Value| {
            vec![AcpMcpServerConfig::Stdio {
                name: "studio".to_string(),
                command: "node".to_string(),
                args: vec!["bridge.mjs".to_string()],
                cwd: binding["cwd"].as_str().unwrap().to_string(),
                env: HashMap::from([
                    (
                        "PRIME_STUDIO_NATIVE_SESSION".to_string(),
                        binding.to_string(),
                    ),
                    ("RELAY_TOKEN".to_string(), "shared-relay".to_string()),
                ]),
            }]
        };
        assert_eq!(bound, expected(&child));
        assert_eq!(
            bind_studio_mcp_servers(bound, &grandchild),
            expected(&grandchild)
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn native_child_startup_holds_first_prompt_and_shutdown_keeps_native_identity() {
        use crate::worker::{Worker, WorkerConfig};
        use serde_json::Value;
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::sync::{mpsc, oneshot};

        let dir = tempfile::TempDir::new().unwrap();
        let socket = dir.path().join("sup.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        let (hooks_tx, mut hooks) = mpsc::unbounded_channel::<(Value, oneshot::Sender<()>)>();
        let server = tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let hooks_tx = hooks_tx.clone();
                tokio::spawn(async move {
                    let (reader, mut writer) = stream.into_split();
                    writer
                        .write_all(b"{\"type\":\"daemon_hello\"}\n")
                        .await
                        .unwrap();
                    let mut line = String::new();
                    BufReader::new(reader).read_line(&mut line).await.unwrap();
                    let request: Value = serde_json::from_str(&line).unwrap();
                    let command = &request["command"];
                    let kind = command["type"].as_str().unwrap();
                    if kind == "request_host_hook" {
                        let (answer, wait) = oneshot::channel();
                        hooks_tx.send((command.clone(), answer)).unwrap();
                        wait.await.unwrap();
                    }
                    let response =
                        crate::protocol::response_line(&crate::protocol::response_success(
                            request["id"].as_str(),
                            kind,
                            Some(json!({"result":{}})),
                        ));
                    let mut line = serde_json::to_vec(&response).unwrap();
                    line.push(b'\n');
                    writer.write_all(&line).await.unwrap();
                });
            }
        });
        let worker = Arc::new(Worker::new(
            WorkerConfig {
                socket_path: dir.path().join("worker.sock"),
                supervisor_socket_path: socket,
                token: "child-token".to_string(),
                worker_instance_id: String::new(),
                active_session_id: "child-active".to_string(),
                agent_dir: dir.path().join("agent"),
                recovery_journal_path: dir.path().join("recovery.jsonl"),
                telemetry_disabled: Some(true),
                script: Some(json!({"responses":["child reply"]})),
            },
            None,
        ));
        let created = tokio::time::timeout(Duration::from_secs(5), worker.dispatch("create", &json!({
            "cwd":dir.path(), "sessionDir":dir.path().join("sessions"), "rlmDepth":1,
            "runtimeMetadata":{"kind":"subagent", "rlmChildId":"kid", "parentSessionId":"parent", "parentActiveSessionId":"parent-active"},
            "hostHooksParent":{"activeSessionId":"parent-active", "hostContractVersion":1, "hooks":["session_start","session_shutdown"], "timeoutMs":150_000},
            "studioHostContractVersion":1, "studioMcpParentSessionId":"parent",
            "studioMcpServers":[{"type":"stdio", "name":"studio", "command":"node", "args":["bridge.mjs"], "cwd":"/parent", "env":{"PRIME_STUDIO_NATIVE_SESSION":"parent"}}],
        }))).await.expect("create must return while startup awaits its answer");
        assert!(created.success, "{created:?}");
        let session_id = created.data.unwrap()["sessionId"]
            .as_str()
            .unwrap()
            .to_string();
        let (startup, answer) = tokio::time::timeout(Duration::from_secs(5), hooks.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(startup["hook"], "session_start");
        assert_eq!(startup["origin"]["sessionId"], session_id);
        assert_eq!(startup["origin"]["activeSessionId"], "child-active");
        assert_eq!(startup["origin"]["parentSessionId"], "parent");
        let snapshot = worker.dispatch("session_snapshot", &json!({})).await;
        assert!(
            snapshot.success,
            "startup callback read must reenter: {snapshot:?}"
        );
        assert!(worker.input_pauses.paused());
        let servers = worker
            .session_mcp_manager()
            .lock()
            .unwrap()
            .get_acp_servers();
        let pa_core::mcp::AcpMcpServerConfig::Stdio { env, .. } = &servers[0] else {
            panic!("stdio binding");
        };
        let binding: Value = serde_json::from_str(&env["PRIME_STUDIO_NATIVE_SESSION"]).unwrap();
        assert_eq!(binding["sessionId"], session_id);
        let mut frames = worker.events.subscribe();
        let queued = worker
            .dispatch("prompt", &json!({"message":"first task"}))
            .await;
        assert!(queued.success, "{queued:?}");
        {
            let core = worker.core.lock().unwrap();
            assert!(!core.busy, "first prompt must remain parked during startup");
            assert_eq!(core.steering.len() + core.follow_up.len(), 1);
        }
        answer.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let frame = frames.recv().await.unwrap();
                let outbound: Value = serde_json::from_slice(&frame.payload).unwrap();
                if outbound["event"]["type"] == "agent_end" {
                    break;
                }
            }
        })
        .await
        .expect("first prompt must finish after startup settles");
        let closing = tokio::spawn({
            let worker = Arc::clone(&worker);
            async move { worker.dispatch("kill", &json!({})).await }
        });
        let (shutdown, answer) = tokio::time::timeout(Duration::from_secs(5), hooks.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(shutdown["hook"], "session_shutdown");
        assert_eq!(shutdown["origin"]["sessionId"], session_id);
        assert_eq!(shutdown["origin"]["rlmChildId"], "kid");
        assert!(
            worker.core.lock().unwrap().created,
            "shutdown retains readable identity"
        );
        answer.send(()).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(5), closing)
                .await
                .unwrap()
                .unwrap()
                .success
        );
        server.abort();
    }
}

#[cfg(test)]
mod harness_seed_tests {
    /// A `/refine --probe` child starts from its seed: `harnessSeed` names a
    /// harness directory whose state becomes the new session's local store.
    #[tokio::test]
    async fn a_harness_seed_becomes_the_new_sessions_local_store() {
        use pa_core::refinement::{
            empty_harness_state, get_local_harness_state_dir, load_harness_state,
            save_harness_state, HarnessEntry, HarnessScope, RefinementKind,
        };
        let dir = tempfile::TempDir::new().expect("temp dir");
        let sessions = dir.path().join("sessions");
        std::fs::create_dir_all(&sessions).expect("sessions dir");
        let mut seed = empty_harness_state();
        seed.entries
            .get_mut(&RefinementKind::Memory)
            .unwrap()
            .insert(
                "m1".to_string(),
                serde_json::from_value::<HarnessEntry>(serde_json::json!({
                    "id": "m1", "kind": "memory", "title": "Tests",
                    "content": "Run the tests before answering", "path": "",
                    "reference": {}, "arguments": {}, "metadata": {}, "source": "probe",
                    "version": 1, "created_at": "t", "updated_at": "t"
                }))
                .expect("entry"),
            );
        let seed_dir = dir.path().join("seed");
        save_harness_state(&seed_dir, &seed).expect("seed saved");
        let worker = crate::worker::Worker::new(
            crate::worker::WorkerConfig {
                socket_path: dir.path().join("worker.sock"),
                supervisor_socket_path: std::path::PathBuf::new(),
                token: "token".to_string(),
                worker_instance_id: String::new(),
                active_session_id: "probe-child".to_string(),
                agent_dir: dir.path().join("agent"),
                recovery_journal_path: dir.path().join("recovery.jsonl"),
                telemetry_disabled: Some(true),
                script: Some(serde_json::json!({ "responses": [] })),
            },
            None,
        );
        let created = worker
            .dispatch(
                "create",
                &serde_json::json!({
                    "cwd": "/tmp",
                    "sessionDir": sessions,
                    "harnessSeed": seed_dir,
                }),
            )
            .await;
        assert!(created.success, "{created:?}");
        let artifacts = {
            let core = worker.core.lock().unwrap();
            let store = core.store.as_ref().expect("persisted session");
            crate::scheduled_jobs::session_artifact_dir(&store.path, store.session_id())
        };
        // The seed is the session's own store, and the session is pinned to it: its
        // local scope resolves there, not to the project store its cwd belongs to.
        let own = get_local_harness_state_dir(artifacts.as_deref()).expect("own harness dir");
        assert_eq!(
            pa_core::refinement::project::resolve_local_harness_state_dir(
                &dir.path().join("agent"),
                std::path::Path::new("/tmp"),
                artifacts.as_deref(),
            ),
            Some(own.clone())
        );
        let local = load_harness_state(&own, HarnessScope::Local);
        assert!(local.entries[&RefinementKind::Memory].contains_key("m1"));
    }
}
