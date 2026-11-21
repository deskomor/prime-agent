//! Required, turn-local Studio Plan contract. Native dispatch never executes the host's commands.
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use pa_agent::{
    abort::AbortSignal,
    agent_loop::RestrictedRun,
    types::{AgentTool, AgentToolResult, AgentToolUpdateCallback, ToolCall},
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::host_hooks::HostHookBroker;

pub(crate) const FEATURE: &str = "required_plan_mode";
pub(crate) const EXPLORE_FEATURE: &str = "required_plan_explore";
const EXPLORE: &str = "studio_plan_explore";
const MAX_TASK: usize = 4000;
pub(crate) const MAX_REPORT: usize = 12000;
pub(crate) const EXPLORE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

pub(crate) fn explore_opted_in(contract: &Value) -> bool {
    opted_in(contract)
        && contract["features"]
            .as_array()
            .is_some_and(|features| features.iter().any(|feature| feature == EXPLORE_FEATURE))
}

pub(crate) fn valid_task(task: &str) -> bool {
    !task.trim().is_empty() && task.len() <= MAX_TASK
}

pub(crate) type Explore = Arc<
    dyn Fn(String, AbortSignal) -> pa_agent::BoxFut<'static, anyhow::Result<Value>> + Send + Sync,
>;
pub(crate) type ReadHook = Arc<
    dyn Fn(String, Value, AbortSignal) -> pa_agent::BoxFut<'static, anyhow::Result<Value>>
        + Send
        + Sync,
>;
pub(crate) const PROMPT: &str = "plan_prompt";
pub(crate) const VETO: &str = "plan_tool_veto";
pub(crate) const TOOL: &str = "plan_tool";
pub(crate) const QUESTION: &str = "plan_question";
pub(crate) const EXPLORER: &str = "plan_explorer";
pub(crate) fn configured_hook(hook: &str) -> &str {
    if matches!(hook, QUESTION | EXPLORER) {
        TOOL
    } else {
        hook
    }
}
pub(crate) fn required_hook(hook: &str) -> bool {
    HOOKS.contains(&hook) || matches!(hook, QUESTION | EXPLORER)
}
pub(crate) const HOOKS: [&str; 3] = [PROMPT, VETO, TOOL];
const READ: &str = "studio_plan_read";
const PROPOSE: &str = "studio_plan_propose";
const ASK: &str = "studio_plan_ask";

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Activation {
    version: u8,
    pub(crate) turn_id: String,
    #[serde(default)]
    ask_enabled: bool,
    #[serde(default)]
    pub(crate) explore_enabled: bool,
}

impl Activation {
    pub(crate) fn parse(value: &Value) -> anyhow::Result<Self> {
        let activation: Self = serde_json::from_value(value.clone())?;
        anyhow::ensure!(
            activation.version == 1
                && !activation.turn_id.trim().is_empty()
                && activation.turn_id.len() <= 256,
            "Invalid Plan mode version or turnId"
        );
        Ok(activation)
    }
}

pub(crate) fn opted_in(contract: &Value) -> bool {
    contract["version"].as_u64() == Some(1)
        && contract["features"]
            .as_array()
            .is_some_and(|features| features.iter().any(|feature| feature == FEATURE))
        && contract["hooks"].as_array().is_some_and(|hooks| {
            HOOKS
                .iter()
                .all(|name| hooks.iter().any(|hook| hook == *name))
        })
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
pub(crate) struct Scope {
    pub(crate) activation: Activation,
    pub(crate) actor: String,
    pub(crate) admission: String,
    pub(crate) owner: String,
}

impl Scope {
    pub(crate) fn wire_scope(&self, actor: &str) -> Value {
        json!({"turnId": self.activation.turn_id, "admissionId": self.admission,
            "rootActiveSessionId": self.actor, "actorActiveSessionId": actor})
    }

    pub(crate) async fn child_read_hook(
        &self,
        broker: &HostHookBroker,
        actor: &str,
        hook: &str,
        mut payload: Value,
    ) -> anyhow::Result<Value> {
        anyhow::ensure!(
            (matches!(hook, VETO | TOOL) && payload["toolName"] == READ)
                || (hook == EXPLORER
                    && matches!(payload["action"].as_str(), Some("admitted" | "finished"))),
            "Only read hooks may be relayed for Plan explorers"
        );
        let scope = self.wire_scope(actor);
        payload["scope"] = scope.clone();
        let answer = broker
            .call_child_plan_read(hook, payload, &self.owner, actor)
            .await
            .map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            answer.get("scope") == Some(&scope),
            "Plan explorer hook returned the wrong scope"
        );
        Ok(answer)
    }

    async fn required(
        &self,
        broker: &HostHookBroker,
        hook: &str,
        mut payload: Value,
        signal: Option<AbortSignal>,
    ) -> anyhow::Result<Value> {
        let scope = self.wire_scope(&self.actor);
        payload["scope"] = scope.clone();
        let answer = broker
            .call_plan(hook, payload, signal, &self.owner)
            .await
            .map_err(|reason| anyhow::anyhow!("Plan mode {hook} failed: {reason}"))?;
        anyhow::ensure!(
            answer.get("scope") == Some(&scope),
            "Plan mode {hook} returned the wrong turn or actor"
        );
        Ok(answer)
    }

    pub(crate) async fn start(
        self,
        broker: Arc<HostHookBroker>,
        signal: AbortSignal,
        explore: Option<Explore>,
    ) -> anyhow::Result<RestrictedRun> {
        let answer = self
            .required(&broker, PROMPT, json!({}), Some(signal))
            .await?;
        let append_system_prompt = text(&answer, "appendSystemPrompt")
            .ok_or_else(|| {
                anyhow::anyhow!("Plan mode host did not supply the required system prompt")
            })?
            .to_owned();
        let mut tools = inventory(self.activation.ask_enabled);
        if self.activation.explore_enabled && explore.is_some() {
            tools.push(Arc::new(Definition(
                EXPLORE,
                "Ask a read-only explorer to inspect one question and report findings.",
                json!({"type":"object","additionalProperties":false,"required":["task"],
                    "properties":{"task":{"type":"string","minLength":1,"maxLength":MAX_TASK}}}),
            )));
        }
        let definitions = tools.clone();
        let stopped = Arc::new(AtomicBool::new(false));
        let accepted = Arc::clone(&stopped);
        let dispatch = Arc::new(
            move |call: ToolCall,
                  signal: AbortSignal|
                  -> pa_agent::BoxFut<'static, AgentToolResult> {
                let scope = self.clone();
                let broker = Arc::clone(&broker);
                let stopped = Arc::clone(&stopped);
                let tools = definitions.clone();
                let explore = explore.clone();
                Box::pin(async move {
                    if stopped.load(Ordering::Acquire) {
                        return AgentToolResult::error(
                            "Plan mode: a proposal was already accepted; no further calls are allowed",
                        );
                    }
                    let result = async {
                    let payload = json!({"toolName":call.name,"toolCallId":call.id,"input":call.arguments});
                    let answer = scope.required(&broker, VETO, payload.clone(), Some(signal.clone())).await?;
                    match answer["allow"].as_bool() {
                        Some(true) => {}
                        Some(false) => anyhow::bail!("Plan mode denied {}: {}", call.name,
                            text(&answer, "reason").unwrap_or("host did not provide a reason")),
                        None => anyhow::bail!("Plan mode veto returned no boolean allow decision"),
                    }
                    let tool = tools.iter().find(|tool| tool.name() == call.name)
                        .ok_or_else(|| anyhow::anyhow!("Plan mode does not permit {}", call.name))?;
                    let validated = pa_agent::validation::validate_tool_arguments(tool.name(), tool.parameters(), &call.arguments)
                        .map_err(anyhow::Error::msg)?;
                    anyhow::ensure!(validated == call.arguments, "Plan mode tool arguments must match the schema without coercion");
                    if call.name == EXPLORE {
                        let task = text(&call.arguments, "task").filter(|task| valid_task(task))
                            .ok_or_else(|| anyhow::anyhow!("Plan exploration requires a non-empty bounded task"))?;
                        let explore = explore.ok_or_else(|| anyhow::anyhow!("Plan exploration was not enabled"))?;
                        let value = explore(task.to_string(), signal).await?;
                        return Ok(AgentToolResult { details: value.clone(), ..AgentToolResult::text(value.to_string()) });
                    }
                    let answer = scope.required(&broker, if call.name == ASK { QUESTION } else { TOOL }, payload, Some(signal)).await?;
                    anyhow::ensure!(answer["ok"].as_bool() == Some(true), "Plan mode tool failed: {}",
                        text(&answer, "reason").unwrap_or("host returned no successful tool result"));
                    let value = answer.get("result").filter(|v| !v.is_null())
                        .ok_or_else(|| anyhow::anyhow!("Plan mode host returned no tool result"))?;
                    if call.name == PROPOSE {
                        let proposal = &value["proposal"];
                        let checked = pa_agent::validation::validate_tool_arguments(PROPOSE, tool.parameters(), proposal)
                            .map_err(anyhow::Error::msg)?;
                        anyhow::ensure!(*proposal == checked && value["accepted"].as_bool() == Some(true)
                            && text(value, "proposalId").is_some(), "Plan mode host returned an invalid proposal acceptance");
                        stopped.store(true, Ordering::Release);
                    }
                    Ok::<_, anyhow::Error>(AgentToolResult {
                        details: value.clone(),
                        terminate: (call.name == PROPOSE).then_some(true),
                        ..AgentToolResult::text(value.to_string())
                    })
                }.await;
                    result.unwrap_or_else(|error| AgentToolResult::error(error.to_string()))
                })
            },
        );
        Ok(RestrictedRun {
            tools,
            append_system_prompt,
            dispatch,
            stopped: Arc::new(move || accepted.load(Ordering::Acquire)),
        })
    }
}

pub(crate) async fn lock_topology(
    supervisor: &crate::supervisor::Supervisor,
    parent: &str,
) -> tokio::sync::OwnedMutexGuard<()> {
    let lock = {
        let mut lanes = supervisor.plan_topology.lock().unwrap();
        lanes.retain(|_, lane| lane.strong_count() != 0);
        if let Some(lock) = lanes.get(parent).and_then(std::sync::Weak::upgrade) {
            lock
        } else {
            let lock = Arc::new(tokio::sync::Mutex::new(()));
            lanes.insert(parent.to_string(), Arc::downgrade(&lock));
            lock
        }
    };
    lock.lock_owned().await
}

pub(crate) struct FamilyGuard {
    roots: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    root: String,
}

impl FamilyGuard {
    pub(crate) async fn acquire(
        supervisor: &Arc<crate::supervisor::Supervisor>,
        resident: &Arc<crate::registry::ResidentWorker>,
    ) -> anyhow::Result<Self> {
        let (root, file, dir) = {
            let descriptor = resident.descriptor.lock().await;
            (
                descriptor.root_active_session_id.clone(),
                descriptor.session_file.clone(),
                descriptor.session_dir.clone(),
            )
        };
        let _topology = lock_topology(supervisor, &root).await;
        anyhow::ensure!(
            supervisor.resident_children_of(resident).await.is_empty(),
            "Plan mode cannot start with native child workers"
        );
        if let Some(file) = file {
            let ledger = supervisor.rlm_spawn_ledger_for(dir.as_deref()).await?;
            anyhow::ensure!(
                ledger
                    .live_edges_of_parent(std::path::Path::new(&file))?
                    .is_empty(),
                "Plan mode cannot start with wakeable native children"
            );
        }
        let roots = Arc::clone(&supervisor.plan_roots);
        anyhow::ensure!(
            roots.lock().unwrap().insert(root.clone()),
            "A Plan mode turn already owns this root"
        );
        Ok(Self { roots, root })
    }
}

impl Drop for FamilyGuard {
    fn drop(&mut self) {
        self.roots.lock().unwrap().remove(&self.root);
    }
}

fn text<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|s| !s.trim().is_empty())
}

struct Definition(&'static str, &'static str, Value);
impl AgentTool for Definition {
    fn name(&self) -> &str {
        self.0
    }
    fn description(&self) -> &str {
        self.1
    }
    fn parameters(&self) -> &Value {
        &self.2
    }
    fn execute(
        self: Arc<Self>,
        _: String,
        _: Value,
        _: AbortSignal,
        _: AgentToolUpdateCallback,
    ) -> pa_agent::BoxFut<'static, anyhow::Result<AgentToolResult>> {
        Box::pin(async { anyhow::bail!("Plan definitions require the bound host dispatch") })
    }
}

pub(crate) fn native_explore(
    link: Arc<crate::supervisor_link::SupervisorLink>,
    token: String,
    admission: String,
    parent: AbortSignal,
) -> Explore {
    Arc::new(move |task, tool| {
        let link = Arc::clone(&link);
        let parent = parent.clone();
        let token = token.clone();
        let admission = admission.clone();
        Box::pin(async move {
            let handle = tokio::spawn(async move {
                pa_agent::abort::throw_if_aborted(&parent)?;
                pa_agent::abort::throw_if_aborted(&tool)?;
                let start = json!({"type":"worker_plan_explore","workerToken":token,"request":{"op":"start","admission":admission,"task":task}});
                // Keep the stop watcher alive even if the model tool future is dropped.
                let answer = tokio::select! {
                    biased;
                    () = async { tokio::select! { () = parent.aborted() => {}, () = tool.aborted() => {} } } => {
                        Err(pa_agent::abort::aborted_error())
                    }
                    answer = link.request(start, EXPLORE_TIMEOUT) => answer,
                };
                match answer {
                    Ok(reply) if reply.success => Ok(reply.data.unwrap_or_default()),
                    Ok(reply) => Err(anyhow::anyhow!(reply
                        .error
                        .unwrap_or_else(|| "Plan exploration failed".into()))),
                    Err(error) => {
                        // A timeout or transport error does not tell us whether startup landed.
                        let stop = json!({"type":"worker_plan_explore","workerToken":token,"request":{"op":"stop","admission":admission}});
                        if let Err(stop_error) = link
                            .request_success(stop, std::time::Duration::from_secs(5))
                            .await
                        {
                            eprintln!("pa-daemon: Plan explorer cancellation could not be confirmed: {stop_error}");
                        }
                        Err(error)
                    }
                }
            });
            handle.await?
        })
    })
}

pub(crate) fn native_read(
    link: Arc<crate::supervisor_link::SupervisorLink>,
    token: String,
) -> ReadHook {
    Arc::new(move |hook, payload, signal| {
        let link = Arc::clone(&link);
        let request = json!({"type":"worker_plan_explore","workerToken":token,"request":{"op":"read","hook":hook,"payload":payload}});
        Box::pin(async move {
            pa_agent::abort::race_with_abort(
                link.request_success(request, std::time::Duration::from_secs(25)),
                &signal,
            )
            .await?
        })
    })
}

pub(crate) fn child_run(read: ReadHook) -> RestrictedRun {
    let mut tools = inventory(/*ask*/ false);
    tools.truncate(1);
    let tool = tools[0].clone();
    RestrictedRun {
        tools,
        append_system_prompt: "You are a read-only Plan explorer. Use studio_plan_read if needed. Return concise findings and file references to the parent; do not implement or propose a plan.".into(),
        stopped: Arc::new(|| false),
        dispatch: Arc::new(move |call, signal| {
            let read = Arc::clone(&read);
            let tool = Arc::clone(&tool);
            Box::pin(async move {
                let result = async {
                    anyhow::ensure!(call.name == READ, "Plan explorer does not permit {}", call.name);
                    let input = pa_agent::validation::validate_tool_arguments(READ, tool.parameters(), &call.arguments)
                        .map_err(anyhow::Error::msg)?;
                    anyhow::ensure!(input == call.arguments, "Plan read arguments must match without coercion");
                    let payload = json!({"toolName":READ,"toolCallId":call.id,"input":input});
                    let answer = read(VETO.into(), payload.clone(), signal.clone()).await?;
                    anyhow::ensure!(answer["allow"] == true, "Plan explorer read denied by host");
                    let answer = read(TOOL.into(), payload, signal).await?;
                    anyhow::ensure!(answer["ok"] == true && !answer["result"].is_null(), "Plan explorer host read failed");
                    Ok::<_, anyhow::Error>(AgentToolResult::text(answer["result"].to_string()))
                }.await;
                result.unwrap_or_else(|error| AgentToolResult::error(error.to_string()))
            })
        }),
    }
}

fn inventory(ask: bool) -> Vec<Arc<dyn AgentTool>> {
    let step = json!({"type":"object","additionalProperties":false,"required":["text"],"properties":{
        "text":{"type":"string","minLength":1,"maxLength":1000},"note":{"type":"string","maxLength":12000}}});
    let mut tools: Vec<Arc<dyn AgentTool>> = vec![
        Arc::new(Definition(
            READ,
            "Inspect the project using a supported read-only command and literal arguments.",
            json!({
            "type":"object","additionalProperties":false,"required":["command","args"],"properties":{
            "command":{"type":"string","minLength":1},"args":{"type":"array","maxItems":40,"items":{"type":"string","minLength":1,"maxLength":2000}}}}),
        )),
        Arc::new(Definition(
            PROPOSE,
            "Submit a plan for the user's approval. Acceptance ends this turn.",
            json!({
            "type":"object","additionalProperties":false,"required":["title","markdown","steps"],"properties":{
            "title":{"type":"string","minLength":1,"maxLength":300},"markdown":{"type":"string","minLength":1,"maxLength":12000},
            "steps":{"type":"array","minItems":1,"maxItems":60,"items":step}}}),
        )),
    ];
    if ask {
        let option = json!({"type":"object","additionalProperties":false,"required":["label","description"],"properties":{
            "label":{"type":"string","minLength":1,"maxLength":60},"description":{"type":"string"},"preview":{"type":"string"}}});
        let question = json!({"type":"object","additionalProperties":false,"required":["question","header","options"],"properties":{
            "question":{"type":"string","minLength":1},"header":{"type":"string","maxLength":16},
            "options":{"type":"array","minItems":2,"maxItems":4,"items":option},"multiSelect":{"type":"boolean"}}});
        tools.push(Arc::new(Definition(ASK, "Ask the user one to four multiple-choice questions. A free-text answer is always offered.", json!({
            "type":"object","additionalProperties":false,"required":["questions"],"properties":{
            "questions":{"type":"array","minItems":1,"maxItems":4,"items":question}}}))));
    }
    tools
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::mpsc;

    const OWNER: &str = "supervisor:connection:root:lease";

    async fn start() -> (
        RestrictedRun,
        Arc<HostHookBroker>,
        mpsc::UnboundedReceiver<Value>,
    ) {
        let (send, mut events) = mpsc::unbounded_channel();
        let broker = HostHookBroker::new(Arc::new(move |event| {
            let _ = send.send(event);
        }));
        broker.set_owner_token(Some(OWNER.to_string()));
        broker
            .set_hooks(
                &HOOKS.map(str::to_string),
                None,
                Some(Duration::from_secs(2)),
            )
            .unwrap();
        let scope = Scope {
            activation: Activation::parse(
                &json!({"version":1,"turnId":"host-turn","askEnabled":true}),
            )
            .unwrap(),
            actor: "root".into(),
            admission: "admission".into(),
            owner: OWNER.into(),
        };
        let task = tokio::spawn(scope.start(Arc::clone(&broker), AbortSignal::default(), None));
        let prompt = events.recv().await.unwrap();
        assert_eq!(prompt["hook"], PROMPT);
        let request = prompt["requestId"].as_str().unwrap();
        let result =
            json!({"scope":prompt["payload"]["scope"],"appendSystemPrompt":"Read-only plan"});
        assert!(!broker.resolve_scoped(request, Ok(result.clone()), Some("foreign-owner")));
        assert!(broker.resolve_scoped(request, Ok(result), Some(OWNER)));
        let run = task.await.unwrap().unwrap();
        assert_eq!(run.append_system_prompt, "Read-only plan");
        assert_eq!(events.recv().await.unwrap()["type"], "host_hook_resolved");
        (run, broker, events)
    }

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            name: name.into(),
            id: "call".into(),
            arguments,
            thought_signature: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn plan_mode_veto_wrong_actor_and_latch_host_proposal() {
        let (run, broker, mut events) = start().await;
        let first = tokio::spawn((run.dispatch)(
            call("bash", json!({"command":"ignored"})),
            AbortSignal::default(),
        ));
        let veto = events.recv().await.unwrap();
        assert_eq!(veto["hook"], VETO);
        let mut scope = veto["payload"]["scope"].clone();
        scope["actorActiveSessionId"] = json!("another-actor");
        assert!(broker.resolve_scoped(
            veto["requestId"].as_str().unwrap(),
            Ok(json!({"scope":scope,"allow":true})),
            Some(OWNER)
        ));
        let denied = first.await.unwrap();
        assert_eq!(
            denied,
            AgentToolResult::error("Plan mode plan_tool_veto returned the wrong turn or actor")
        );
        assert_eq!(events.recv().await.unwrap()["type"], "host_hook_resolved");
        assert!(events.try_recv().is_err());

        let proposal = json!({"title":"Title","markdown":"Plan","steps":[{"text":"Do work"}]});
        let task = tokio::spawn((run.dispatch)(
            call(PROPOSE, proposal.clone()),
            AbortSignal::default(),
        ));
        for hook in [VETO, TOOL] {
            let frame = events.recv().await.unwrap();
            assert_eq!(frame["hook"], hook);
            let mut response = if hook == VETO {
                json!({"allow":true})
            } else {
                json!({"ok":true,"result":{"accepted":true,"proposalId":"p1","proposal":proposal}})
            };
            response["scope"] = frame["payload"]["scope"].clone();
            assert!(broker.resolve_scoped(
                frame["requestId"].as_str().unwrap(),
                Ok(response),
                Some(OWNER)
            ));
            assert_eq!(events.recv().await.unwrap()["type"], "host_hook_resolved");
        }
        let result = task.await.unwrap();
        assert_eq!(
            result.details,
            json!({"accepted":true,"proposalId":"p1","proposal":proposal})
        );
        assert_eq!(result.terminate, Some(true));
        assert!((run.stopped)());
        assert!(
            (run.dispatch)(
                call(READ, json!({"command":"ls","args":[]})),
                AbortSignal::default()
            )
            .await
            .is_error
        );
        assert!(events.try_recv().is_err());
    }

    #[tokio::test]
    async fn plan_explorer_dispatch_allows_reads_but_never_writes_proposes_or_redelegates() {
        let (send, mut seen) = mpsc::unbounded_channel();
        let run = child_run(Arc::new(move |hook, payload, _| {
            send.send((hook.clone(), payload)).unwrap();
            Box::pin(async move {
                Ok(if hook == VETO {
                    json!({"allow":true})
                } else {
                    json!({"ok":true,"result":"found"})
                })
            })
        }));
        for name in ["bash", "edit", "python_repl", PROPOSE, EXPLORE, ASK] {
            let result = (run.dispatch)(call(name, json!({})), AbortSignal::default()).await;
            assert!(result.is_error, "unexpected success for {name}");
        }
        assert!(seen.try_recv().is_err());
        let result = (run.dispatch)(
            call(READ, json!({"command":"ls","args":[]})),
            AbortSignal::default(),
        )
        .await;
        assert!(!result.is_error);
        for hook in [VETO, TOOL] {
            assert_eq!(
                seen.try_recv().unwrap(),
                (
                    hook.to_string(),
                    json!({
                        "toolName": READ, "toolCallId":"call", "input":{"command":"ls","args":[]}
                    })
                )
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn plan_mode_question_outlives_fast_hook_and_cancels() {
        let (run, broker, mut events) = start().await;
        let input = json!({"questions":[{"question":"Which?","header":"Choose","options":[{"label":"A","description":"A"},{"label":"B","description":"B"}]}]});
        for cancel in [false, true] {
            let controller = pa_agent::abort::AbortController::new();
            let task = tokio::spawn((run.dispatch)(
                call(ASK, input.clone()),
                controller.signal(),
            ));
            let veto = events.recv().await.unwrap();
            assert_eq!(veto["hook"], VETO);
            assert!(broker.resolve_scoped(
                veto["requestId"].as_str().unwrap(),
                Ok(json!({"scope":veto["payload"]["scope"],"allow":true})),
                Some(OWNER)
            ));
            assert_eq!(events.recv().await.unwrap()["type"], "host_hook_resolved");
            let question = events.recv().await.unwrap();
            assert_eq!(question["hook"], QUESTION);
            tokio::time::advance(Duration::from_secs(91)).await;
            let id = question["requestId"].as_str().unwrap();
            if cancel {
                controller.abort();
            } else {
                assert!(broker.resolve_scoped(id, Ok(json!({"scope":question["payload"]["scope"],"ok":true,"result":{"answers":["A"]}})), Some(OWNER)));
            }
            assert_eq!(task.await.unwrap().is_error, cancel);
            assert_eq!(
                events.recv().await.unwrap()["outcome"],
                if cancel { "failed" } else { "answered" }
            );
            assert!(!broker.resolve_scoped(id, Ok(Value::Null), Some(OWNER)));
        }
    }
}
