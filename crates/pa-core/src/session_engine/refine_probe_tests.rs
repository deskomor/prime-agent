//! `/refine --probe` end to end over a scripted session: a fake child runner
//! stands in for the daemon's RLM children.

use super::refine_probe::{ProbeChildOutput, ProbeChildSpec, ProbeRequest, ProbeRunnerFn};
use super::*;
use pa_agent::agent::{AgentInitialState, AgentOptions};
use pa_agent::scripted::ScriptedProvider;

fn loop_model() -> pa_agent::types::Model {
    serde_json::from_value(serde_json::json!({
        "id": "m", "name": "m", "api": "openai-completions", "provider": "test",
        "baseUrl": "http://localhost", "reasoning": false, "input": ["text"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": 1000, "maxTokens": 100
    }))
    .unwrap()
}

fn ai_model() -> pa_types::ai::Model {
    serde_json::from_value(serde_json::json!({
        "id": "m", "name": "m", "api": "openai-completions", "provider": "test",
        "baseUrl": "http://localhost", "reasoning": false, "input": ["text"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": 100_000, "maxTokens": 8000
    }))
    .unwrap()
}

fn reply(text: String) -> pa_types::ai::AssistantMessage {
    pa_types::ai::AssistantMessage {
        content: vec![pa_types::ai::AssistantContentBlock::Text(
            pa_types::ai::TextContent {
                text,
                text_signature: None,
                rest: serde_json::Map::default(),
            },
        )],
        api: "openai-completions".to_string(),
        provider: "test".to_string(),
        model: "m".to_string(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: pa_types::ai::Usage::default(),
        stop_reason: pa_types::ai::StopReason::Stop,
        stop_reason_raw: None,
        error_message: None,
        timestamp: 0,
        rest: serde_json::Map::default(),
    }
}

/// A refiner seam answering `text`, recording the user prompt it was sent.
fn answering(
    text: &'static str,
    prompt: Arc<std::sync::Mutex<String>>,
) -> crate::refinement::executor::RefinerFn {
    Box::new(move |_model, _system, user_prompt| {
        *prompt.lock().unwrap() = user_prompt;
        Box::pin(async move { Ok(reply(text.to_string())) })
    })
}

fn git(dir: &std::path::Path, args: &[&str]) {
    let status = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .status()
        .unwrap();
    assert!(status.success(), "git {args:?}");
}

/// A persisted session whose cwd is `cwd`.
async fn session_in(cwd: &std::path::Path, root: &std::path::Path) -> AgentSession {
    let provider = Arc::new(ScriptedProvider::new(loop_model()));
    let agent = Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            model: Some(loop_model()),
            ..Default::default()
        },
        stream_fn: Some(provider.stream_fn()),
        ..Default::default()
    });
    let session_dir = root.join("session");
    std::fs::create_dir_all(&session_dir).unwrap();
    let mut manager = SessionManager::in_memory(cwd);
    manager.materialize_session_file(Some(session_dir));
    AgentSession::new(Arc::new(agent), manager, vec![])
        .await
        .unwrap()
}

const TWO_EDIT_PLAN: &str = r#"{"summary":"two notes","rationale":"r","expectedOutcome":"e","edits":[
  {"action":"create","kind":"memory","id":"m1","title":"Style","content":"Prefer tabs","reason":"turn 1"},
  {"action":"create","kind":"memory","id":"m2","title":"Tests","content":"Run the tests before answering","reason":"turn 2"}]}"#;
const KEEP_SECOND: &str = r#"{"keep":[1],"notes":["kept -> edit 1 -> the candidate ran the tests","dropped -> edit 0 -> never exercised"],"verdicts":[{"editIndex":1,"verdict":"helped","evidence":"candidate turn 2 ran the tests"},{"editIndex":0,"verdict":"no_effect","evidence":"no style choice came up"}]}"#;

#[tokio::test]
async fn a_probe_keeps_the_edit_the_candidate_run_supports_and_removes_its_worktrees() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = tmp.path().join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.email", "test@example.com"]);
    git(&repo, &["config", "user.name", "Test"]);
    std::fs::write(repo.join("README.md"), "hello\n").unwrap();
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-q", "-m", "init"]);
    let session = session_in(&repo, tmp.path()).await;

    let specs: Arc<std::sync::Mutex<Vec<(ProbeChildSpec, usize)>>> = Arc::default();
    let seen = Arc::clone(&specs);
    let runner: ProbeRunnerFn = Arc::new(move |spec: ProbeChildSpec| {
        // The seed only lives while the probe runs: read it now.
        let seeded = crate::refinement::load_harness_state(
            &spec.harness_seed,
            crate::refinement::HarnessScope::Local,
        );
        let memories = seeded.entries[&crate::refinement::RefinementKind::Memory].len();
        seen.lock().unwrap().push((spec.clone(), memories));
        Box::pin(async move {
            if spec.label == "candidate" {
                std::fs::write(spec.cwd.join("tests_ran.txt"), "ok\n").unwrap();
            }
            Ok(ProbeChildOutput {
                messages: Vec::new(),
                answer: Some(format!("{} answer", spec.label)),
                error: None,
            })
        })
    });
    let compose_prompt = Arc::new(std::sync::Mutex::new(String::new()));
    let outcome = session
        .refine_probed_with(
            &refine::RefineOptions::default(),
            &ProbeRequest {
                task: Some("fix the build".to_string()),
            },
            &ai_model(),
            answering(TWO_EDIT_PLAN, Arc::default()),
            answering(KEEP_SECOND, Arc::clone(&compose_prompt)),
            runner,
            tmp.path().join("harness"),
        )
        .await
        .unwrap();

    // Each run started from its own seed: only the candidate's carries the proposal.
    let specs = specs.lock().unwrap().clone();
    assert_eq!(specs.len(), 2);
    for (spec, memories) in &specs {
        assert_eq!(*memories, if spec.label == "candidate" { 2 } else { 0 });
        assert_eq!(spec.task, "fix the build");
        assert_eq!(spec.model, "test/m");
    }
    // The candidate's worktree change reached the compose call as evidence.
    assert!(compose_prompt.lock().unwrap().contains("tests_ran.txt"));

    // Only the kept edit applied, as one ordinary refinement.
    let applied: Vec<&str> = outcome
        .result
        .applied_edits
        .iter()
        .filter(|edit| edit.applied)
        .map(|edit| edit.id.as_str())
        .collect();
    assert_eq!(applied, vec!["m2"]);
    assert!(outcome
        .notice
        .starts_with("probe: kept [1], dropped [0] of 2 proposed edits"));

    // The per-edit verdicts are recorded on the applied refinement.
    let verdicts = session.session_handle().lock().await.refinement_verdicts();
    assert_eq!(verdicts.len(), 2);
    assert!(verdicts
        .iter()
        .all(|verdict| verdict.refinement_id == outcome.result.id));

    // The worktrees and the scratch directory are gone.
    let worktrees = std::process::Command::new("git")
        .arg("-C")
        .arg(&repo)
        .args(["worktree", "list"])
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&worktrees.stdout).lines().count(),
        1
    );
    assert!(!specs[0].0.harness_seed.parent().unwrap().exists());
}

#[tokio::test]
async fn a_probe_outside_a_git_repository_refuses_before_any_model_call() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().join("plain");
    std::fs::create_dir_all(&cwd).unwrap();
    let session = session_in(&cwd, tmp.path()).await;
    let never: crate::refinement::executor::RefinerFn =
        Box::new(|_model, _system, _prompt| panic!("no model call before the git check"));
    let never_compose: crate::refinement::executor::RefinerFn =
        Box::new(|_model, _system, _prompt| panic!("no compose call"));
    let runner: ProbeRunnerFn = Arc::new(|_spec| panic!("no probe run"));
    let error = session
        .refine_probed_with(
            &refine::RefineOptions::default(),
            &ProbeRequest {
                task: Some("fix the build".to_string()),
            },
            &ai_model(),
            never,
            never_compose,
            runner,
            tmp.path().join("harness"),
        )
        .await
        .err()
        .expect("refused");
    assert!(
        error.to_string().contains("needs a git repository"),
        "{error:#}"
    );
}
