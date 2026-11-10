//! `prime-agent harness`: the global harness as a file tree, for an offline
//! harness optimiser such as Harness² (fork-only).
//!
//! Sub-verbs:
//!   - `export --out <dir> [--add <project>:<kind>:<id>]... [--omit
//!     <kind>:<id>]...`: write `<dir>/tree` (prompt and memory entries as
//!     marked markdown sections, folder skills), `<dir>/export.json` (what
//!     was exported) and `<dir>/verdicts.json` (what `/refine review` and
//!     probes have recorded, for the optimiser to read). `--add` and `--omit`
//!     write the tree as it would be with a project entry promoted or a
//!     global entry gone, so an optimiser can test that change first.
//!   - `plan --export <dir> --seed <tree> --tree <tree> --reason <text>
//!     --out <file>`: diff the tree an optimiser ended with against the seed
//!     it started from and write the difference as a refinement plan.
//!   - `import --plan <file> [--dry-run]`: apply a plan as ONE global
//!     refinement, through the same validation and history as `/refine`, so
//!     `/refine rollback <id>` undoes its entries. A plan may also name
//!     project entries to promote to global or delete (`projects`) and
//!     verdicts to record on earlier refinements (`verdicts`).
//!   - `stage --plan <file> --cycle <name>`, `pending`, `resolve --cycle
//!     <name> --as <applied|dismissed>`: a plan waiting for the user's
//!     approval lives in `<global store>/pending/<name>.json`; `pending`
//!     lists them and `resolve` files one away once it was imported or
//!     turned down.
//!   - `path [--cwd <dir>]`: print the global store and the project store of
//!     a working directory.
//!
//! Every verb prints one JSON object on stdout.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use pa_core::refinement::executor::{
    always_on_chars, apply_refinement_plan, generate_refinement_id, RefineOptions, RefinementPlan,
    DEFAULT_ALWAYS_ON_SOFT_LIMIT,
};
use pa_core::refinement::planner::{RefinementEdit, RefinementProposal};
use pa_core::refinement::verdicts::{
    append_global_verdicts, latest_verdicts, load_global_verdicts, RefinementVerdict, Verdict,
};
use pa_core::refinement::{
    append_global_refinement, get_global_harness_state_dir, load_global_refinement_history,
    load_harness_state, load_refinement_history_log, tree, update_harness_state, HarnessScope,
    RefinementAction, RefinementKind,
};
use serde_json::{json, Value};

const USAGE: &str = "usage: prime-agent harness <export --out <dir> | plan --export <dir> --seed <tree> --tree <tree> --reason <text> --out <file> | import --plan <file> [--dry-run] | stage --plan <file> --cycle <name> | pending | resolve --cycle <name> --as <applied|dismissed> | path [--cwd <dir>]>";
const PENDING_DIR_NAME: &str = "pending";
const SKILL_BACKUPS_DIR_NAME: &str = "skill-backups";

pub fn run_harness_command(args: &[String]) -> i32 {
    let agent_dir = crate::config::get_agent_dir();
    let result = match args.first().map(String::as_str) {
        Some("export") => flags(&args[1..], &["--out", "--add", "--omit"], &[]).and_then(|flags| {
            export(
                &agent_dir,
                Path::new(&flags.value("--out")?),
                &flags.values("--add"),
                &flags.values("--omit"),
            )
        }),
        Some("plan") => flags(
            &args[1..],
            &["--export", "--seed", "--tree", "--reason", "--out"],
            &[],
        )
        .and_then(|flags| {
            let plan = plan(
                Path::new(&flags.value("--export")?),
                Path::new(&flags.value("--seed")?),
                Path::new(&flags.value("--tree")?),
                &flags.value("--reason")?,
            )?;
            let out = flags.value("--out")?;
            let text = serde_json::to_string_pretty(&plan).map_err(|error| error.to_string())?;
            std::fs::write(&out, text + "\n").map_err(|error| format!("{out}: {error}"))?;
            Ok(json!({
                "plan": out,
                "edits": plan["edits"].as_array().map_or(0, Vec::len),
                "skills": plan["skills"].as_array().map_or(0, Vec::len),
            }))
        }),
        Some("import") => flags(&args[1..], &["--plan"], &["--dry-run"]).and_then(|flags| {
            let path = flags.value("--plan")?;
            let text = std::fs::read_to_string(&path).map_err(|error| format!("{path}: {error}"))?;
            let cwd = std::env::current_dir().unwrap_or_default();
            import(&agent_dir, &cwd, &text, flags.has("--dry-run"))
        }),
        Some("stage") => flags(&args[1..], &["--plan", "--cycle"], &[]).and_then(|flags| {
            pending(
                &agent_dir,
                Pending::Stage {
                    plan: PathBuf::from(flags.value("--plan")?),
                    cycle: flags.value("--cycle")?,
                },
            )
        }),
        Some("pending") => {
            flags(&args[1..], &[], &[]).and_then(|_| pending(&agent_dir, Pending::List))
        }
        Some("resolve") => flags(&args[1..], &["--cycle", "--as"], &[]).and_then(|flags| {
            pending(
                &agent_dir,
                Pending::Resolve {
                    cycle: flags.value("--cycle")?,
                    outcome: flags.value("--as")?,
                },
            )
        }),
        Some("path") => flags(&args[1..], &["--cwd"], &[]).map(|flags| {
            let cwd = flags
                .value("--cwd")
                .map_or_else(|_| std::env::current_dir().unwrap_or_default(), PathBuf::from);
            json!({
                "global": get_global_harness_state_dir(&agent_dir),
                "project": pa_core::refinement::project::get_project_harness_state_dir(&agent_dir, &cwd),
                "projectKey": pa_core::refinement::project::project_key(&cwd),
            })
        }),
        _ => Err(USAGE.to_string()),
    };
    match result {
        Ok(value) => {
            println!("{value}");
            0
        }
        Err(error) => {
            eprintln!("Error: {error}");
            eprintln!("Run `prime-agent help harness` for usage.");
            1
        }
    }
}

/// Parsed `--flag value` pairs and bare switches.
struct Flags(Vec<(String, Option<String>)>);

impl Flags {
    fn value(&self, name: &str) -> Result<String, String> {
        self.0
            .iter()
            .find(|(flag, _)| flag == name)
            .and_then(|(_, value)| value.clone())
            .ok_or_else(|| format!("{name} is required"))
    }

    /// Every value of a flag that may repeat.
    fn values(&self, name: &str) -> Vec<String> {
        self.0
            .iter()
            .filter(|(flag, _)| flag == name)
            .filter_map(|(_, value)| value.clone())
            .collect()
    }

    fn has(&self, name: &str) -> bool {
        self.0.iter().any(|(flag, _)| flag == name)
    }
}

fn flags(args: &[String], valued: &[&str], switches: &[&str]) -> Result<Flags, String> {
    let mut parsed = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if switches.contains(&arg.as_str()) {
            parsed.push((arg.clone(), None));
        } else if valued.contains(&arg.as_str()) {
            let value = args.next().ok_or_else(|| format!("{arg} needs a value"))?;
            parsed.push((arg.clone(), Some(value.clone())));
        } else {
            return Err(format!("unknown argument {arg:?}; {USAGE}"));
        }
    }
    Ok(Flags(parsed))
}

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// Entries of one store whose refinement a probe kept and no offline cycle
/// has judged yet: the candidates the next cycle tests on more tasks.
fn probe_kept(harness_dir: &Path, scope: HarnessScope) -> Vec<Value> {
    let verdicts = load_global_verdicts(harness_dir);
    let state = load_harness_state(harness_dir, scope);
    load_refinement_history_log(harness_dir, scope)
        .iter()
        .filter(|refinement| {
            let of_refinement = |prefix: &str| {
                verdicts.iter().any(|verdict| {
                    verdict.refinement_id == refinement.id && verdict.evidence.starts_with(prefix)
                })
            };
            of_refinement("probe:") && !of_refinement("harness2")
        })
        .filter_map(|refinement| {
            // Only what is still in the store: a later edit or rollback wins.
            let entries: Vec<Value> = refinement
                .applied_edits
                .iter()
                .filter(|edit| edit.applied && edit.action != RefinementAction::Delete)
                .filter_map(|edit| state.entries.get(&edit.kind)?.get(&edit.id))
                .filter(|entry| {
                    matches!(entry.kind, RefinementKind::Prompt | RefinementKind::Memory)
                })
                .map(|entry| {
                    json!({
                        "kind": entry.kind, "id": entry.id, "title": entry.title,
                        "content": entry.content, "path": entry.path,
                    })
                })
                .collect();
            (!entries.is_empty()).then(|| {
                json!({
                    "refinementId": refinement.id,
                    "summary": refinement.summary,
                    "entries": entries,
                })
            })
        })
        .collect()
}

/// A name from a plan or the command line that becomes a path segment.
fn plain_name(what: &str, name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.starts_with('.')
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(format!("{what} name {name:?} is not a plain folder name"));
    }
    Ok(())
}

/// What to do with the plans waiting for approval.
enum Pending {
    Stage { plan: PathBuf, cycle: String },
    List,
    Resolve { cycle: String, outcome: String },
}

fn pending(agent_dir: &Path, action: Pending) -> Result<Value, String> {
    let dir = get_global_harness_state_dir(agent_dir).join(PENDING_DIR_NAME);
    let io = |error: std::io::Error| format!("{}: {error}", dir.display());
    let file = |cycle: &str| plain_name("cycle", cycle).map(|()| dir.join(format!("{cycle}.json")));
    match action {
        Pending::Stage { plan, cycle } => {
            let target = file(&cycle)?;
            let text = std::fs::read_to_string(&plan)
                .map_err(|error| format!("{}: {error}", plan.display()))?;
            serde_json::from_str::<Value>(&text).map_err(|error| error.to_string())?;
            std::fs::create_dir_all(&dir).map_err(io)?;
            std::fs::write(&target, text).map_err(io)?;
            Ok(json!({ "cycle": cycle, "path": target }))
        }
        Pending::List => {
            let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
                .into_iter()
                .flatten()
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
                .collect();
            paths.sort();
            let plans: Vec<Value> = paths
                .iter()
                .filter_map(|path| {
                    let plan: Value =
                        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
                    Some(json!({
                        "cycle": path.file_stem()?.to_string_lossy(),
                        "path": path,
                        "plan": plan,
                    }))
                })
                .collect();
            Ok(json!({ "pending": plans }))
        }
        Pending::Resolve { cycle, outcome } => {
            if outcome != "applied" && outcome != "dismissed" {
                return Err(format!("--as takes applied or dismissed, got {outcome:?}"));
            }
            let source = file(&cycle)?;
            let target = dir.join(&outcome).join(format!("{cycle}.json"));
            std::fs::create_dir_all(dir.join(&outcome)).map_err(io)?;
            std::fs::rename(&source, &target)
                .map_err(|error| format!("{}: {error}", source.display()))?;
            Ok(json!({ "cycle": cycle, "as": outcome, "path": target }))
        }
    }
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    std::fs::write(path, text + "\n").map_err(|error| format!("{}: {error}", path.display()))
}

fn export(agent_dir: &Path, out: &Path, add: &[String], omit: &[String]) -> Result<Value, String> {
    let io = |error: std::io::Error| format!("{}: {error}", out.display());
    let tree_dir = out.join("tree");
    if tree_dir.exists() {
        std::fs::remove_dir_all(&tree_dir).map_err(io)?;
    }
    std::fs::create_dir_all(&tree_dir).map_err(io)?;
    let global_dir = get_global_harness_state_dir(agent_dir);
    let mut state = load_harness_state(&global_dir, HarnessScope::Global);
    let kind_of = |name: &str| {
        serde_json::from_value::<RefinementKind>(json!(name))
            .map_err(|_| format!("unknown entry kind {name:?}"))
    };
    for spec in add {
        let mut parts = spec.splitn(3, ':');
        let (Some(key), Some(kind), Some(id)) = (parts.next(), parts.next(), parts.next()) else {
            return Err(format!("--add takes <project>:<kind>:<id>, got {spec:?}"));
        };
        plain_name("project", key)?;
        let kind = kind_of(kind)?;
        let project = agent_dir
            .join(pa_core::refinement::project::PROJECTS_DIR_NAME)
            .join(key)
            .join(pa_core::refinement::HARNESS_STATE_DIR_NAME);
        let entry = load_harness_state(&project, HarnessScope::Local)
            .entries
            .get(&kind)
            .and_then(|entries| entries.get(id))
            .cloned()
            .ok_or_else(|| format!("project {key} has no {kind:?} entry {id:?}"))?;
        state
            .entries
            .entry(kind)
            .or_default()
            .insert(id.to_string(), entry);
    }
    for spec in omit {
        let Some((kind, id)) = spec.split_once(':') else {
            return Err(format!("--omit takes <kind>:<id>, got {spec:?}"));
        };
        let kind = kind_of(kind)?;
        if state
            .entries
            .get_mut(&kind)
            .and_then(|entries| entries.remove(id))
            .is_none()
        {
            return Err(format!("the global harness has no {kind:?} entry {id:?}"));
        }
    }
    for (file, text) in tree::export_sections(&state) {
        let path = tree_dir.join(file);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(io)?;
        }
        std::fs::write(path, text).map_err(io)?;
    }
    let skills = tree::skill_names(agent_dir);
    for name in &skills {
        copy_dir(
            &agent_dir.join(tree::SKILLS_DIR).join(name),
            &tree_dir.join(tree::SKILLS_DIR).join(name),
        )
        .map_err(io)?;
    }
    let entries: Vec<String> = [RefinementKind::Prompt, RefinementKind::Memory]
        .iter()
        .flat_map(|kind| state.entries[kind].keys().cloned())
        .collect();
    write_json(
        &out.join("export.json"),
        &json!({ "agentDir": agent_dir, "entries": entries, "skills": skills }),
    )?;
    // What online use has recorded, for the optimiser: the latest verdict on
    // each global refinement, every refinement (so a rollback is visible),
    // and the probe-kept entries of the global store and of each project.
    let verdicts = load_global_verdicts(&global_dir);
    let refinements: Vec<Value> = load_global_refinement_history(&global_dir)
        .iter()
        .map(|refinement| {
            json!({
                "id": refinement.id,
                "summary": refinement.summary,
                "rollbackOf": refinement.rollback_of,
                "edits": refinement.applied_edits.iter().filter(|edit| edit.applied).map(|edit| json!({
                    "action": edit.action, "kind": edit.kind, "id": edit.id,
                    "title": edit.title, "content": edit.content,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let mut to_promote = Vec::new();
    if let Ok(projects) =
        std::fs::read_dir(agent_dir.join(pa_core::refinement::project::PROJECTS_DIR_NAME))
    {
        let mut projects: Vec<PathBuf> = projects.flatten().map(|entry| entry.path()).collect();
        projects.sort();
        for project in projects {
            let kept = probe_kept(
                &project.join(pa_core::refinement::HARNESS_STATE_DIR_NAME),
                HarnessScope::Local,
            );
            if !kept.is_empty() {
                to_promote.push(json!({
                    "project": project.file_name().map(|name| name.to_string_lossy().to_string()),
                    "refinements": kept,
                }));
            }
        }
    }
    write_json(
        &out.join("verdicts.json"),
        &json!({
            "verdicts": latest_verdicts(&verdicts),
            "refinements": refinements,
            "toPromote": to_promote,
            "toConfirm": probe_kept(&global_dir, HarnessScope::Global),
        }),
    )?;
    Ok(json!({
        "tree": tree_dir,
        "entries": entries.len(),
        "skills": skills.len(),
        "toPromote": to_promote.len(),
    }))
}

fn plan(export_dir: &Path, seed: &Path, winner: &Path, reason: &str) -> Result<Value, String> {
    for dir in [seed, winner] {
        if !dir.is_dir() {
            return Err(format!("{} is not a directory", dir.display()));
        }
    }
    let export_path = export_dir.join("export.json");
    let exported: Value = std::fs::read_to_string(&export_path)
        .map_err(|error| format!("{}: {error}", export_path.display()))
        .and_then(|text| serde_json::from_str(&text).map_err(|error| error.to_string()))?;
    let exported_skills: BTreeSet<String> = exported["skills"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    let proposal = tree::plan_from_trees(seed, winner, reason);
    let (written, deleted) = tree::skill_changes(seed, winner, &exported_skills);
    let skills: Vec<Value> = written
        .iter()
        .map(|name| {
            json!({
                "action": "write", "name": name,
                "source": winner.join(tree::SKILLS_DIR).join(name),
            })
        })
        .chain(
            deleted
                .iter()
                .map(|name| json!({ "action": "delete", "name": name })),
        )
        .collect();
    Ok(json!({
        "summary": proposal.summary,
        "rationale": proposal.rationale,
        "expectedOutcome": proposal.expected_outcome,
        "edits": proposal.edits,
        "skills": skills,
        "reason": reason,
    }))
}

fn import(agent_dir: &Path, cwd: &Path, plan_text: &str, dry_run: bool) -> Result<Value, String> {
    let settings = pa_core::settings::SettingsManager::create(cwd, agent_dir);
    if settings.settings().harness_refinements == Some(false) {
        return Err("Harness refinements are disabled in /settings".to_string());
    }
    let soft_limit = settings
        .settings()
        .refine
        .as_ref()
        .and_then(|refine| refine.always_on_soft_limit)
        .unwrap_or(DEFAULT_ALWAYS_ON_SOFT_LIMIT);
    let plan_value: Value = serde_json::from_str(plan_text).map_err(|error| error.to_string())?;
    let mut proposal = pa_core::refinement::planner::parse_proposal(plan_text)?;
    let reason = plan_value["reason"]
        .as_str()
        .unwrap_or(&proposal.summary)
        .to_string();
    let global_dir = get_global_harness_state_dir(agent_dir);
    let project_dir = |key: &str| {
        agent_dir
            .join(pa_core::refinement::project::PROJECTS_DIR_NAME)
            .join(key)
            .join(pa_core::refinement::HARNESS_STATE_DIR_NAME)
    };
    // Project entries: a promoted one joins the global refinement and leaves
    // its project store; a deleted one only leaves. Checked before any write.
    let global_before = load_harness_state(&global_dir, HarnessScope::Global);
    let mut project_plans: Vec<(String, RefinementProposal)> = Vec::new();
    for project in plan_value["projects"].as_array().into_iter().flatten() {
        let key = project["project"].as_str().unwrap_or_default();
        plain_name("project", key)?;
        let state = load_harness_state(&project_dir(key), HarnessScope::Local);
        let mut leaving = Vec::new();
        for (list, promoted) in [("promote", true), ("delete", false)] {
            for item in project[list].as_array().into_iter().flatten() {
                let kind: RefinementKind = serde_json::from_value(item["kind"].clone())
                    .map_err(|error| format!("project {key}: {error}"))?;
                let id = item["id"].as_str().unwrap_or_default();
                let entry = state
                    .entries
                    .get(&kind)
                    .and_then(|entries| entries.get(id))
                    .ok_or_else(|| format!("project {key} has no {kind:?} entry {id:?}"))?;
                if promoted {
                    let known = global_before
                        .entries
                        .get(&kind)
                        .is_some_and(|entries| entries.contains_key(id));
                    let mut metadata = entry.metadata.clone();
                    metadata.insert("promotedFrom".to_string(), json!(key));
                    proposal.edits.push(RefinementEdit {
                        action: Some(if known {
                            RefinementAction::Update
                        } else {
                            RefinementAction::Create
                        }),
                        kind: Some(kind),
                        id: Some(id.to_string()),
                        title: Some(entry.title.clone()),
                        content: Some(entry.content.clone()),
                        path: (!entry.path.is_empty()).then(|| entry.path.clone()),
                        reference: Some(entry.reference.clone()),
                        arguments: Some(entry.arguments.clone()),
                        metadata: Some(metadata),
                        reason: Some(format!("promoted from project {key}: {reason}")),
                    });
                }
                leaving.push(RefinementEdit {
                    action: Some(RefinementAction::Delete),
                    kind: Some(kind),
                    id: Some(id.to_string()),
                    reason: Some(if promoted {
                        format!("promoted to global: {reason}")
                    } else {
                        reason.clone()
                    }),
                    ..RefinementEdit::default()
                });
            }
        }
        if !leaving.is_empty() {
            project_plans.push((
                key.to_string(),
                RefinementProposal {
                    summary: format!("harness2: {reason}"),
                    rationale:
                        "Offline harness adaptation tested these project entries on more tasks."
                            .to_string(),
                    expected_outcome: String::new(),
                    edits: leaving,
                },
            ));
        }
    }
    // Verdicts on earlier refinements, each for the global log or one project's.
    let mut verdicts: Vec<(PathBuf, RefinementVerdict)> = Vec::new();
    for item in plan_value["verdicts"].as_array().into_iter().flatten() {
        let dir = match item["project"].as_str() {
            Some(key) => {
                plain_name("project", key)?;
                project_dir(key)
            }
            None => global_dir.clone(),
        };
        let mut verdict: RefinementVerdict =
            serde_json::from_value(item.clone()).map_err(|error| format!("verdict: {error}"))?;
        verdict.created_at = pa_core::session::manager::format_iso_now();
        verdicts.push((dir, verdict));
    }
    // Skills are checked whole before anything is written.
    let mut skills: Vec<(bool, String, Option<PathBuf>)> = Vec::new();
    for skill in plan_value["skills"].as_array().into_iter().flatten() {
        let name = skill["name"].as_str().unwrap_or_default().to_string();
        plain_name("skill", &name)?;
        match skill["action"].as_str() {
            Some("write") => {
                let source = PathBuf::from(skill["source"].as_str().unwrap_or_default());
                if !source.join("SKILL.md").is_file() {
                    return Err(format!(
                        "skill {name}: {} has no SKILL.md",
                        source.display()
                    ));
                }
                let mut pending = vec![source.clone()];
                while let Some(current) = pending.pop() {
                    for entry in std::fs::read_dir(&current).into_iter().flatten().flatten() {
                        let path = entry.path();
                        if path.is_dir() {
                            pending.push(path);
                        } else if std::fs::read_to_string(&path).is_ok_and(|text| {
                            pa_core::refinement::planner::looks_like_credential(&text)
                        }) {
                            return Err(format!(
                                "skill {name}: {} looks like a credential",
                                path.display()
                            ));
                        }
                    }
                }
                skills.push((true, name, Some(source)));
            }
            Some("delete") => skills.push((false, name, None)),
            other => return Err(format!("skill {name}: unknown action {other:?}")),
        }
    }
    let id = generate_refinement_id();
    let options = RefineOptions {
        global: true,
        ..RefineOptions::default()
    };
    let make_plan = || RefinementPlan {
        proposal: proposal.clone(),
        id: id.clone(),
        rollback_of: None,
        rollback_scope: None,
    };
    let global_edits = !proposal.edits.is_empty();
    // Nothing to apply globally (a plan of verdicts or project deletes only)
    // leaves no global refinement behind.
    let (result, chars) = if dry_run || !global_edits {
        let mut state = load_harness_state(&global_dir, HarnessScope::Global);
        let result = apply_refinement_plan(&mut state, make_plan(), &options, None, false);
        (result, always_on_chars(&state))
    } else {
        let ((mut result, chars), state_path) =
            update_harness_state(&global_dir, HarnessScope::Global, |state| {
                let result = apply_refinement_plan(state, make_plan(), &options, None, false);
                (result, always_on_chars(state))
            })
            .map_err(|error| format!("{error:#}"))?;
        result.harness_state_path = state_path.to_string_lossy().to_string();
        append_global_refinement(&global_dir, &result).map_err(|error| format!("{error:#}"))?;
        // The optimiser's own evidence: one verdict per applied edit, so
        // `/refine review` and the refiner see the offline result.
        let evidence = format!("harness2 {reason}");
        let helped: Vec<RefinementVerdict> = result
            .applied_edits
            .iter()
            .filter(|edit| edit.applied)
            .map(|edit| RefinementVerdict {
                refinement_id: id.clone(),
                edit_id: Some(edit.id.clone()),
                verdict: Verdict::Helped,
                evidence: evidence.clone(),
                created_at: pa_core::session::manager::format_iso_now(),
            })
            .collect();
        append_global_verdicts(&global_dir, &helped).map_err(|error| format!("{error:#}"))?;
        (result, chars)
    };
    let mut project_rows = Vec::new();
    for (key, leaving) in project_plans {
        let dir = project_dir(&key);
        let plan = RefinementPlan {
            proposal: leaving,
            id: generate_refinement_id(),
            rollback_of: None,
            rollback_scope: None,
        };
        let local = RefineOptions::default();
        let result = if dry_run {
            let mut state = load_harness_state(&dir, HarnessScope::Local);
            apply_refinement_plan(&mut state, plan, &local, None, false)
        } else {
            let (mut result, state_path) =
                update_harness_state(&dir, HarnessScope::Local, |state| {
                    apply_refinement_plan(state, plan, &local, None, false)
                })
                .map_err(|error| format!("{error:#}"))?;
            result.harness_state_path = state_path.to_string_lossy().to_string();
            append_global_refinement(&dir, &result).map_err(|error| format!("{error:#}"))?;
            result
        };
        project_rows.push(json!({
            "project": key,
            "refinementId": result.id,
            "edits": result.applied_edits.iter().map(|edit| json!({
                "action": edit.action, "kind": edit.kind, "id": edit.id,
                "applied": edit.applied, "error": edit.error,
            })).collect::<Vec<_>>(),
        }));
    }
    if !dry_run {
        for (dir, verdict) in &verdicts {
            append_global_verdicts(dir, std::slice::from_ref(verdict))
                .map_err(|error| format!("{error:#}"))?;
        }
    }
    let mut skill_rows = Vec::new();
    for (write, name, source) in &skills {
        let target = agent_dir.join(tree::SKILLS_DIR).join(name);
        let backup = global_dir.join(SKILL_BACKUPS_DIR_NAME).join(&id).join(name);
        if !dry_run {
            let io = |error: std::io::Error| format!("skill {name}: {error}");
            if target.exists() {
                // The copy `/refine rollback` cannot restore: kept for a manual undo.
                copy_dir(&target, &backup).map_err(io)?;
                std::fs::remove_dir_all(&target).map_err(io)?;
            }
            if let Some(source) = source {
                copy_dir(source, &target).map_err(io)?;
            }
        }
        skill_rows.push(json!({
            "action": if *write { "write" } else { "delete" },
            "name": name,
            "path": target,
            "backup": backup.exists().then_some(backup),
        }));
    }
    let over = chars as u64 > soft_limit;
    Ok(json!({
        "refinementId": global_edits.then_some(id),
        "dryRun": dry_run,
        "projects": project_rows,
        "verdicts": verdicts.len(),
        "summary": result.summary,
        "edits": result.applied_edits.iter().map(|edit| json!({
            "action": edit.action, "kind": edit.kind, "id": edit.id,
            "applied": edit.applied, "error": edit.error,
        })).collect::<Vec<_>>(),
        "skills": skill_rows,
        "alwaysOnChars": chars,
        "alwaysOnSoftLimit": soft_limit,
        "warning": over.then(|| format!(
            "always-on harness text is {chars} characters, past the soft limit of {soft_limit}; trim prompt or memory entries"
        )),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(path: &Path) -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn an_optimised_tree_comes_back_as_one_global_refinement_with_its_skill() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path().join("agent");
        let skill = agent.join("skills").join("review");
        std::fs::create_dir_all(&skill).unwrap();
        std::fs::write(
            skill.join("SKILL.md"),
            "---\nname: review\n---\nRead the diff.\n",
        )
        .unwrap();
        // The agent's harness holds one memory already.
        let seeded = import(
            &agent,
            dir.path(),
            r#"{"summary":"seed","edits":[{"action":"create","kind":"memory","id":"docker","title":"Tests","content":"Tests need docker.","reason":"seed"}]}"#,
            false,
        )
        .unwrap();
        assert_eq!(seeded["edits"][0]["applied"], true);

        let out = dir.path().join("export");
        export(&agent, &out, &[], &[]).unwrap();
        let seed = out.join("tree");
        assert_eq!(read(&out.join("export.json"))["skills"], json!(["review"]));
        // Unchanged, the export plans nothing.
        let unchanged = plan(&out, &seed, &seed, "cycle 0").unwrap();
        assert_eq!(
            (unchanged["edits"].clone(), unchanged["skills"].clone()),
            (json!([]), json!([]))
        );

        // The optimiser's winner: a new guardrail, a rewritten skill.
        let winner = dir.path().join("winner");
        copy_dir(&seed, &winner).unwrap();
        std::fs::write(
            winner.join("guardrails.md"),
            "## Scope\nChange only what the task names.\n",
        )
        .unwrap();
        std::fs::write(
            winner.join("skills/review/SKILL.md"),
            "---\nname: review\n---\nRead the diff twice.\n",
        )
        .unwrap();
        let planned = plan(&out, &seed, &winner, "cycle 1: +0.4 on 5 tasks").unwrap();
        let applied = import(&agent, dir.path(), &planned.to_string(), false).unwrap();
        assert_eq!(
            applied["edits"],
            json!([{ "action": "create", "kind": "prompt", "id": "h2-prompt-scope", "applied": true, "error": null }])
        );
        assert_eq!(
            std::fs::read_to_string(skill.join("SKILL.md")).unwrap(),
            "---\nname: review\n---\nRead the diff twice.\n"
        );
        let backup = PathBuf::from(applied["skills"][0]["backup"].as_str().unwrap());
        assert!(std::fs::read_to_string(backup.join("SKILL.md"))
            .unwrap()
            .contains("Read the diff.\n"));

        // One global refinement in the history, with the offline verdict on its edit.
        let global = get_global_harness_state_dir(&agent);
        let history = load_global_refinement_history(&global);
        let last = history.last().unwrap();
        assert_eq!(last.summary, "harness2: cycle 1: +0.4 on 5 tasks");
        assert_eq!(
            load_global_verdicts(&global)
                .iter()
                .filter(|verdict| verdict.refinement_id == last.id)
                .map(|verdict| (
                    verdict.edit_id.clone(),
                    verdict.verdict,
                    verdict.evidence.clone()
                ))
                .collect::<Vec<_>>(),
            vec![(
                Some("h2-prompt-scope".to_string()),
                Verdict::Helped,
                "harness2 cycle 1: +0.4 on 5 tasks".to_string()
            )]
        );
    }

    #[test]
    fn a_plan_naming_a_skill_outside_the_skills_folder_is_refused_before_any_write() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path().join("agent");
        let error = import(
            &agent,
            dir.path(),
            r#"{"summary":"x","edits":[{"action":"create","kind":"memory","id":"m","title":"M","content":"C","reason":"r"}],"skills":[{"action":"delete","name":"../harness"}]}"#,
            false,
        )
        .unwrap_err();
        assert!(error.contains("not a plain folder name"), "{error}");
        assert!(!get_global_harness_state_dir(&agent).exists());
    }

    #[test]
    fn a_probe_kept_project_entry_is_offered_then_promoted_to_global() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path().join("agent");
        let project = agent.join("projects/repo-abc123/harness");
        // A project refinement that a probe kept.
        let plan = RefinementPlan {
            proposal: pa_core::refinement::planner::parse_proposal(
                r#"{"summary":"lint","edits":[{"action":"create","kind":"memory","id":"lint","title":"Lint","content":"Run make lint.","reason":"r"}]}"#,
            )
            .unwrap(),
            id: "p1".to_string(),
            rollback_of: None,
            rollback_scope: None,
        };
        let (result, _) = update_harness_state(&project, HarnessScope::Local, |state| {
            apply_refinement_plan(state, plan, &RefineOptions::default(), None, false)
        })
        .unwrap();
        append_global_refinement(&project, &result).unwrap();
        let probe = RefinementVerdict {
            refinement_id: "p1".to_string(),
            edit_id: Some("lint".to_string()),
            verdict: Verdict::Helped,
            evidence: "probe: kept [0], dropped [] of 1 proposed edits".to_string(),
            created_at: String::new(),
        };
        append_global_verdicts(&project, &[probe]).unwrap();

        let out = dir.path().join("export");
        export(&agent, &out, &[], &[]).unwrap();
        let offered = read(&out.join("verdicts.json"))["toPromote"].clone();
        assert_eq!(
            (
                offered[0]["project"].clone(),
                offered[0]["refinements"][0]["refinementId"].clone(),
                offered[0]["refinements"][0]["entries"][0]["content"].clone()
            ),
            (json!("repo-abc123"), json!("p1"), json!("Run make lint."))
        );

        // The cycle can first see the harness as it would be with the entry promoted.
        let variant = dir.path().join("variant");
        export(
            &agent,
            &variant,
            &["repo-abc123:memory:lint".to_string()],
            &[],
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(variant.join("tree").join(tree::MEMORY_FILE)).unwrap(),
            "\n## Lint\n<!-- h2:id=lint -->\nRun make lint.\n<!-- /h2 -->\n"
        );
        assert!(!get_global_harness_state_dir(&agent)
            .join("harness_state.json")
            .exists());

        // The cycle tested it on more tasks and it held up.
        let applied = import(
            &agent,
            dir.path(),
            &json!({
                "summary": "harness2: cycle 2", "reason": "cycle 2: +0.3 on 4 tasks", "edits": [],
                "projects": [{ "project": "repo-abc123", "promote": [{ "kind": "memory", "id": "lint" }] }],
                "verdicts": [{
                    "project": "repo-abc123", "refinementId": "p1", "editId": "lint",
                    "verdict": "helped", "evidence": "harness2 cycle 2: +0.3 on 4 tasks",
                }],
            })
            .to_string(),
            false,
        )
        .unwrap();
        let global = get_global_harness_state_dir(&agent);
        let promoted = &load_harness_state(&global, HarnessScope::Global).entries
            [&RefinementKind::Memory]["lint"];
        assert_eq!(
            (
                promoted.content.as_str(),
                promoted.metadata["promotedFrom"].clone()
            ),
            ("Run make lint.", json!("repo-abc123"))
        );
        assert!(
            load_harness_state(&project, HarnessScope::Local).entries[&RefinementKind::Memory]
                .is_empty()
        );
        // The project's own history can undo its half, and its log holds the offline verdict.
        assert_eq!(
            load_refinement_history_log(&project, HarnessScope::Local)
                .last()
                .unwrap()
                .id,
            applied["projects"][0]["refinementId"].as_str().unwrap()
        );
        assert_eq!(
            load_global_verdicts(&project).last().unwrap().evidence,
            "harness2 cycle 2: +0.3 on 4 tasks"
        );
        export(&agent, &out, &[], &[]).unwrap();
        assert_eq!(read(&out.join("verdicts.json"))["toPromote"], json!([]));
    }

    #[test]
    fn a_staged_plan_waits_until_it_is_resolved() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path().join("agent");
        let plan = dir.path().join("plan.json");
        std::fs::write(&plan, r#"{"summary":"harness2: cycle 1","edits":[]}"#).unwrap();
        let stage = |cycle: &str| {
            pending(
                &agent,
                Pending::Stage {
                    plan: plan.clone(),
                    cycle: cycle.to_string(),
                },
            )
        };
        stage("2026-11-10-a").unwrap();
        stage("2026-11-10-b").unwrap();
        assert!(stage("../escape")
            .unwrap_err()
            .contains("not a plain folder name"));

        pending(
            &agent,
            Pending::Resolve {
                cycle: "2026-11-10-a".to_string(),
                outcome: "dismissed".to_string(),
            },
        )
        .unwrap();
        let waiting = pending(&agent, Pending::List).unwrap();
        let global = get_global_harness_state_dir(&agent);
        assert_eq!(
            waiting,
            json!({ "pending": [{
                "cycle": "2026-11-10-b",
                "path": global.join("pending/2026-11-10-b.json"),
                "plan": { "summary": "harness2: cycle 1", "edits": [] },
            }] })
        );
        assert!(global.join("pending/dismissed/2026-11-10-a.json").is_file());
    }
}
