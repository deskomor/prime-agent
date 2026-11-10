//! `prime-agent harness`: the global harness as a file tree, for an offline
//! harness optimiser such as Harness² (fork-only).
//!
//! Sub-verbs:
//!   - `export --out <dir>`: write `<dir>/tree` (prompt and memory entries as
//!     marked markdown sections, folder skills), `<dir>/export.json` (what
//!     was exported) and `<dir>/verdicts.json` (what `/refine review` and
//!     probes have recorded, for the optimiser to read).
//!   - `plan --export <dir> --seed <tree> --tree <tree> --reason <text>
//!     --out <file>`: diff the tree an optimiser ended with against the seed
//!     it started from and write the difference as a refinement plan.
//!   - `import --plan <file> [--dry-run]`: apply a plan as ONE global
//!     refinement, through the same validation and history as `/refine`, so
//!     `/refine rollback <id>` undoes its entries.
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
use pa_core::refinement::verdicts::{
    append_global_verdicts, latest_verdicts, load_global_verdicts, RefinementVerdict, Verdict,
};
use pa_core::refinement::{
    append_global_refinement, get_global_harness_state_dir, load_global_refinement_history,
    load_harness_state, load_refinement_history_log, tree, update_harness_state, HarnessScope,
    RefinementAction, RefinementKind,
};
use serde_json::{json, Value};

const USAGE: &str = "usage: prime-agent harness <export --out <dir> | plan --export <dir> --seed <tree> --tree <tree> --reason <text> --out <file> | import --plan <file> [--dry-run] | path [--cwd <dir>]>";
const SKILL_BACKUPS_DIR_NAME: &str = "skill-backups";

pub fn run_harness_command(args: &[String]) -> i32 {
    let agent_dir = crate::config::get_agent_dir();
    let result = match args.first().map(String::as_str) {
        Some("export") => flags(&args[1..], &["--out"], &[])
            .and_then(|flags| export(&agent_dir, Path::new(&flags.value("--out")?))),
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

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    std::fs::write(path, text + "\n").map_err(|error| format!("{}: {error}", path.display()))
}

fn export(agent_dir: &Path, out: &Path) -> Result<Value, String> {
    let io = |error: std::io::Error| format!("{}: {error}", out.display());
    let tree_dir = out.join("tree");
    if tree_dir.exists() {
        std::fs::remove_dir_all(&tree_dir).map_err(io)?;
    }
    std::fs::create_dir_all(&tree_dir).map_err(io)?;
    let global_dir = get_global_harness_state_dir(agent_dir);
    let state = load_harness_state(&global_dir, HarnessScope::Global);
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
    let proposal = pa_core::refinement::planner::parse_proposal(plan_text)?;
    // Skills are checked whole before anything is written.
    let mut skills: Vec<(bool, String, Option<PathBuf>)> = Vec::new();
    for skill in plan_value["skills"].as_array().into_iter().flatten() {
        let name = skill["name"].as_str().unwrap_or_default().to_string();
        // The name must stay inside `<agent dir>/skills`.
        if name.is_empty()
            || name.starts_with('.')
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        {
            return Err(format!("skill name {name:?} is not a plain folder name"));
        }
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
    let global_dir = get_global_harness_state_dir(agent_dir);
    let make_plan = || RefinementPlan {
        proposal: proposal.clone(),
        id: id.clone(),
        rollback_of: None,
        rollback_scope: None,
    };
    let (result, chars) = if dry_run {
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
        let evidence = format!(
            "harness2 {}",
            plan_value["reason"].as_str().unwrap_or(&proposal.summary)
        );
        let verdicts: Vec<RefinementVerdict> = result
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
        append_global_verdicts(&global_dir, &verdicts).map_err(|error| format!("{error:#}"))?;
        (result, chars)
    };
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
        "refinementId": id,
        "dryRun": dry_run,
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
        export(&agent, &out).unwrap();
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
}
