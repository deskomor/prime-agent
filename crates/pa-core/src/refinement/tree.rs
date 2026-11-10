//! The harness as a file tree, for an offline harness optimiser (Harness²).
//!
//! `export` writes the global store's prompt and memory entries into the
//! tree's markdown files, one marked section per entry. `plan` reads two
//! trees (the seed an optimiser started from and the tree it ended with) and
//! returns the difference as an ordinary refinement proposal, so the result
//! applies through the same validation, history and rollback as `/refine`.
//!
//! A section looks like this; the markers carry the entry id through the
//! round trip, and the text between them is the entry content verbatim:
//!
//! ```text
//! ## Title
//! <!-- h2:id=entry-id -->
//! content
//! <!-- /h2 -->
//! ```
//!
//! Text outside markers that the seed does not have becomes a new entry.
//! Fork-only.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use super::planner::{RefinementEdit, RefinementProposal};
use super::{HarnessState, RefinementAction, RefinementKind};

pub const MEMORY_FILE: &str = "LongTermMEMORY.md";
pub const GUARDRAILS_FILE: &str = "guardrails.md";
pub const SYSTEM_PROMPT_FILE: &str = "systemprompt.md";
pub const TOOL_GUIDANCE_FILE: &str = "tool_descriptions/tool_guidance.md";
pub const SKILLS_DIR: &str = "skills";

/// Every tree file that holds entries, with the entry kind and the entry
/// `path` that sends an entry back to the same file on the next export.
const ENTRY_FILES: [(&str, RefinementKind, &str); 4] = [
    (MEMORY_FILE, RefinementKind::Memory, "harness2"),
    (
        GUARDRAILS_FILE,
        RefinementKind::Prompt,
        "harness2/guardrails",
    ),
    (
        SYSTEM_PROMPT_FILE,
        RefinementKind::Prompt,
        "harness2/systemprompt",
    ),
    (
        TOOL_GUIDANCE_FILE,
        RefinementKind::Prompt,
        "harness2/tool_guidance",
    ),
];

const OPEN_MARKER_PREFIX: &str = "<!-- h2:id=";
const MARKER_SUFFIX: &str = " -->";
const CLOSE_MARKER: &str = "<!-- /h2 -->";

/// One entry as the tree holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Section {
    id: Option<String>,
    title: String,
    content: String,
}

/// The tree file a prompt entry is written to: the one its `path` names,
/// else the guardrails file.
fn prompt_file(path: &str) -> &'static str {
    ENTRY_FILES
        .iter()
        .find(|(_, kind, entry_path)| *kind == RefinementKind::Prompt && *entry_path == path)
        .map_or(GUARDRAILS_FILE, |(file, _, _)| file)
}

/// The sections `export` appends to each tree file (relative path to text),
/// for the store's prompt and memory entries. Other kinds have no tree form.
#[must_use]
pub fn export_sections(state: &HarnessState) -> BTreeMap<&'static str, String> {
    let mut files: BTreeMap<&'static str, String> = BTreeMap::new();
    for (kind, entries) in &state.entries {
        for entry in entries.values() {
            let file = match kind {
                RefinementKind::Memory => MEMORY_FILE,
                RefinementKind::Prompt => prompt_file(&entry.path),
                RefinementKind::Skill | RefinementKind::Subagent | RefinementKind::Factory => {
                    continue
                }
            };
            let text = files.entry(file).or_default();
            // Writing to a String cannot fail.
            let _ = write!(
                text,
                "\n## {}\n{OPEN_MARKER_PREFIX}{}{MARKER_SUFFIX}\n{}\n{CLOSE_MARKER}\n",
                entry.title.replace('\n', " "),
                entry.id,
                entry.content
            );
        }
    }
    files
}

/// Split one tree file into its marked sections and its unmarked ones (one
/// per markdown heading; text before the first heading is titled by its
/// first line). HTML comments outside markers are not content.
fn parse_sections(text: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut unmarked: Vec<&str> = Vec::new();
    let mut lines = text.lines().peekable();
    let mut in_comment = false;
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if let Some(id) = trimmed
            .strip_prefix(OPEN_MARKER_PREFIX)
            .and_then(|rest| rest.strip_suffix(MARKER_SUFFIX))
        {
            // The heading right above the marker is this section's title.
            while unmarked.last().is_some_and(|last| last.trim().is_empty()) {
                unmarked.pop();
            }
            let title = match unmarked.last() {
                Some(last) if last.trim_start().starts_with('#') => {
                    let title = last.trim_start().trim_start_matches('#').trim().to_string();
                    unmarked.pop();
                    title
                }
                _ => id.to_string(),
            };
            flush_unmarked(&mut unmarked, &mut sections);
            let mut content: Vec<&str> = Vec::new();
            for body in lines.by_ref() {
                if body.trim() == CLOSE_MARKER {
                    break;
                }
                content.push(body);
            }
            sections.push(Section {
                id: Some(id.trim().to_string()),
                title,
                content: content.join("\n"),
            });
            continue;
        }
        if in_comment || trimmed.starts_with("<!--") {
            in_comment = !trimmed.ends_with("-->");
            continue;
        }
        if trimmed.starts_with('#') {
            flush_unmarked(&mut unmarked, &mut sections);
        }
        unmarked.push(line);
    }
    flush_unmarked(&mut unmarked, &mut sections);
    sections
}

/// Close the unmarked section collected so far, if it has any text.
fn flush_unmarked(lines: &mut Vec<&str>, sections: &mut Vec<Section>) {
    let text = lines.join("\n");
    lines.clear();
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    let (first, rest) = text.split_once('\n').unwrap_or((text, ""));
    let (title, content) = if first.starts_with('#') {
        (
            first.trim_start_matches('#').trim().to_string(),
            rest.trim(),
        )
    } else {
        (first.chars().take(80).collect(), text)
    };
    if content.is_empty() {
        return;
    }
    sections.push(Section {
        id: None,
        title,
        content: content.to_string(),
    });
}

/// Every section of the tree's entry files, keyed by tree file.
fn read_tree(tree: &Path) -> Vec<(&'static str, RefinementKind, &'static str, Vec<Section>)> {
    ENTRY_FILES
        .iter()
        .map(|(file, kind, path)| {
            let text = std::fs::read_to_string(tree.join(file)).unwrap_or_default();
            (*file, *kind, *path, parse_sections(&text))
        })
        .collect()
}

/// The edits that turn the harness the `seed` tree was exported from into
/// the one the `tree` describes. `reason` is recorded on every edit.
#[must_use]
pub fn plan_from_trees(seed: &Path, tree: &Path, reason: &str) -> RefinementProposal {
    let seed_files = read_tree(seed);
    let mut seed_marked: BTreeMap<(RefinementKind, String), (Section, &'static str)> =
        BTreeMap::new();
    let mut seed_unmarked: BTreeSet<(String, String)> = BTreeSet::new();
    for (_, kind, path, sections) in &seed_files {
        for section in sections {
            match &section.id {
                Some(id) => {
                    seed_marked.insert((*kind, id.clone()), (section.clone(), *path));
                }
                None => {
                    seed_unmarked.insert((section.title.clone(), section.content.clone()));
                }
            }
        }
    }
    let mut used_ids: BTreeSet<String> = seed_marked.keys().map(|(_, id)| id.clone()).collect();
    let mut kept: BTreeSet<(RefinementKind, String)> = BTreeSet::new();
    let mut edits = Vec::new();
    let edit = |action, kind, id: &str, section: Option<(&Section, &str)>| RefinementEdit {
        action: Some(action),
        kind: Some(kind),
        id: Some(id.to_string()),
        title: section.map(|(section, _)| section.title.clone()),
        content: section.map(|(section, _)| section.content.clone()),
        path: section.map(|(_, path)| path.to_string()),
        reference: None,
        arguments: None,
        metadata: section.map(|_| {
            serde_json::Map::from_iter([("origin".to_string(), serde_json::json!("harness2"))])
        }),
        reason: Some(reason.to_string()),
    };
    for (_, kind, path, sections) in read_tree(tree) {
        for section in &sections {
            if let Some(id) = &section.id {
                let key = (kind, id.clone());
                match seed_marked.get(&key) {
                    Some((before, before_path)) if before == section && *before_path == path => {}
                    Some(_) => {
                        edits.push(edit(
                            RefinementAction::Update,
                            kind,
                            id,
                            Some((section, path)),
                        ));
                    }
                    None => {
                        used_ids.insert(id.clone());
                        edits.push(edit(
                            RefinementAction::Create,
                            kind,
                            id,
                            Some((section, path)),
                        ));
                    }
                }
                kept.insert(key);
            } else {
                if seed_unmarked.contains(&(section.title.clone(), section.content.clone())) {
                    continue;
                }
                let prefix = match kind {
                    RefinementKind::Memory => "h2-mem",
                    RefinementKind::Prompt
                    | RefinementKind::Skill
                    | RefinementKind::Subagent
                    | RefinementKind::Factory => "h2-prompt",
                };
                let mut slug = String::new();
                for c in section.title.chars() {
                    if c.is_ascii_alphanumeric() {
                        slug.push(c.to_ascii_lowercase());
                    } else if !slug.ends_with('-') && !slug.is_empty() {
                        slug.push('-');
                    }
                }
                let slug: String = slug.trim_end_matches('-').chars().take(48).collect();
                let base = format!("{prefix}-{slug}");
                let mut id = base.clone();
                let mut n = 2;
                while !used_ids.insert(id.clone()) {
                    id = format!("{base}-{n}");
                    n += 1;
                }
                edits.push(edit(
                    RefinementAction::Create,
                    kind,
                    &id,
                    Some((section, path)),
                ));
            }
        }
    }
    for (kind, id) in seed_marked.keys() {
        if !kept.contains(&(*kind, id.clone())) {
            edits.push(edit(RefinementAction::Delete, *kind, id, None));
        }
    }
    RefinementProposal {
        summary: format!("harness2: {reason}"),
        rationale: format!(
            "Offline harness adaptation kept these edits after testing them on tasks ({reason})."
        ),
        expected_outcome: "The adapted harness scored higher than the seed on the benchmark tasks; /refine review records whether that holds in real sessions.".to_string(),
        edits,
    }
}

/// The file contents of one skill folder, by relative path.
fn skill_files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(current) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if let (Ok(relative), Ok(bytes)) = (path.strip_prefix(dir), std::fs::read(&path))
            {
                files.insert(relative.to_string_lossy().to_string(), bytes);
            }
        }
    }
    files
}

/// The names of the skill folders under a tree's `skills/`.
#[must_use]
pub fn skill_names(tree: &Path) -> BTreeSet<String> {
    std::fs::read_dir(tree.join(SKILLS_DIR))
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.path().join("SKILL.md").is_file())
                .map(|entry| entry.file_name().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// Folder-skill changes between two trees: `(written, deleted)`. A skill is
/// written when the tree's copy is new or differs from the seed's, and
/// deleted when it is one of `exported` (the agent's own) and the tree
/// dropped it. A benchmark's own skill that the tree dropped is not ours.
#[must_use]
pub fn skill_changes(
    seed: &Path,
    tree: &Path,
    exported: &BTreeSet<String>,
) -> (Vec<String>, Vec<String>) {
    let now = skill_names(tree);
    let written = now
        .iter()
        .filter(|name| {
            skill_files(&tree.join(SKILLS_DIR).join(name))
                != skill_files(&seed.join(SKILLS_DIR).join(name))
        })
        .cloned()
        .collect();
    let deleted = exported.difference(&now).cloned().collect();
    (written, deleted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refinement::executor::{apply_refinement_plan, RefineOptions, RefinementPlan};
    use crate::refinement::{empty_harness_state, HarnessEntry};

    fn entry(kind: &str, id: &str, title: &str, content: &str, path: &str) -> HarnessEntry {
        serde_json::from_value(serde_json::json!({
            "id": id, "kind": kind, "title": title, "content": content, "path": path,
            "scope": "global", "reference": {}, "arguments": {}, "metadata": {},
            "source": "refine", "version": 1, "created_at": "t", "updated_at": "t"
        }))
        .unwrap()
    }

    fn write_tree(dir: &Path, state: &HarnessState, base: &str) {
        std::fs::create_dir_all(dir.join("tool_descriptions")).unwrap();
        for (file, _, _) in ENTRY_FILES {
            std::fs::write(dir.join(file), base).unwrap();
        }
        for (file, sections) in export_sections(state) {
            let path = dir.join(file);
            let text = std::fs::read_to_string(&path).unwrap();
            std::fs::write(path, text + &sections).unwrap();
        }
    }

    #[test]
    fn an_unchanged_export_plans_nothing_and_an_edited_tree_plans_exactly_its_edits() {
        let mut state = empty_harness_state();
        for (kind, item) in [
            (
                RefinementKind::Memory,
                entry(
                    "memory",
                    "docker",
                    "Tests",
                    "Tests need docker.\n\n## not a heading of ours",
                    "general",
                ),
            ),
            (
                RefinementKind::Memory,
                entry("memory", "stale", "Stale", "Old fact.", "general"),
            ),
            (
                RefinementKind::Prompt,
                entry(
                    "prompt",
                    "tools",
                    "Tool use",
                    "Prefer rg.",
                    "harness2/tool_guidance",
                ),
            ),
        ] {
            state
                .entries
                .get_mut(&kind)
                .unwrap()
                .insert(item.id.clone(), item);
        }
        let dir = tempfile::tempdir().unwrap();
        let seed = dir.path().join("seed");
        let base = "<!-- a benchmark's\n     own comment -->\n";
        write_tree(&seed, &state, base);
        assert_eq!(plan_from_trees(&seed, &seed, "cycle 1").edits, Vec::new());

        // The optimiser rewrites one entry, drops one, and adds two notes of its own.
        let winner = dir.path().join("winner");
        write_tree(&winner, &state, base);
        let memory = std::fs::read_to_string(winner.join(MEMORY_FILE)).unwrap();
        let stale_at = memory.find("\n## Stale").unwrap();
        std::fs::write(
            winner.join(MEMORY_FILE),
            memory[..stale_at].replace("Tests need docker.", "Tests need the docker daemon.")
                + "\n## Flaky suite\nRetry the e2e suite once.\n",
        )
        .unwrap();
        std::fs::write(
            winner.join(GUARDRAILS_FILE),
            format!("{base}Never push from a rollout.\n"),
        )
        .unwrap();
        let proposal = plan_from_trees(&seed, &winner, "cycle 1: +0.4 on 5 tasks");
        let got: Vec<(RefinementAction, RefinementKind, String, Option<String>)> = proposal
            .edits
            .iter()
            .map(|edit| {
                (
                    edit.action.unwrap(),
                    edit.kind.unwrap(),
                    edit.id.clone().unwrap(),
                    edit.content.clone(),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (
                    RefinementAction::Update,
                    RefinementKind::Memory,
                    "docker".to_string(),
                    Some("Tests need the docker daemon.\n\n## not a heading of ours".to_string()),
                ),
                (
                    RefinementAction::Create,
                    RefinementKind::Memory,
                    "h2-mem-flaky-suite".to_string(),
                    Some("Retry the e2e suite once.".to_string()),
                ),
                (
                    RefinementAction::Create,
                    RefinementKind::Prompt,
                    "h2-prompt-never-push-from-a-rollout".to_string(),
                    Some("Never push from a rollout.".to_string()),
                ),
                (
                    RefinementAction::Delete,
                    RefinementKind::Memory,
                    "stale".to_string(),
                    None,
                ),
            ]
        );

        // Applied, the store exports back to the winner's entries: a second
        // plan against the new export is empty.
        let result = apply_refinement_plan(
            &mut state,
            RefinementPlan {
                proposal,
                id: "refine_1".to_string(),
                rollback_of: None,
                rollback_scope: None,
            },
            &RefineOptions {
                global: true,
                ..RefineOptions::default()
            },
            None,
            false,
        );
        assert!(result.applied_edits.iter().all(|edit| edit.applied));
        let reexport = dir.path().join("reexport");
        write_tree(&reexport, &state, base);
        let again = dir.path().join("again");
        write_tree(&again, &state, base);
        assert_eq!(
            plan_from_trees(&reexport, &again, "cycle 2").edits,
            Vec::new()
        );
        assert_eq!(
            state.entries[&RefinementKind::Prompt]["tools"].path,
            "harness2/tool_guidance"
        );
    }
}
