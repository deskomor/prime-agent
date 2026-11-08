//! Outcome verdicts on past refinements (fork-only): did an applied
//! refinement help, hurt, do nothing observable, or is its effect confounded
//! with other edits? The auto-refine review and `/refine review` record them;
//! the next `/refine` sees them in its history section, and a `hurt` verdict
//! surfaces a rollback notice.
//!
//! Verdicts live in their own stores so existing refinement rows keep their
//! shape for TS v0.9.8: the session custom row
//! [`REFINEMENT_VERDICT_CUSTOM_TYPE`], plus `harness/refinement_verdicts.jsonl`
//! for verdicts on global refinements.

use std::path::{Path, PathBuf};

use pa_types::session::AgentMessage;
use serde::{Deserialize, Serialize};

use super::RefinementResult;

/// Session custom row type carrying one verdict (TS ignores unknown types).
pub const REFINEMENT_VERDICT_CUSTOM_TYPE: &str = "prime-agent.refinement_verdict";
/// Global verdict log, next to `refinement_history.jsonl`.
pub const REFINEMENT_VERDICTS_FILE_NAME: &str = "refinement_verdicts.jsonl";
/// Turns a refinement must be behind the conversation's end before a review
/// judges it, so there is behaviour to judge.
pub const VERDICT_MIN_TURNS: u32 = 3;

/// What the trajectory after a refinement shows about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Helped,
    Hurt,
    NoEffect,
    Confounded,
}

impl Verdict {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Helped => "helped",
            Self::Hurt => "hurt",
            Self::NoEffect => "no_effect",
            Self::Confounded => "confounded",
        }
    }
}

/// One verdict on a refinement, or on one edit of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinementVerdict {
    pub refinement_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit_id: Option<String>,
    pub verdict: Verdict,
    #[serde(default)]
    pub evidence: String,
    #[serde(default)]
    pub created_at: String,
}

/// Parse a model's `verdicts` array leniently: items that are not objects,
/// lack a refinement id, or carry an unknown verdict are skipped.
#[must_use]
pub fn parse_verdicts(
    value: Option<&serde_json::Value>,
    created_at: &str,
) -> Vec<RefinementVerdict> {
    let Some(items) = value.and_then(serde_json::Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let refinement_id = item.get("refinementId")?.as_str()?.trim();
            if refinement_id.is_empty() {
                return None;
            }
            let verdict = serde_json::from_value::<Verdict>(item.get("verdict")?.clone()).ok()?;
            Some(RefinementVerdict {
                refinement_id: refinement_id.to_string(),
                edit_id: item
                    .get("editId")
                    .and_then(serde_json::Value::as_str)
                    .filter(|id| !id.trim().is_empty())
                    .map(str::to_string),
                verdict,
                evidence: item
                    .get("evidence")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                created_at: created_at.to_string(),
            })
        })
        .collect()
}

#[must_use]
pub fn get_refinement_verdicts_path(harness_state_dir: &Path) -> PathBuf {
    harness_state_dir.join(REFINEMENT_VERDICTS_FILE_NAME)
}

/// Append verdicts to the global verdict log.
///
/// # Errors
///
/// Returns an I/O or serialization error.
pub fn append_global_verdicts(
    harness_state_dir: &Path,
    verdicts: &[RefinementVerdict],
) -> anyhow::Result<()> {
    use std::io::Write;
    if verdicts.is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(harness_state_dir)?;
    let mut lines = String::new();
    for verdict in verdicts {
        lines.push_str(&serde_json::to_string(verdict)?);
        lines.push('\n');
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(get_refinement_verdicts_path(harness_state_dir))?
        .write_all(lines.as_bytes())?;
    Ok(())
}

/// Load the global verdict log; malformed lines are skipped.
#[must_use]
pub fn load_global_verdicts(harness_state_dir: &Path) -> Vec<RefinementVerdict> {
    std::fs::read_to_string(get_refinement_verdicts_path(harness_state_dir))
        .map(|content| {
            content
                .lines()
                .filter_map(|line| serde_json::from_str(line.trim()).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// The latest verdict per (refinement, edit), in first-seen order.
#[must_use]
pub fn latest_verdicts(verdicts: &[RefinementVerdict]) -> Vec<&RefinementVerdict> {
    let mut order: Vec<(&str, Option<&str>)> = Vec::new();
    let mut latest: std::collections::HashMap<(&str, Option<&str>), &RefinementVerdict> =
        std::collections::HashMap::new();
    for verdict in verdicts {
        let key = (verdict.refinement_id.as_str(), verdict.edit_id.as_deref());
        if latest.insert(key, verdict).is_none() {
            order.push(key);
        }
    }
    order.into_iter().map(|key| latest[&key]).collect()
}

/// Refinement ids whose latest verdict (on the whole refinement or any of its
/// edits) is `hurt`, in first-seen order.
#[must_use]
pub fn harmful_refinement_ids(verdicts: &[RefinementVerdict]) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    for verdict in latest_verdicts(verdicts) {
        if verdict.verdict == Verdict::Hurt && !ids.contains(&verdict.refinement_id) {
            ids.push(verdict.refinement_id.clone());
        }
    }
    ids
}

/// The prompt lines for the latest verdicts, e.g.
/// `refine_ab12 (edit m1): hurt -- turn 41: the new memory made it skip tests`.
#[must_use]
pub fn verdicts_for_prompt(verdicts: &[RefinementVerdict]) -> String {
    let latest = latest_verdicts(verdicts);
    if latest.is_empty() {
        return "No verdicts recorded yet.".to_string();
    }
    latest
        .into_iter()
        .map(|verdict| {
            let edit = verdict
                .edit_id
                .as_deref()
                .map(|id| format!(" (edit {id})"))
                .unwrap_or_default();
            let evidence = if verdict.evidence.trim().is_empty() {
                String::new()
            } else {
                format!(" -- {}", verdict.evidence.trim())
            };
            format!(
                "- {}{edit}: {}{evidence}",
                verdict.refinement_id,
                verdict.verdict.as_str()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Merge global and session verdicts into one list in `created_at` order, so
/// [`latest_verdicts`] keeps the newest verdict per (refinement, edit).
#[must_use]
pub fn merge_verdicts(
    global: &[RefinementVerdict],
    session: &[RefinementVerdict],
) -> Vec<RefinementVerdict> {
    let mut merged: Vec<RefinementVerdict> = global.iter().chain(session).cloned().collect();
    merged.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    merged
}

/// Epoch millis a `refine_<YYYYMMDDHHMMSSmmm>` id was minted at.
#[must_use]
pub fn refinement_id_millis(id: &str) -> Option<u64> {
    let digits = id.strip_prefix("refine_")?;
    if digits.len() < 14 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let millis = digits.get(14..17).unwrap_or("000");
    let iso = format!(
        "{}-{}-{}T{}:{}:{}.{millis:0<3}Z",
        &digits[0..4],
        &digits[4..6],
        &digits[6..8],
        &digits[8..10],
        &digits[10..12],
        &digits[12..14],
    );
    crate::cron::parse_iso_millis(&iso)
}

/// Refinements with an applied edit and at least [`VERDICT_MIN_TURNS`]
/// assistant turns after them in `messages`, with that turn count.
#[must_use]
pub fn judgeable_refinements<'a>(
    history: &'a [RefinementResult],
    messages: &[AgentMessage],
) -> Vec<(&'a str, u32)> {
    history
        .iter()
        .filter(|item| item.applied_edits.iter().any(|edit| edit.applied))
        .filter_map(|item| {
            let minted = refinement_id_millis(&item.id)?;
            let turns = messages
                .iter()
                .filter(|message| {
                    matches!(message, AgentMessage::Assistant(reply) if reply.timestamp > minted)
                })
                .count();
            let turns = u32::try_from(turns).unwrap_or(u32::MAX);
            (turns >= VERDICT_MIN_TURNS).then_some((item.id.as_str(), turns))
        })
        .collect()
}

/// The `<refinement_verdicts>` prompt section. `judgeable` is given only to
/// the review, which may add verdicts.
#[must_use]
pub fn verdicts_section(
    verdicts: &[RefinementVerdict],
    judgeable: Option<&[(&str, u32)]>,
) -> String {
    let mut body = format!("Recorded verdicts:\n{}", verdicts_for_prompt(verdicts));
    if let Some(judgeable) = judgeable {
        body.push_str("\n\nOld enough to judge (assistant turns since): ");
        if judgeable.is_empty() {
            body.push_str("none yet.");
        } else {
            body.push_str(
                &judgeable
                    .iter()
                    .map(|(id, turns)| format!("{id} ({turns})"))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
    }
    format!("<refinement_verdicts>\n{body}\n</refinement_verdicts>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsing_skips_malformed_items_and_latest_verdict_wins() {
        let raw = serde_json::json!([
            {"refinementId": "refine_a", "editId": "m1", "verdict": "helped", "evidence": "turn 9"},
            {"refinementId": "refine_b", "verdict": "sideways"},
            {"verdict": "hurt"},
            "not an object",
            {"refinementId": "refine_a", "editId": "m1", "verdict": "hurt", "evidence": "turn 41: skipped tests"},
            {"refinementId": "refine_c", "verdict": "no_effect"},
        ]);
        let verdicts = parse_verdicts(Some(&raw), "2026-11-08T00:00:00Z");
        assert_eq!(verdicts.len(), 3);
        assert_eq!(harmful_refinement_ids(&verdicts), vec!["refine_a"]);
        assert_eq!(
            verdicts_for_prompt(&verdicts),
            "- refine_a (edit m1): hurt -- turn 41: skipped tests\n- refine_c: no_effect"
        );
    }

    #[test]
    fn refinement_ids_decode_to_their_mint_time() {
        let now =
            crate::cron::parse_iso_millis(&crate::session::manager::format_iso_now()).unwrap();
        let minted =
            refinement_id_millis(&crate::refinement::executor::generate_refinement_id()).unwrap();
        assert!(minted.abs_diff(now) < 60_000);
        assert_eq!(
            refinement_id_millis("refine_20261108123456789"),
            crate::cron::parse_iso_millis("2026-11-08T12:34:56.789Z")
        );
        assert_eq!(refinement_id_millis("refine_abc"), None);
    }

    #[test]
    fn global_log_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let verdict = RefinementVerdict {
            refinement_id: "refine_g".to_string(),
            edit_id: None,
            verdict: Verdict::Confounded,
            evidence: "three edits landed together".to_string(),
            created_at: "t".to_string(),
        };
        append_global_verdicts(dir.path(), std::slice::from_ref(&verdict)).unwrap();
        append_global_verdicts(dir.path(), &[]).unwrap();
        assert_eq!(load_global_verdicts(dir.path()), vec![verdict]);
    }
}
