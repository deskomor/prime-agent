//! `/refine --probe` (fork-only): test a refinement proposal before keeping
//! it, as Harness² does. The same task runs twice, once on the current
//! harness (base) and once with the proposal applied (candidate); a compose
//! call keeps only the edits the candidate run supports. The runs themselves
//! live in the session layer; this module holds the pure parts: picking the
//! task, seeding the two harnesses, the compose prompt and its answer.

use pa_types::ai::{AssistantContentBlock, UserContent, UserContentBlock};
use pa_types::session::AgentMessage;

use super::executor::RefinementPlan;
use super::planner::{apply_refinement_proposal, extract_json_object, ApplyOptions};
use super::verdicts::{RefinementVerdict, Verdict};
use super::{HarnessScope, HarnessState};

/// Default wall-clock budget for one probe run.
pub const DEFAULT_PROBE_TIMEOUT_MS: u64 = 900_000;
/// Cap on each run's trajectory and diff in the compose prompt.
pub const PROBE_EVIDENCE_CHARS: usize = 20_000;

/// The task to replay: the latest user prompt that started real work, meaning
/// one that the assistant answered with at least one tool call.
#[must_use]
pub fn probe_task(messages: &[AgentMessage]) -> Option<String> {
    let mut current: Option<String> = None;
    let mut task: Option<String> = None;
    for message in messages {
        match message {
            AgentMessage::User(user) => {
                let text = user_text(&user.content);
                current = (!text.trim().is_empty()).then_some(text);
            }
            AgentMessage::Assistant(reply)
                if reply
                    .content
                    .iter()
                    .any(|block| matches!(block, AssistantContentBlock::ToolCall(_))) =>
            {
                if let Some(text) = &current {
                    task = Some(text.clone());
                }
            }
            _ => {}
        }
    }
    task
}

fn user_text(content: &UserContent) -> String {
    match content {
        UserContent::Text(text) => text.clone(),
        UserContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                UserContentBlock::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// The local harness stores the two probe runs start from (each run also
/// reads the shared global store): the base is the session's local state,
/// the candidate is that state with the proposal applied. An update or
/// delete aimed at a global entry finds nothing to change here, so the
/// candidate run cannot try it.
#[must_use]
pub fn seed_states(
    local_state: &HarnessState,
    plan: &RefinementPlan,
    factory_enabled: bool,
) -> (HarnessState, HarnessState) {
    let base = local_state.clone();
    let mut candidate = base.clone();
    apply_refinement_proposal(
        &mut candidate,
        &plan.proposal,
        ApplyOptions {
            id: plan.id.clone(),
            rollback_of: None,
            scope: Some(HarnessScope::Local),
            baseline_state: None,
            factory_enabled,
        },
    );
    (base, candidate)
}

/// What one probe run produced.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProbeRun {
    pub final_answer: Option<String>,
    /// The run's conversation, as the refiner reads one.
    pub trajectory: String,
    /// `git diff --stat` and the diff of the run's worktree.
    pub diff: String,
    pub error: Option<String>,
}

impl ProbeRun {
    /// A run delivered something when it answered or changed files.
    #[must_use]
    pub fn delivered(&self) -> bool {
        self.final_answer
            .as_deref()
            .is_some_and(|answer| !answer.trim().is_empty())
            || !self.diff.trim().is_empty()
    }

    fn for_prompt(&self, tag: &str) -> String {
        let cap = |text: &str| -> String {
            let chars: Vec<char> = text.chars().collect();
            if chars.len() <= PROBE_EVIDENCE_CHARS {
                text.to_string()
            } else {
                chars[chars.len() - PROBE_EVIDENCE_CHARS..].iter().collect()
            }
        };
        format!(
            "<{tag} delivered=\"{}\">\n<final_answer>\n{}\n</final_answer>\n<error>\n{}\n</error>\n<diff>\n{}\n</diff>\n<trajectory>\n{}\n</trajectory>\n</{tag}>",
            self.delivered(),
            self.final_answer.as_deref().unwrap_or("(none)"),
            self.error.as_deref().unwrap_or("(none)"),
            cap(&self.diff),
            cap(&self.trajectory),
        )
    }
}

/// The compose call's user prompt.
#[must_use]
pub fn compose_prompt(
    task: &str,
    plan: &RefinementPlan,
    base: &ProbeRun,
    candidate: &ProbeRun,
) -> String {
    let edits = plan
        .proposal
        .edits
        .iter()
        .enumerate()
        .map(|(index, edit)| {
            format!(
                "edit {index}: {}",
                serde_json::to_string(edit).unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    [
        format!("<task>\n{task}\n</task>"),
        format!(
            "<proposal>\nsummary: {}\nexpected outcome: {}\n{edits}\n</proposal>",
            plan.proposal.summary, plan.proposal.expected_outcome
        ),
        base.for_prompt("base_run"),
        candidate.for_prompt("candidate_run"),
        "Return only the JSON object. Keeping nothing is a valid answer.".to_string(),
    ]
    .join("\n\n")
}

/// The compose call's decision.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ComposeDecision {
    /// Kept edit indexes, ascending.
    pub keep: Vec<usize>,
    pub notes: Vec<String>,
    pub verdicts: Vec<(usize, Verdict, String)>,
}

/// Parse the compose answer. Indexes outside the proposal are ignored, and
/// when the candidate run delivered nothing every edit is dropped whatever
/// the answer says.
///
/// # Errors
///
/// Returns an error when the answer holds no JSON object.
pub fn parse_compose(
    text: &str,
    edit_count: usize,
    candidate_delivered: bool,
) -> anyhow::Result<ComposeDecision> {
    let value = extract_json_object(text).map_err(anyhow::Error::msg)?;
    let index = |item: &serde_json::Value| {
        item.as_u64()
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| *index < edit_count)
    };
    let mut keep: Vec<usize> = value
        .get("keep")
        .and_then(serde_json::Value::as_array)
        .map(|items| items.iter().filter_map(index).collect())
        .unwrap_or_default();
    keep.sort_unstable();
    keep.dedup();
    let mut notes: Vec<String> = value
        .get("notes")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if !candidate_delivered && !keep.is_empty() {
        keep.clear();
        notes.push("dropped -> all edits -> the candidate run delivered nothing".to_string());
    }
    let verdicts = value
        .get("verdicts")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let edit = index(item.get("editIndex")?)?;
                    let verdict =
                        serde_json::from_value::<Verdict>(item.get("verdict")?.clone()).ok()?;
                    let evidence = item
                        .get("evidence")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    Some((edit, verdict, evidence))
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(ComposeDecision {
        keep,
        notes,
        verdicts,
    })
}

/// The plan reduced to the kept edits.
#[must_use]
pub fn kept_plan(mut plan: RefinementPlan, decision: &ComposeDecision) -> RefinementPlan {
    plan.proposal.edits = plan
        .proposal
        .edits
        .into_iter()
        .enumerate()
        .filter(|(index, _)| decision.keep.contains(index))
        .map(|(_, edit)| edit)
        .collect();
    plan
}

/// The probe's per-edit verdicts as outcome verdicts on refinement
/// `refinement_id`. An edit without an id is named `edit<N>`.
#[must_use]
pub fn probe_verdicts(
    decision: &ComposeDecision,
    plan: &RefinementPlan,
    refinement_id: &str,
    created_at: &str,
) -> Vec<RefinementVerdict> {
    decision
        .verdicts
        .iter()
        .map(|(index, verdict, evidence)| RefinementVerdict {
            refinement_id: refinement_id.to_string(),
            edit_id: Some(
                plan.proposal
                    .edits
                    .get(*index)
                    .and_then(|edit| edit.id.clone())
                    .unwrap_or_else(|| format!("edit{index}")),
            ),
            verdict: *verdict,
            evidence: format!("probe: {evidence}"),
            created_at: created_at.to_string(),
        })
        .collect()
}

/// The probe line added to the refinement notice.
#[must_use]
pub fn probe_notice(decision: &ComposeDecision, edit_count: usize) -> String {
    let dropped: Vec<usize> = (0..edit_count)
        .filter(|index| !decision.keep.contains(index))
        .collect();
    let mut text = format!(
        "probe: kept {:?}, dropped {dropped:?} of {edit_count} proposed edits",
        decision.keep
    );
    for note in &decision.notes {
        text.push_str("\n- ");
        text.push_str(note);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refinement::planner::{RefinementEdit, RefinementProposal};
    use crate::refinement::{empty_harness_state, RefinementAction, RefinementKind};

    fn plan(ids: &[&str]) -> RefinementPlan {
        RefinementPlan {
            proposal: RefinementProposal {
                summary: "s".to_string(),
                rationale: "r".to_string(),
                expected_outcome: "e".to_string(),
                edits: ids
                    .iter()
                    .map(|id| RefinementEdit {
                        action: Some(RefinementAction::Create),
                        kind: Some(RefinementKind::Memory),
                        id: Some((*id).to_string()),
                        title: Some(format!("T {id}")),
                        content: Some(format!("C {id}")),
                        reason: Some("turn 2".to_string()),
                        ..RefinementEdit::default()
                    })
                    .collect(),
            },
            id: "refine_20261108000000000".to_string(),
            rollback_of: None,
            rollback_scope: None,
        }
    }

    #[test]
    fn the_task_is_the_latest_prompt_that_led_to_a_tool_call() {
        use pa_types::ai::{AssistantMessage, TextContent, ToolCall, UserMessage};
        let user = |text: &str| {
            AgentMessage::User(UserMessage {
                content: UserContent::Text(text.to_string()),
                timestamp: 0,
                rest: serde_json::Map::default(),
            })
        };
        let assistant = |block: AssistantContentBlock| {
            let mut reply: AssistantMessage = serde_json::from_value(serde_json::json!({
                "role": "assistant", "content": [], "api": "a", "provider": "p", "model": "m",
                "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                          "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
                "stopReason": "stop", "timestamp": 0
            }))
            .unwrap();
            reply.content = vec![block];
            AgentMessage::Assistant(reply)
        };
        let tool = || {
            AssistantContentBlock::ToolCall(ToolCall {
                id: "c1".to_string(),
                name: "bash".to_string(),
                arguments: serde_json::Map::default(),
                thought_signature: None,
                rest: serde_json::Map::default(),
            })
        };
        let text = AssistantContentBlock::Text(TextContent {
            text: "hi".to_string(),
            text_signature: None,
            rest: serde_json::Map::default(),
        });
        let messages = vec![
            user("fix the build"),
            assistant(tool()),
            user("thanks"),
            assistant(text),
        ];
        assert_eq!(probe_task(&messages).as_deref(), Some("fix the build"));
        assert_eq!(probe_task(&messages[2..]), None);
    }

    #[test]
    fn the_candidate_harness_carries_the_proposal_and_the_base_does_not() {
        let (base, candidate) = seed_states(&empty_harness_state(), &plan(&["m1"]), false);
        assert!(base.entries[&RefinementKind::Memory].is_empty());
        let entry = &candidate.entries[&RefinementKind::Memory]["m1"];
        assert_eq!(entry.scope, Some(HarnessScope::Local));
    }

    #[test]
    fn compose_keeps_listed_edits_and_drops_all_when_the_candidate_delivered_nothing() {
        let answer = r#"{"keep":[1,1,7],"notes":["kept -> edit 1 -> turn 3 ran the tests"],"verdicts":[{"editIndex":1,"verdict":"helped","evidence":"turn 3"},{"editIndex":0,"verdict":"no_effect"},{"editIndex":9,"verdict":"hurt"}]}"#;
        let decision = parse_compose(answer, 2, true).unwrap();
        assert_eq!(decision.keep, vec![1]);
        assert_eq!(decision.verdicts.len(), 2);
        let kept = kept_plan(plan(&["m1", "m2"]), &decision);
        assert_eq!(kept.proposal.edits[0].id.as_deref(), Some("m2"));
        let verdicts = probe_verdicts(&decision, &plan(&["m1", "m2"]), "refine_x", "t");
        assert_eq!(verdicts[0].edit_id.as_deref(), Some("m2"));
        assert_eq!(
            probe_notice(&decision, 2),
            "probe: kept [1], dropped [0] of 2 proposed edits\n- kept -> edit 1 -> turn 3 ran the tests"
        );
        assert!(parse_compose(answer, 2, false).unwrap().keep.is_empty());
    }
}
