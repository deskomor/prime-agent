//! Admission-time host skill paths over the native discovery and expansion APIs.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::AgentSession;
use crate::skills::Skill;

#[derive(Default)]
pub(super) struct HostResourceSkills {
    skills: Vec<Skill>,
    owner: Option<(Arc<std::sync::atomic::AtomicU64>, u64)>,
    pub(super) prepared_inventory: Option<HostResourceInventory>,
}

#[derive(Clone, Default)]
pub(super) struct HostResourceInventory {
    pub(super) prompt: String,
    pub(super) owner: Option<(Arc<std::sync::atomic::AtomicU64>, u64)>,
}

impl AgentSession {
    pub(super) fn effective_skills(&self) -> Vec<Skill> {
        let mut skills = self.skills.clone();
        let hooks = self.host_hooks.read().expect("host hooks lock");
        let resources = self.host_resources.read().expect("host resources lock");
        if hooks
            .contract_v1
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst))
            && resources.owner.as_ref().is_some_and(|(epoch, expected)| {
                Arc::ptr_eq(epoch, &hooks.contract_epoch)
                    && epoch.load(Ordering::SeqCst) == *expected
            })
        {
            skills.extend(
                resources
                    .skills
                    .iter()
                    .filter(|skill| !self.skills.iter().any(|native| native.name == skill.name))
                    .cloned(),
            );
        }
        skills
    }

    /// Prepare resources before the worker records an expanded accepted skill row.
    pub(super) async fn prepare_host_resources_for_submission(&self) {
        if self.agent.state().await.is_streaming {
            return;
        }
        let hooks = self.host_hooks.read().expect("host hooks lock").clone();
        let prepared = self
            .host_resources
            .read()
            .expect("host resources lock")
            .prepared_inventory
            .as_ref()
            .is_some_and(|inventory| {
                hooks
                    .contract_v1
                    .as_ref()
                    .is_some_and(|flag| flag.load(Ordering::SeqCst))
                    && inventory.owner.as_ref().is_some_and(|(epoch, expected)| {
                        Arc::ptr_eq(epoch, &hooks.contract_epoch)
                            && epoch.load(Ordering::SeqCst) == *expected
                    })
            });
        if prepared {
            return;
        }
        let inventory = self.refresh_host_resources().await;
        self.host_resources
            .write()
            .expect("host resources lock")
            .prepared_inventory = Some(inventory);
    }

    pub(super) async fn refresh_host_resources(&self) -> HostResourceInventory {
        let hooks = self.host_hooks.read().expect("host hooks lock").clone();
        let had_resources = {
            let mut resources = self.host_resources.write().expect("host resources lock");
            let had_resources = !resources.skills.is_empty();
            *resources = HostResourceSkills::default();
            had_resources
        };
        let enabled = hooks
            .contract_v1
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst));
        let expected = hooks.contract_epoch.load(Ordering::SeqCst);
        if !enabled {
            return HostResourceInventory {
                prompt: String::new(),
                owner: had_resources.then(|| (Arc::clone(&hooks.contract_epoch), expected)),
            };
        }
        let owner = Some((Arc::clone(&hooks.contract_epoch), expected));
        let Some(hook) = hooks.resources_discover.as_ref() else {
            return HostResourceInventory {
                prompt: String::new(),
                owner,
            };
        };
        let cwd = self.session.lock().await.get_cwd().to_path_buf();
        let answer = hook(serde_json::json!({ "cwd": cwd })).await;
        let paths: Vec<String> = answer
            .as_ref()
            .and_then(|answer| answer.get("skillPaths"))
            .and_then(|paths| serde_json::from_value(paths.clone()).ok())
            .unwrap_or_default();
        let agent_dir = self.agent_dir.clone().unwrap_or_else(|| cwd.clone());
        let skills = if paths.is_empty() {
            Vec::new()
        } else {
            match tokio::task::spawn_blocking(move || {
                crate::skills::load_skills(&crate::skills::LoadSkillsOptions {
                    cwd,
                    agent_dir,
                    skill_paths: paths,
                    include_defaults: false,
                })
            })
            .await
            {
                Ok(result) => {
                    for diagnostic in result.diagnostics {
                        tracing::warn!(?diagnostic, "host skill discovery diagnostic");
                    }
                    result.skills
                }
                Err(error) => {
                    tracing::warn!(%error, "host skill discovery failed");
                    Vec::new()
                }
            }
        };
        let active = hooks
            .contract_v1
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst))
            && hooks.contract_epoch.load(Ordering::SeqCst) == expected;
        if !active {
            return HostResourceInventory {
                prompt: String::new(),
                owner,
            };
        }
        let skills: Vec<Skill> = skills
            .into_iter()
            .filter(|skill| !self.skills.iter().any(|native| native.name == skill.name))
            .collect();
        let prompt = crate::skills::format_skills_for_prompt(&skills);
        *self.host_resources.write().expect("host resources lock") = HostResourceSkills {
            skills,
            owner: owner.clone(),
            prepared_inventory: None,
        };
        HostResourceInventory { prompt, owner }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::manager::SessionManager;
    use crate::session_engine::host_hooks::SessionHostHooks;
    use crate::session_engine::PromptOptions;
    use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
    use pa_agent::scripted::ScriptedProvider;
    use pa_agent::types::{Message, UserContent, UserPart};
    use serde_json::json;

    #[tokio::test]
    async fn generated_skill_paths_feed_native_prompt_and_expansion_until_disabled() {
        let temp = tempfile::tempdir().unwrap();
        let library = temp.path().join("generated");
        let skill = library.join("generated-plan");
        std::fs::create_dir_all(&skill).unwrap();
        std::fs::write(skill.join("SKILL.md"),
            "---\nname: generated-plan\ndescription: A generated planning skill\n---\nGENERATED BODY"
        ).unwrap();
        let model = pa_agent::types::Model::unknown();
        let provider = Arc::new(ScriptedProvider::new(model.clone()));
        provider.push_text_turn("done");
        provider.push_text_turn("done without host skills");
        let agent = Arc::new(Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                system_prompt: Some("NATIVE-BASE".into()),
                model: Some(model),
                ..Default::default()
            },
            stream_fn: Some(provider.stream_fn()),
            ..Default::default()
        }));
        let session = AgentSession::new(agent, SessionManager::in_memory(temp.path()), vec![])
            .await
            .unwrap();
        let enabled = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let expected_cwd = temp.path().to_path_buf();
        let discoveries = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count_discoveries = Arc::clone(&discoveries);
        session.set_host_hooks(SessionHostHooks {
            contract_v1: Some(Arc::clone(&enabled)),
            resources_discover: Some(Arc::new(move |payload| {
                let library = library.clone();
                let expected_cwd = expected_cwd.clone();
                let count_discoveries = Arc::clone(&count_discoveries);
                Box::pin(async move {
                    count_discoveries.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(payload, json!({"cwd":expected_cwd}));
                    Some(json!({"skillPaths":[library]}))
                })
            })),
            ..Default::default()
        });
        session.prepare_host_resources_for_submission().await;
        session.prepare_host_resources_for_submission().await;
        let accepted = crate::skills::expand_skill_command(
            "/skill:generated-plan details",
            &session.effective_skills(),
        )
        .0;
        let accepted_row: pa_types::session::AgentMessage = serde_json::from_value(json!({
            "role":"user", "content":accepted, "timestamp":1,
        }))
        .unwrap();
        let accepted_id = {
            let mut manager = session.session.lock().await;
            let id = manager.append_message(accepted_row.clone()).unwrap();
            manager.replay_admitted_user_rows(vec![id.clone()]).unwrap();
            id
        };
        session
            .prompt(&accepted, PromptOptions::default())
            .await
            .unwrap();
        assert_eq!(discoveries.load(Ordering::SeqCst), 1);
        let entries = session.session.lock().await.get_entries();
        let saved = entries
            .iter()
            .find(|entry| entry.id() == Some(accepted_id.as_str()))
            .unwrap();
        let pa_types::session::FileEntry::Message { message, .. } = saved else {
            panic!("accepted row must remain a message")
        };
        assert_eq!(message, &accepted_row);
        let calls = provider.calls();
        let prompt = calls[0].system_prompt.as_deref().unwrap();
        assert!(prompt.starts_with("NATIVE-BASE"));
        assert!(prompt.contains("<name>generated-plan</name>"));
        let text = |content: &UserContent| match content {
            UserContent::Text(text) => text.clone(),
            UserContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    UserPart::Text(text) => Some(text.text.as_str()),
                    UserPart::Image(_) => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        };
        let Message::User(first) = calls[0].messages.last().unwrap() else {
            panic!("expected user prompt")
        };
        let expanded = text(&first.content);
        assert_eq!(expanded, accepted);
        assert!(expanded.contains("GENERATED BODY"));
        assert!(expanded.contains("<skill name=\"generated-plan\""));
        enabled.store(false, Ordering::SeqCst);
        assert_eq!(
            crate::skills::expand_skill_command(
                "/skill:generated-plan",
                &session.effective_skills()
            )
            .0,
            "/skill:generated-plan"
        );
        session
            .prompt("/skill:generated-plan", PromptOptions::default())
            .await
            .unwrap();
        let calls = provider.calls();
        assert_eq!(calls[1].system_prompt.as_deref(), Some("NATIVE-BASE"));
        let Message::User(second) = calls[1].messages.last().unwrap() else {
            panic!("expected user prompt")
        };
        assert_eq!(text(&second.content), "/skill:generated-plan");
    }
}
