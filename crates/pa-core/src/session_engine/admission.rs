use super::slash_commands::parse_session_command;
use super::{
    session_message_to_loop, user_prompt_message, AgentSession, PromptOptions, PromptOutcome,
    SessionAgentMessage, SessionSlashCommand, StreamingBehavior,
};

impl AgentSession {
    /// Submit a prompt. Session commands (compact/refine/goal/autonomous)
    /// are recognized before admission and never reach the model.
    ///
    /// # Errors
    ///
    /// The underlying prompt admission error (see [`AgentSession::prompt_with_images`]).
    pub async fn prompt(
        &self,
        text: &str,
        options: PromptOptions,
    ) -> anyhow::Result<PromptOutcome> {
        self.prompt_with_images(text, Vec::new(), options).await
    }

    /// Admit an injected custom message as the turn's prompt (TS
    /// `_promptInjectedMessage`): the loop context and transcript hold ONE
    /// representation — the custom row itself — while the provider request
    /// carries its user-role view; never template-expanded or command-parsed.
    ///
    /// # Errors
    ///
    /// Error when the session is busy, the digest capture fails, or the
    /// agent rejects the prompt.
    pub async fn prompt_injected_message(
        &self,
        message: &pa_types::session::CustomMessage,
    ) -> anyhow::Result<PromptOutcome> {
        let state = self.agent.state().await;
        let busy = state.is_streaming;
        if busy {
            anyhow::bail!(
                "Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to queue the message."
            );
        }
        let mut prompt_messages = Vec::new();
        if let Some(digest_row) = self.pending_digest_prompt_row().await? {
            prompt_messages.push(digest_row);
        }
        prompt_messages.extend(self.take_next_turn_rows().await);
        let custom_row = session_message_to_loop(&SessionAgentMessage::Custom(message.clone()))
            .ok_or_else(|| anyhow::anyhow!("injected custom message conversion failed"))?;
        // The dispatch-time routing fires for every dispatched turn; injected
        // rows do not re-run the route decision (a goal objective start's
        // images ride the row's content and the row runs on the session's
        // model), so a route left by the previous turn clears here.
        self.apply_image_model_routing(&[], &[]).await?;
        prompt_messages.push(custom_row);
        self.agent
            .prompt(pa_agent::agent::AgentPromptInput::Messages(prompt_messages))
            .await?;
        Ok(PromptOutcome::Prompt)
    }

    /// Classify a prompt as a session command without admitting it (the same
    /// expansion-plus-grammar parse `prompt` applies). Pre-turn compaction
    /// arms stay off the session-command path (TS never reaches `_prepareForCommit`).
    pub fn classify_session_command(&self, text: &str) -> Option<SessionSlashCommand> {
        let normalized = crate::skills::expand_prompt_template(text, &self.prompt_templates);
        parse_session_command(&self.slash_commands, &normalized)
    }

    /// Prompt with images attached (the ACP prompt-capability path). Busy
    /// sessions queue the text and images together as one follow-up batch,
    /// so an admitted prompt never loses its images to a queue race.
    ///
    /// # Errors
    ///
    /// Error when the prompt fails validation, the session is busy, or the
    /// agent rejects the turn.
    ///
    /// # Panics
    ///
    /// Panics if the host resources or host hooks lock is poisoned.
    pub async fn prompt_with_images(
        &self,
        text: &str,
        images: Vec<pa_agent::types::ImageContent>,
        options: PromptOptions,
    ) -> anyhow::Result<PromptOutcome> {
        let busy = self.agent.state().await.is_streaming;
        let host_inventory = if !busy && self.classify_session_command(text).is_none() {
            let prepared = self
                .host_resources
                .write()
                .expect("host resources lock")
                .prepared_inventory
                .take();
            let hooks = self.host_hooks.read().expect("host hooks lock").clone();
            match prepared.filter(|inventory| {
                hooks
                    .contract_v1
                    .as_ref()
                    .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::SeqCst))
                    && inventory.owner.as_ref().is_some_and(|(epoch, expected)| {
                        std::sync::Arc::ptr_eq(epoch, &hooks.contract_epoch)
                            && epoch.load(std::sync::atomic::Ordering::SeqCst) == *expected
                    })
            }) {
                Some(inventory) => inventory,
                None => self.refresh_host_resources().await,
            }
        } else {
            super::host_resources::HostResourceInventory::default()
        };
        let skills = self.effective_skills();
        let expand =
            options.restricted.is_none() && options.expand_prompt_templates.unwrap_or(true);
        // TS `_finishSubmissionNormalization` order: skill commands expand first, prompt templates
        // second; both are gated by the same policy flag.
        let (normalized, used_skill) = if expand {
            let (skill_expanded, used_skill) = crate::skills::expand_skill_command(text, &skills);
            let normalized =
                crate::skills::expand_prompt_template(&skill_expanded, &self.prompt_templates);
            (normalized, used_skill)
        } else {
            (text.to_string(), None)
        };

        if let Some(command) = parse_session_command(&self.slash_commands, &normalized) {
            anyhow::ensure!(
                options.restricted.is_none(),
                "Plan mode does not execute session commands"
            );
            return Ok(PromptOutcome::SessionCommand(command));
        }

        // The `skill_use_count` session counter counts at the admission
        // seam: an admitted user turn whose text IS a skill block counts
        // once. A pre-expanded block (the daemon emits the accepted row
        // before admission) counts here too — the block parse carries the
        // skill identity.
        let skill_used = used_skill.is_some()
            || pa_types::skill_blocks::parse_skill_block(&normalized)
                .is_some_and(|block| skills.iter().any(|skill| skill.name == block.name));
        if skill_used {
            if let Some(telemetry) = &self.skill_telemetry {
                telemetry.note_skill_used();
            }
        }
        if busy && (options.restricted.is_some() || options.streaming_behavior.is_none()) {
            anyhow::bail!(
                "Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to queue the message."
            );
        }
        // User messages persist through the loop's `message_end` event
        // (matching TS `_processAgentEvent`); appending here too would
        // double-persist.

        if busy {
            let message = user_prompt_message(&normalized, &images);
            match options.streaming_behavior {
                Some(StreamingBehavior::Steer) => self.agent.steer(message),
                Some(StreamingBehavior::FollowUp) => self.agent.follow_up(message),
                None => unreachable!("busy without a streaming behavior errors above"),
            }
        } else {
            // The dispatch-time image-model routing decision: an
            // image-attaching batch routes to the host's configured image
            // model or fails with the actionable refusal, never silently
            // downgrading the images to placeholders.
            self.apply_image_model_routing(&images, &options.batch)
                .await?;
            let inventory = host_inventory
                .owner
                .as_ref()
                .filter(|(epoch, expected)| {
                    epoch.load(std::sync::atomic::Ordering::SeqCst) == *expected
                        && self
                            .host_hook(|hooks| hooks.contract_v1.clone())
                            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::SeqCst))
                })
                .map_or("", |_| host_inventory.prompt.as_str());
            let mut start_rows = Vec::new();
            let mut start_owner = None;
            // The host's `before_agent_start` sees native and discovered skill inventory.
            if let Some(hook) = self.host_hook(|hooks| hooks.before_agent_start.clone()) {
                let owner = self
                    .host_hook(|hooks| hooks.contract_v1.clone())
                    .filter(|flag| flag.load(std::sync::atomic::Ordering::SeqCst))
                    .and_then(|_| {
                        self.host_hook(|hooks| Some(std::sync::Arc::clone(&hooks.contract_epoch)))
                    })
                    .map(|epoch| {
                        let expected = epoch.load(std::sync::atomic::Ordering::SeqCst);
                        (epoch, expected)
                    });
                let native_prompt = self.agent.state().await.system_prompt;
                let system_prompt = format!("{native_prompt}{inventory}");
                let answer = hook(serde_json::json!({
                    "prompt": normalized,
                    "systemPrompt": system_prompt,
                    "imageCount": images.len(),
                }))
                .await;
                let append = answer
                    .as_ref()
                    .and_then(|answer| answer.get("appendSystemPrompt"))
                    .and_then(serde_json::Value::as_str)
                    .filter(|text| !text.trim().is_empty())
                    .map(str::to_string);
                // Rows this hook returns ride this admission only, after its
                // user rows (TS `_appendBeforeAgentStartMessages` order).
                // Only a v1 host adds rows; its epoch is rechecked at delivery.
                if let Some((epoch, expected)) = owner.as_ref() {
                    start_rows = answer
                        .as_ref()
                        .map(super::host_hooks::admission_messages)
                        .unwrap_or_default();
                    start_owner = Some((std::sync::Arc::clone(epoch), *expected));
                }
                let replacement = owner
                    .is_some()
                    .then_some(answer.as_ref())
                    .flatten()
                    .and_then(|answer| answer.get("systemPrompt"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
                let append = if replacement.is_none() {
                    let combined = match append {
                        Some(append) if inventory.is_empty() => append,
                        Some(append) => format!("{inventory}\n\n{append}"),
                        None => inventory.to_string(),
                    };
                    (!combined.trim().is_empty()).then_some(combined)
                } else {
                    None
                };
                self.agent.set_host_prompt(replacement, append, owner).await;
            } else if host_inventory.owner.is_some() {
                self.agent
                    .set_host_prompt(
                        None,
                        (!inventory.trim().is_empty()).then(|| inventory.to_string()),
                        host_inventory.owner,
                    )
                    .await;
            }
            // The turn's prompt messages: the deferred first-turn harness
            // digest rides first when one is due, streamed ahead of the user
            // prompt and carried on `agent_end`.
            let mut prompt_messages = Vec::new();
            if let Some(digest_row) = self.pending_digest_prompt_row().await? {
                prompt_messages.push(digest_row);
            }
            prompt_messages.extend(self.take_next_turn_rows().await);
            prompt_messages.push(user_prompt_message(&normalized, &images));
            // Each batched action contributes its user row after the
            // primary, through the same admission normalization (TS
            // normalizes each submission at queue time; this engine
            // normalizes every row at the shared admission).
            for row in &options.batch {
                let row_text = if expand {
                    let (skill_expanded, _) =
                        crate::skills::expand_skill_command(&row.text, &skills);
                    crate::skills::expand_prompt_template(&skill_expanded, &self.prompt_templates)
                } else {
                    row.text.clone()
                };
                prompt_messages.push(user_prompt_message(&row_text, &row.images));
            }
            // A host that detached during the hook or digest await adds nothing.
            if start_owner.is_some_and(|(epoch, expected)| {
                epoch.load(std::sync::atomic::Ordering::SeqCst) == expected
            }) {
                prompt_messages.extend(
                    start_rows.into_iter().filter_map(|row| {
                        session_message_to_loop(&SessionAgentMessage::Custom(row))
                    }),
                );
            }
            if let Some(restricted) = options.restricted {
                self.agent
                    .prompt_restricted(prompt_messages, restricted)
                    .await?;
            } else if options.return_after_accepted {
                // `returnAfterAccepted`: this prompt never started, so its
                // route must not survive (the failure path unwinds it like
                // the plain prompt branch below).
                if let Err(error) = self
                    .agent
                    .prompt_until_accepted(pa_agent::agent::AgentPromptInput::Messages(
                        prompt_messages,
                    ))
                    .await
                {
                    if !self.agent.state().await.is_streaming {
                        if let Some(router) = self.image_model_router.as_ref() {
                            (router.swap_target)(None);
                            self.agent.set_model_override(None);
                        }
                    }
                    return Err(error);
                }
            } else if let Err(error) = self
                .agent
                .prompt(pa_agent::agent::AgentPromptInput::Messages(prompt_messages))
                .await
            {
                // A concurrent admission won the agent's run slot: this
                // prompt never started, so its route must not survive. The
                // unwind only happens while NO run streams (the winner keeps
                // its target); TS's single-threaded commit cannot see this race.
                if !self.agent.state().await.is_streaming {
                    if let Some(router) = self.image_model_router.as_ref() {
                        (router.swap_target)(None);
                        self.agent.set_model_override(None);
                    }
                }
                return Err(error);
            }
        }
        Ok(PromptOutcome::Prompt)
    }

    /// Queue one custom row for the next admitted turn: the row rides the
    /// turn's prompt messages ahead of the prompt's own user row.
    pub fn queue_next_turn_row(&self, message: pa_types::session::CustomMessage) {
        self.pending_next_turn_rows
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(message);
    }

    /// Adopt a shared next-turn mailbox (the engine's restore-notice seam):
    /// rows a kernel boot parked before the session existed merge in, and
    /// later pushes land in the same queue: the kernel provisioner outlives
    /// the construction order, so the mailbox must cross the build boundary.
    pub fn adopt_next_turn_rows(
        &mut self,
        shared: std::sync::Arc<std::sync::Mutex<Vec<pa_types::session::CustomMessage>>>,
    ) {
        let own_rows: Vec<_> = {
            let mut own = self
                .pending_next_turn_rows
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            own.drain(..).collect()
        };
        {
            let mut next = shared
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // The shared mailbox is authoritative: rows parked pre-build
            // (a restore that finished during construction) come first,
            // then anything this session queued before adoption.
            next.extend(own_rows);
        }
        self.pending_next_turn_rows = shared;
    }

    /// Drain the queued next-turn rows: the admitting turn owns them; an
    /// empty take leaves nothing for later turns.
    pub fn take_next_turn_rows(
        &self,
    ) -> impl std::future::Future<Output = Vec<pa_agent::types::AgentMessage>> {
        std::future::ready(
            self.pending_next_turn_rows
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .drain(..)
                .filter_map(|row| session_message_to_loop(&SessionAgentMessage::Custom(row)))
                .collect(),
        )
    }
}
