//! Reentrant host writes use the mirrored live agent, not the core-session lock.

use super::{AgentSessionEngine, Arc, Value};

impl AgentSessionEngine {
    async fn host_agent(&self) -> anyhow::Result<Arc<pa_agent::agent::Agent>> {
        let existing = self.turn_agent.lock().expect("turn agent lock").clone();
        if let Some(agent) = existing {
            return Ok(agent);
        }
        let model = self.session_model()?;
        self.ensure_core_session_async(&model).await?;
        self.turn_agent
            .lock()
            .expect("turn agent lock")
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Session agent is not available"))
    }

    /// Current model-visible tools, including their native schemas.
    pub(crate) async fn host_tool_definitions(&self) -> Vec<Value> {
        let agent = match self.host_agent().await {
            Ok(agent) => agent,
            Err(error) => {
                eprintln!("pa-daemon: cannot read host tool definitions: {error:#}");
                return Vec::new();
            }
        };
        agent
            .state()
            .await
            .tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters(),
                })
            })
            .collect()
    }

    /// Select tools from the complete native inventory, including after narrowing.
    pub(crate) async fn set_host_active_tools(&self, names: Vec<String>) -> anyhow::Result<()> {
        let epoch = self
            .host_hooks
            .lock()
            .expect("host hooks lock")
            .contract_epoch
            .clone();
        let expected_epoch = epoch.load(std::sync::atomic::Ordering::SeqCst);
        self.host_agent().await?;
        let mut selection = self.host_active_tools.lock().await;
        let owner_active = self
            .host_hooks
            .lock()
            .expect("host hooks lock")
            .contract_v1
            .as_ref()
            .is_some_and(|owner| owner.load(std::sync::atomic::Ordering::SeqCst));
        anyhow::ensure!(
            owner_active && epoch.load(std::sync::atomic::Ordering::SeqCst) == expected_epoch,
            "Host adapter is no longer attached"
        );
        let agent = self
            .turn_agent
            .lock()
            .expect("turn agent lock")
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Session agent is not available"))?;
        let inventory = self
            .host_tool_inventory
            .lock()
            .expect("host tool inventory lock")
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Session tool inventory is not available"))?;
        for name in &names {
            anyhow::ensure!(
                inventory.iter().any(|tool| tool.name() == name),
                "Unknown active tool: {name}"
            );
        }
        let tools = inventory
            .into_iter()
            .filter(|tool| names.iter().any(|name| name == tool.name()))
            .collect();
        agent.set_tools(tools).await;
        *selection = Some(names);
        Ok(())
    }

    /// Detach removes only the host selection and restores the native tools.
    pub(crate) async fn clear_host_active_tools(&self) -> anyhow::Result<()> {
        let mut selection = self.host_active_tools.lock().await;
        let agent = self.turn_agent.lock().expect("turn agent lock").clone();
        let inventory = self
            .host_tool_inventory
            .lock()
            .expect("host tool inventory lock")
            .clone();
        if let (Some(agent), Some(inventory)) = (agent, inventory) {
            agent.set_tools(inventory).await;
        }
        *selection = None;
        Ok(())
    }

    /// Remove host prompt and tool adaptations without changing native defaults.
    pub(crate) async fn clear_host_overrides(&self) -> anyhow::Result<()> {
        self.clear_host_active_tools().await?;
        let agent = self.turn_agent.lock().expect("turn agent lock").clone();
        if let Some(agent) = agent {
            agent.set_system_prompt_override(None).await;
            agent.set_system_prompt_suffix(None).await;
        }
        Ok(())
    }

    /// Add one canonical custom row to live context; the worker owns persistence.
    pub(crate) async fn inject_host_custom_message(&self, row: Value) -> anyhow::Result<()> {
        anyhow::ensure!(
            row.get("role").and_then(Value::as_str) == Some("custom"),
            "Host context injection requires a custom message"
        );
        let entry_id = row
            .get("entryId")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("Host context message requires a saved entryId"))?
            .to_string();
        let message: pa_agent::types::CustomAgentMessage = serde_json::from_value(row)?;
        let agent = self.host_agent().await?;
        let handles = self.goal_runtime.lock().expect("goal runtime lock").clone();
        if let Some(handles) = handles {
            // A lazy build may already have restored the worker's newly saved row.
            if handles
                .session
                .lock()
                .await
                .get_entry_by_id(&entry_id)
                .is_some()
            {
                return Ok(());
            }
        }
        agent.inject_host_message(message).await
    }
}
