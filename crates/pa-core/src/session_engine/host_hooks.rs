//! Hooks a host (the daemon worker, and through it an ACP client) installs
//! into a session's engine. Every field defaults to `None`, which keeps the
//! session exactly as without a host.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

/// Decides one kernel host call (`rlm.spawn`, `agent_message`, ...) before
/// its handler runs: `Ok(())` lets it run, an error refuses it and becomes
/// the error the Python caller sees. The gate enforces a rule, so a host
/// that cannot answer must return an error.
pub type HostCallGateFn = Arc<
    dyn Fn(String, Value) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>> + Send + Sync,
>;

/// The host hooks one session build installs.
#[derive(Clone, Default)]
pub struct SessionHostHooks {
    /// Awaited after a tool call's arguments validate and before it runs;
    /// a `block` answer turns the call into an error tool result carrying
    /// the reason.
    pub before_tool_call: Option<pa_agent::agent_loop::BeforeToolCallFn>,
    /// Awaited after a tool call runs; a returned override replaces the
    /// result's fields.
    pub after_tool_call: Option<pa_agent::agent_loop::AfterToolCallFn>,
    /// Awaited before every kernel host call's handler.
    pub host_call_gate: Option<HostCallGateFn>,
}
