# pa-agent

The agent loop.

## Scope
Turn loop: drives a provider stream, dispatches tool calls, enforces max-turn and sequencing semantics, handles partial streams, retries on stream failure, propagates aborts.

## Non-goals
No tool implementations (pa-core), no providers (pa-ai), no session persistence, no UI. The loop is tool-agnostic: it receives a `ToolDispatcher` boundary.

## Public API
`run_loop` (session inputs -> stream of loop events), `ToolDispatcher` trait. Loop internals `pub(crate)`.

The settled-turn hooks receive `ShouldStopAfterTurnContext.tool_execution` (the
batch's effective `ToolExecutionMode`) and `tool_exchanges` (`Vec<NativeToolExchange>`).
Each exchange keeps its zero-based source order, call identity, original arguments,
validated execution arguments, and the final `ToolResultMessage` after the result
hook. Calls rejected before execution retain their original arguments as execution
input. Exchanges follow assistant source order, including duplicate call IDs, and
come from native execution rather than streamed events.

`AgentLoopConfig::get_tools` refreshes the live tool inventory after the awaited
provider-context transform, for both provider schemas and native dispatch.
`Agent::inject_host_message` accepts an already-persisted custom row with its
canonical `entryId`: idle rows append to live context; active rows enter steering.
Run settlement retains any host row arriving after the last steering poll in the
idle transcript rather than leaving it queued for an unrelated turn.

## Depends on
No workspace dependencies.
