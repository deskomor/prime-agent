# pa-daemon

Session supervision and wire serving.

## Scope
ACP stdio transport (`acp`): the JSON-RPC serve surface for Agent
Client Protocol clients, served over a daemon session (initialize, session/new, session/prompt, session/close,
session/cancel, and the outgoing session/update notification with
namespaced `_meta` correlation), owned by this crate because
wire-protocol serving is its area; the worker turn loop owns compaction
arms, auto-refine, and goal continuation. The compact-trigger
auto-refine scheduling owns its surface here (`compact_autorefine.rs`
for the daemon worker: every compaction arm plus the manual `compact`
command arm the pa-core trigger, and the quiescent turn boundaries /
the command path consume the gated review - TS
`_scheduleAutoRefineAfterCompaction` plus the background
`_maybeAutoRefine("compact")`).
Supervisor process (one worker process per active session), restart/backoff
supervision, session registry/roster + worker self-registration (session
identity survives supervisor restarts: workers re-register with backoff and
the roster rebuilds), parent-death child cleanup (`supervisor_parent_death.rs`:
an unexpectedly exited worker's resident RLM children close with it - the
same plain-stop semantics as the worker-side #246 close, which SIGKILL
bypasses - via the durable create's `parentActiveSessionId` join), append-only session store ownership (including the
compaction-entry fold for compacted message reads), client attach/detach
(full-snapshot and chunked `session_snapshot_begin`/`chunk`/`end`
streaming), direct-attach transport (supervisor-issued single-use tickets
with a 10s TTL, worker-side peer grants burned on first use, session-plane
command gating on peer links), kernel agent_message/agent_observe bridge (`agent_messaging.rs`: the worker-side controllers over the supervisor link; the family view joins the supervisor roster with the session's own RLM children registry - the same registry `rlm.list_subagents` reads - so registry children are Child members addressable by name, RLM child id, and persisted session id, the spawning session is the Parent member for a subagent worker, and everything else stays a sibling; delivery tries the direct peer transport then the supervisor-routed `send_message`), worker-to-worker peer messaging (stage 3:
`worker`-purpose single-use grants minted by `get_worker_peer_transport`
for a source worker's kernel `agent_message.send`, direct
`worker_deliver_message` on the target worker's socket with the supervisor
routed `send_message` as the never-retried fallback),
saved-session wake
for non-resident `send_message` targets (`session_catalog.rs`: catalog
resolve by session-id prefix or exact name, cwd-scoped first; a live miss
lazily restores a matching file from `sessions-archive/` without
overwriting an existing live file; no new sessions are physically archived;
a resident worker hosting the file is reused, otherwise one spawns over it; RLM
children fall back to the spawn ledger's live child edges when the catalog
misses - children persist outside the sessions dir, under the parent's
session-artifacts tree - and wake over their own session file), wire protocol serve/negotiation (including the
`compact`/`abort_compaction`/`set_auto_compaction` commands and their
`compaction_start`/`compaction_end` events, and the live
`set_model`/`set_thinking_level` commands — `model_switch.rs`: registry
resolution, the durable `model_change`/`thinking_level_change` rows, the
settings defaults, and the engine's live agent/provider-target/level
switch), session export commands (`session_export.rs`: the worker-side
`export_html`/`export_jsonl` handlers; the HTML data carries the engine's
tools section and custom-tool pre-render through the
`SessionEngine::export_tools`/`export_rendered_tools` seams, built
lazily when an export precedes the first turn - the TS state exists from
create), the `get_model_catalog` command (the TS `refreshModelCatalog`:
registry refresh — live Prime Inference catalog fetch, disk cache at
`<agent-dir>/prime-inference-models-cache.json`, bundled fallback — then
the full catalog minus unauthorized private models plus the configured
providers; the worker create path fires the same refresh at startup), the
agent-roster arms
(`roster_subscribe`/`roster_unsubscribe` with the full snapshot, live
`roster_update` pushes keyed by the TS roster `agentId` = session id, and
authenticated `worker_roster_delta` self-reports so live status reaches
subscribers without polling) plus the live-roster ledger seed (TS
`seedRosterLedger`, `supervisor_roster.rs`: passivated ledger descendants
of resident workers seed into the live roster - roots are the resident
session files, descent is membership at any parent-walk step, and
present rows by agent id or session file are never overwritten - at
subscribe, spawn-admission, and worker-stop moments, so subscribers see
the full family, not just resident rows), cloud sandbox attach, session
leases (`core/session-lease.ts` port). Daemon-owned RLM spawn ledger (`rlm_ledger.rs`):
one append-only JSONL per sessions dir (spawn/rename/delete admissions with
the TS `rlm-ledger.ts` record grammar, bounds, stat-guarded replay, and
legacy-registry seeding) plus the per-child display files; family topology
is read from the ledger, never re-derived from session files. Passive-RLM
roster walk (`rlm_roster.rs`): `list --all` and the saved-session catalog
merge non-resident ledger children (walk roots = saved + resident session
files), so passivated children stay roster-visible (TS
`walkPassiveRlmSubagents` / `withPassiveRlmDescendantInfos`). The saved
session scan (`session_store.rs`) folds each file once into the durable
catalog row.
Supervisor-backed RLM child sessions
(`rlm_children.rs`, the daemon side of the pa-core `RlmSubagentHost` seam):
`rlm.spawn`/`rlm.create_session` create real daemon sessions through the
worker's supervisor link - one supervised worker process per child - and the
parent-side registry serves `rlm.list_subagents`/`rlm.collect`/
`rlm.delete_subagent` with TS-parity selector errors; child model resolution
and thinking-level validation live in `rlm_child_model.rs`; the create
command carries the RLM recursion identity (`rlmDepth`/`rlmMaxDepth`/
`parentSessionPath`/`thinking`) so respawned children keep their depth. Per-session model binding: the
create-config `provider`/`model`/`apiKey` are authoritative for worker model
resolution (explicit CLI flags reach the worker; env remains the no-flag
fallback). Worker session files carry the TS creation prefix
(`model_change`/`thinking_level_change`/`service_tier_change`), and queue
snapshots persist to the worker recovery journal, not the session file.
Queue-lane command surface (`queue_commands.rs`): the full TS
`DAEMON_COMMAND_TYPES` accept list with exhaustive router tables in
`protocol.rs`, plus the worker's `mutate_queued_message`/`resume_queue` arms
(`AgentSession.mutateQueuedMessage`/`resumeQueuedWork`: preview-addressed
delete/move/replace over the two lanes with the TS status vocabulary, and
the empty-queue resume refusal). The `sessionActions` queue projection
(`SessionActionSnapshot`) carries one Rust-native typed rider the TS wire
has no counterpart for: `rlmChildStatus`, the parked RLM child status
notices' lane indices, derived per item from the injected custom row (the
`rlm_child_terminal_notice`/`rlm_child_failure` kinds) at projection time —
so journal recovery re-derives it, the lane strings stay the TS
`queuedAgentMessagePreview` projection verbatim, and the condensed queue
strip folds exactly these rows (a user-typed lookalike never flags); the
rider serializes only when a notice is parked. The reserved kinds are
daemon provenance, never client data (`child_status_notices.rs`): the
prompt/steer/follow-up parse and the `restore_actions` validation answer a
caller-supplied custom row claiming one loudly — it never parks, never
folds — and the daemon's own notice injection rides the follow-up route
with a one-shot capability minted in the worker process (the wire's
optional `rlmNoticeNonce`), so the queue's classification only ever sees
daemon-authentic rows.
Prompt attachments: the `prompt`/`steer`/`follow_up` wire `images` array
(base64 payload + mime type) rides the queue item into the session engine
as multimodal user content (images on a queued prompt do not survive a
worker respawn - the recovery journal keeps the text lanes only, the TS
command-recovery shape).
Eager turn abort (TS `requestAbort`'s closing `agent.abort()`): every
worker abort surface — `abort`/`abort_and_clear_queue`, the compaction and
branch-navigation interrupt-and-settle waits, shutdown, kill, and the
`cancel_prompt_admission` cancel-owned arm — funnels through
`SessionEngine::abort_in_flight_turn` (the engine's build-time agent
mirror), so an abort landing mid-provider-wait cancels the in-flight fetch
immediately and the aborted turn settles on its zero-usage aborted message
(the usage-accounting parity of the #238 adjacent gap 3 fix; the transport
half is pa-core's provider-adapter cancellation token).
Queued-input suspension (TS `_sessionInputPumpSuspended`, the #227/#233
ruling): `abort`/`abort_and_clear_queue`/manual
`compact` suspend queued-input admission indefinitely; while suspended a
plain `prompt`/`prompt_and_wait` is rejected with the TS admission error
and the lanes park. Resume sites: a prompt carrying `streamingBehavior`,
`steer`/`follow_up`, `resume_queue` (even on the empty queue), an applied
`mutate_queued_message`, a cron/heartbeat fire, and a successful compact
with an active goal. Agent-message delivery is rejected while suspended
and idle (TS `acceptAgentMessagePrompt` runs with `resumeIfIdle: false`).
The compact-with-active-goal resume is also the TS post-compaction
continue (agent-session.ts `compact()`'s `didCompact` branch): with no
queued work parked, the worker mints the owed goal continuation through
the engine (`SessionEngine::mint_post_compaction_goal_continuation`, the
`_maybeResumeGoalContinuationAfterRlmWork` mint — the follow-up lane item
with the durable goal-context row), emits the mint's `goal_update`, then
resumes: the parked continuation crosses the suspension gate through the
resume site and the turn runner drives it (the `_schedulePostCompactionContinue`
schedule). The compact-trigger auto-refine defers behind that continuation
(TS `_compactAutoRefinePending`), servicing at its turn boundary.
Thread-goal durability (TS one-store parity): every `goal_update`
announcement mirrors the new state as a `thread_goal_state` custom row in
the worker session file (the turn emit closure; the compact mint persists
its row at the mint site, outside a turn), and the engine's session-build
adoption rehydrates the fresh goal driver from the moved branch's or the
session file's latest valid `thread_goal_state` entry
(`goal_state_persist.rs`) — status, objective, usage counters, and
continuation counts continue across a worker-recovery rebuild, and the
seeded published baseline keeps the rehydrated state from announcing
itself (TS loads at construction without emitting). Recovery branch
rebuild (TS one-store parity): the owned-session worker respawns with
`--resume <sessionFile>` (`createRpcRecoveryArgs`), so the rebuilt
session's branch is the durable history — while the Rust worker owns the
file writes and the engine keeps an in-memory manager, the engine's
session-build adoption rebuilds the branch from the worker session file
(the `pending_branch` seam's recovery twin), so a post-recovery compaction
walk (and the live loop context) reads the full durable history instead
of a fresh engine's empty branch skipping "Session is too short to
compact". Platform wall
(`platform`): per-OS endpoint naming and socket identity; the transport itself
is the shared trait in `pa_types::platform::transport`, and the private-frame
codec plus command planes are the shared wire contract in
`pa_types::daemon` (clients in pa-tui/pa-cli speak them).

Live token-stream rendering
(`streaming`): turn events forward from the hosted engine to the worker's
emit path as they arrive (one `message_update` per provider delta, the
full partial message per frame - never buffered until turn settle); the
worker coalesces them for broadcast in a single-slot coalescer with a
50ms flusher, while `message_start`/`message_end`/tool frames flush the
parked update first and go out directly, so wire order and event-sequence
order match uncoalesced streaming. The supervisor stays payload-free:
deltas ride the worker -> client session-event stream (direct-attach or
supervisor-routed).

Autonomous
continuation driving in the worker's engine: per-message usage accounting
runs in the agent-loop subscription, and the in-run continuation hook
(`autonomous_continuation.rs`) consults the pa-core `AutonomousDriver`
policy at the agent loop's natural turn end (product default: shell
quality gates in the session cwd; deterministic drivers injectable for
eval harnesses) — continuations mint as durable user rows churned
inside the one prompt wait (the TS `getContinuationMessages` shape), a
threshold compaction holds its minted continuation for the worker's
queued `followUp` admission, and gate pass/fail or limit stops surface no
row (the headless status request and the print exit contract carry
them).

Live MCP product-path verifier
(`tests/mcp_product_path_e2e.rs`): a settings-declared stdio MCP server
round-trips through a real worker session and kernel - the kernel cell's
`mcp.list_tools`/`mcp.call_tool` resolve `mcp.config` through the
session's host handlers, spawn the fixture
(`tests/fixtures/mcp_echo_server.py`), and echo back.

Update-prepare transaction (`update_prepare.rs`): the supervisor-side FSM
`Draining -> Fenced -> Snapshotted -> Prepared -> Stopping` with the
mutation-drain latch + admission gate (mutating commands refused with
"Daemon is preparing an update restart" while active; reads/attach and the
TS abort-family drain commands stay served), the hard prepare deadline
(90 s default) and the durable 45 s marker self-expiry as watchdogs — both
re-checked on a timer and on any later command — idempotent
`prepare_update_restart` on `updateId` (a different id is a typed refusal
the coordinator maps to Join), and `Aborted -> Serving` with prepared-dir
cleanup as the failure default. The prepare RPC drives accept
through the mutation drain to `Fenced`, snapshots every resident worker
(`update_snapshot`, `worker.rs`), assembles the roster (`update_roster.rs`:
sessions/workers/subagents from the ledger + displays, heartbeats as a
read-only `scheduled-jobs.json` projection), writes
`prepared/<id>/{roster,marker}.json` durably, and acks `Prepared` — all
inside the prepare budget. `commit_update_restart` consumes the
transaction (`update_stop.rs`): the acked worker `shutdown` (its handler is
the flush barrier) with the 30s + 30s budgets, abandon-on-refusal (the
supervisor resumes `Serving`, refused sessions untouched, stopped workers
relaunched — never a kill), or the update exit when all workers stopped
(descriptors survive for the new supervisor's create-or-adopt restore).
The new supervisor's boot owns the update's restore side
(`update_restore.rs`, spec §6): it consumes `PRIME_AGENT_UPDATE_ROSTER`
from its spawn env before the unconditional scratch-dir sweep (this
socket's `update-restarts/` subtree plus the TS-era legacy names — no
liveness checks, no exceptions), then restores the roster rows bottom-up
(create-or-adopt; per-session failure records, restore never fails the
boot), re-arms scheduled work (a boot scan of `scheduled-jobs.json`;
due active jobs of sessionless files are woken once, never archived), and
reports the pass over `update_restore_status` while `hello.update_resume`
carries the settle state; client attaches to a not-yet-restored roster row
queue behind the pass instead of failing.
For historical TypeScript update manifests, the restore pass recreates parents
before their children, remaps active parent IDs, and restores pending context
and action queues explicitly because those manifests predate Rust's journals.
Attaches remain gated until that replay completes.
The heartbeat-catalog change
broadcast owns its surface here too (the worker's cron-store
`on_heartbeat_change` listener emits a `heartbeats_changed` outbound frame
— TS daemon-mode's `broadcastGlobal` — and the supervisor re-broadcasts it
to every client, so any heartbeat mutation anywhere in the daemon reaches
attached UIs daemon-wide).

Host state access (`agent_engine/host_session.rs`) uses the mirrored live agent
rather than holding the core-session mutex across an awaited host request.
The worker owns custom-message persistence; injection adds its canonical row only
to live agent context. Host active-tool selections narrow the retained native
inventory and can widen back to it; detach restores that inventory and clears only
host prompt overrides and suffixes, leaving the native prompt unchanged. Selection
writes, detach, and runtime adoption serialize through the same async lock.
`SessionEngine::host_context_projections` exposes cached applied projections by
purpose, separate from the worker's durable canonical context. The connection
installing host hooks owns their lease: only its disconnect clears hooks,
host prompt/tool overrides, and the host refinement pause. Supervisor-routed hooks
carry a unique root-adapter owner token from the original client connection,
not the shared worker-hop token. Explicit root detach releases its targeted
sessions and inherited descendants; socket disconnect releases all of that
socket's adapters. Other roots and owners remain attached. Cleanup resolves
rebound selectors and removes stale inherited config from child descriptors.
Child admission fences inheritance against owner release, so a captured spawn
cannot restore a detached adapter.

## ACP host session access

Negotiated Studio host contract v1 exposes `session_snapshot`,
`append_custom_message`, and `set_active_tools` in the
`_ai.primeintellect.prime-agent/` namespace. `sessionId` always names the ACP
root; optional `targetActiveSessionId` must resolve through native parent links
as that root or a descendant. `append_session_entry` and `set_auto_refine`
accept the same scoped target; their legacy root-only calls need no negotiation.

Snapshots carry native durable identity, the header and leaf, a revision tuple,
active tool schemas, model and context usage, and faithful branch entries,
including bookkeeping. `offset` and `limit` page entries, canonical context, and
each available applied projection independently; each collection has its own
`total` and `nextOffset`. `entryRefs.messageIndex` is local to the returned
message page. Projection identities come only from applied native projections.
`contextUsage` uses the applied provider projection only when its complete
source matches the current canonical context; `contextUsageSource` names that
choice. `canonicalContextUsage` remains available, and cached projection usage
is labeled `lastAppliedProjectionUsage` rather than assumed current.
Pages stay below the wire budget without changing row content; a single row or
metadata too large for that budget returns an explicit error. A revision change
between pages requires a fresh read. Reentrant handlers use worker state and
cached live engine handles, not the engine's outer session mutex.

Custom messages persist once before context injection and return the durable
`entryId`; a failed injection reports the already-persisted identity. Empty host
hook registration restores native tools and clears host prompt overrides.

## Optional Studio ACP host contract

Initialize opts in with `clientCapabilities._meta["ai.primeintellect.prime-agent"].hostContract.version = 1`.
The response publishes the implemented v1 features. Hook callbacks use
`_ai.primeintellect.prime-agent/hook`; negotiated hooks default to 150 seconds.
The native origin identifies the executing durable session, including child hooks.
The root admits `session/new` before awaiting its actual `session_start`; callback
snapshots and writes can reenter while prompts wait on the session's startup gate.
Close and disconnect cancel that session's startup wait, not a replacement's gate.
An opted-in child emits its actual `session_start` after native create/resume and
before queued input can run. Create returns to let the supervisor mark its route
ready; an owned queued-work pause remains held until the native hook settles,
so callback snapshots and writes can reenter without racing the first prompt.
Graceful shutdown and kill/cascade paths cancel work before native
`session_shutdown`, whose optional notification has a two-second budget before
session identity is cleared. Abrupt process death cannot emit a hook.
The `family_scope` feature declares authoritative caller-family payloads and native
durable-session scope enforcement; it is not a runtime switch. `hostContract.build`
reports `{commit, dirty}` captured by this crate's std-only `build.rs`, with null
fields when Git identity is unavailable. Git HEAD, its resolved branch ref,
packed refs, index and tracked source inputs invalidate the compile metadata.
`dirty:false` requires explicit build-time `PRIME_AGENT_CLEAN_BUILD_COMMIT` matching
actual HEAD and an empty Git status. Ordinary builds report observed dirty state
or null, never implicit clean status; changing/removing that build input reruns
the capture. `protocol::build_identity()` exposes the same immutable fields to the CLI; product
version and mutable runtime checkout state are not build identity.
Guarded v1 `agent_message.*` and `agent_observe.*` host calls include top-level
`payload.family = { current, agents }` from the executing worker's native observe
controller. Rows carry native session IDs, names, relationships, cwd, parent/child
IDs, runtime kind and state flags, never message previews. Optional native fields
remain absent when unavailable. User `data.family` is not authoritative. Each child
reads its own family before relay; the root preserves it. Missing/failed native
reads refuse guarded calls. Hooks off, legacy calls and non-family calls do not
read this roster. Host-returned durable session scopes still constrain native
family lists, alias selection and sends.

Terminal `turn_end` advice resumes the same prompt only with explicit `continue:true`
and nonempty custom messages. Without v1, the original hook behavior remains.

`session_snapshot`, `append_session_entry`, `append_custom_message`, `set_active_tools`,
and `set_auto_refine` share that extension namespace. `sessionId` identifies the ACP
root; optional `targetActiveSessionId` must resolve to that root or its native
descendants. Snapshots page faithful branch entries and active context separately;
context refs and applied projection metadata remain separate from durable history.
Writes return canonical committed entry IDs and remain callable during awaited hooks.

Studio MCP inheritance requires v1 and per-stdio-server
`_meta["ai.primeintellect.prime-agent"].inheritStudioMcp = true`. Native code stamps
`PRIME_STUDIO_NATIVE_SESSION` for the root and each actual child before discovery.
The durable child create payload retains the v1 marker, server configs and parent
session ID, so initial launch and replay both reach the child's identity binding.
A child never inherits a parent's native identity. Generic manager ownership and
non-Studio server behavior remain unchanged.

## Non-goals
No agent behavior inside workers beyond hosting a pa-core engine; no UI.

## Public API
Supervisor entrypoint, worker entrypoint, `protocol::build_identity()` (immutable
build-source commit and explicit clean/dirty metadata shared with CLI and ACP),
`mcp_login::{WorkerMcpLoginUi, wire_worker_mcp_login}` (the worker's browser+callback login behind `mcp.begin_login`; wired by the agent engine before sessions register host handlers), client connection API for pa-tui/pa-cli, `acp::daemon::{run_daemon_attached_acp_mode, DaemonAcpOptions}` (pa-cli dispatches `--mode acp` through it), `agent_messaging::LinkAgentMessageController` + `rlm_children::{SupervisorChildSessions, ParentIdentity, RlmChildIdentity}` (e2e verifiers construct the worker-side family controller and the children registry; the engine wires the same types), `agent_engine::AgentSessionEngine::dispose_kernel` (the session-end kernel teardown the worker invokes at kill/shutdown/orphan exit — the engine outlives the session, so the pa-core engine-drop teardown cannot run there), the worker's `get_mcp_connections` command (`mcp_connections.rs`: the `/mcp` view's roster from the session's MCP manager overlaid with the session kernel's per-server tool listing, plus the api-key credential rows the same response serves — Rust-native session-plane extension; the TS daemon has no counterpart). Supervision internals `pub(crate)`.


## Depends on
pa-types, pa-core (one-way).

The composition root initializes the packaged product version with
`protocol::configure_app_version` before entering daemon or worker mode.
`protocol::app_version` supplies that identity to both hello frames and update
checkpoints; library embedders fall back to the compiled version. Release
manifest restamping therefore remains consistent with CLI `--version`.
