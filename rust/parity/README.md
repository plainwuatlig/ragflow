---
why: the Rust runtime claims upstream's Go semantics; this oracle runs both on the same canvas, query and recorded LLM answers and diffs the event streams
lane: pull
verify: rust/parity/run.sh  (exit 0 when no case FAILs; KNOWN cases are ruled differences)
invalidates-when: upstream renames or reshapes service.AgentService.buildRunFunc, canvas.Runner, runtime.BindComponentFraction or models.GetPreconfiguredDriver; or the corpus gains a component the oracle cannot run without a DB
---

# rust/parity — Go vs Rust parity oracle

One command runs the whole corpus:

    rust/parity/run.sh                 # all cases
    rust/parity/run.sh --only switch_  # cases whose path contains "switch_"

It builds `oracle-go` (upstream's Go runtime with stubbed models), runs every
`corpus/*.json` case through it and through `ragflow-runtime`, and prints a
PASS/FAIL table followed by a readable diff per failing case. Per-case
artifacts (`dsl.json`, `recordings.json`, `go.jsonl`, `rust.jsonl`,
`go.stderr`) land in `rust/target/parity/<case>/`.

Needs: Go (any version; `run.sh` sets `GOTOOLCHAIN=go1.27.2` because go.mod's
bare `go 1.27` makes `GOTOOLCHAIN=auto` ask for a release named `go1.27`,
which does not exist), cargo, network for the first Go module download. No
DB, Redis, docker or model credentials.

## Pieces

- `oracle-go/` — `package main` inside the `ragflow` Go module (allowed to
  import `internal/`). It runs a canvas the way `RunAgent` does:
  `dsl.NormalizeForRun` → `canvas.DecodeFromDSL` → state seeding →
  `canvas.Compile` → `Workflow.Invoke` under `canvas.Runner`, emitting
  upstream's event names and data shapes. `run.go` is a copy of
  `service.AgentService.buildRunFunc` with every DB / Redis / run-tracker /
  upload / persistence step removed; re-sync it when rebasing.
  - Models: `conf/models` is loaded with `models.InitProviderManager` and
    every provider's `ModelDriver` is replaced by a stub, so Agent (eino
    ReAct → `EinoChatModel.Stream`) and Categorize (`einoChatInvoker` →
    `EinoChatModel.Generate`) reach it through the production
    `GetPreconfiguredDriver` lookup. The model id must name a provider in
    `conf/models` (the corpus uses `parity-chat@OpenAI`).
  - Caller identity: Agent and Categorize are wrapped via
    `canvas.WithComponentFactory`; the wrapper reads the cpn id the node
    wrapper binds with `runtime.BindComponentFraction` (through a run-level
    progress-fraction callback) and carries it into the call context,
    including into a deferred Agent stream that the Message opens later.
  - Built with `CGO_ENABLED=0`, which selects upstream's own pure-Go stub of
    the C++ tokenizer (`internal/binding/rag_analyzer_nocgo.go`).
  - `dao.DB` stays nil: the v1 components here never touch it when the model
    id carries a provider and `sys.tenant_id` is unset (credential lookup is
    skipped).
- `crates/ragflow-parity` — the Rust half: `run` (Rust runtime, same JSON
  lines), `diff`, and `corpus` (orchestrates both, compares, prints the
  table).
- `corpus/` — one JSON file per case; `corpus/canvases/` holds the
  hand-written canvases. Upstream templates are referenced in place from
  `internal/agent/templates/`.

## Cases and recordings

```json
{
  "canvas": "canvases/agent_chain.json",
  "query": "Write a tagline for a bakery",
  "patch":   { "Agent:Draft": { "llm_id": "parity-chat@OpenAI" } },
  "globals": { "env.limit": 50 },
  "recordings": { "Agent:Draft#0": "...", "Agent:Polish#0": "..." }
}
```

- Recordings are keyed `<component id>#<call index>`, not by prompt hash:
  the prompts themselves are under test. Both sides log every request as an
  `llm_call` line (model, messages, temperature/top_p/max_tokens) and the
  comparator diffs them.
- A missing key fails loudly on both sides (exit 3 / `MISSING-RECORDING`),
  printing the key and the messages, so a recording can be authored from
  the output.
- `sys.date` is pinned to `2026-01-01 00:00:00` unless the case sets it.
- Recordings must not contain `<think>` (the oracle's message emitter does
  not port upstream's inline think splitter, and refuses them).

## What is compared

Normalised away: ids, `created_at`, `elapsed_time`, message/session ids,
and the `_created_time` / `_elapsed_time` keys Go stamps into every output
bucket; numbers compare as f64.

Hard (fail the case): order of `node_finished` component ids; each
component's outputs; every LLM call; the concatenated `message` content;
`workflow_finished.outputs`; whether the run ended in `error`.
Soft (reported, never fail): error text; how message content is split into
frames.

Known (reported as KNOWN; the case neither passes nor fails; `run.sh` exits
0): ruled divergences, listed in `crates/ragflow-parity/src/compare.rs`.
There is one (Plain, 2026-10-14): a deferred Agent's `node_finished`
outputs, where Go reports the unopened stream `{"content":{"Open":null}}`
and Rust the Agent's real outputs. Every case with a deferred Agent shows
it.

## Scope

Begin, Categorize, Switch, Agent (no tools), Message. Retrieval, tools,
UserFillUp interrupts, history (multi-turn) and uploads are out of scope;
the oracle errors on an interrupt.
