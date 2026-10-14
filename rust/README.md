---
why: the fork must rebase onto infiniflow/ragflow without conflicts; only rust/ and one workflow file are added
lane: pull
verify: cd rust && cargo test
invalidates-when: the runtime moves out of rust/, or the fork starts patching upstream files
---

# rust/ — a Rust runtime for RAGFlow agent canvases

Compiles a RAGFlow agent canvas (DSL v2) and executes it with the semantics of
upstream's Go runtime (`internal/agent/`). Components outside
`ragflow_runtime::SUPPORTED` are refused at compile time.

## Build

    cd rust && cargo test && cargo clippy --all-targets -- -D warnings

## Staying on upstream

Everything this fork adds lives in `rust/` plus `.github/workflows/rust-runtime.yml`;
no upstream file is edited, so a rebase never conflicts:

    git fetch upstream
    git rebase upstream/main

## ragflow-analyzer — upstream's C++ text analyzer over FFI

`crates/ragflow-analyzer` binds `internal/binding/cpp` (the analyzer the Go server
links through cgo) and ships a CLI that prints, per stdin line,
`{"text", "tokens", "fine_grained"}`. The C++ archive is built by upstream's own
`./build.sh --cpp` inside upstream's builder image, so the crate is not a default
workspace member; build it with Docker (linux/amd64):

    docker build -f rust/docker/analyzer.Dockerfile -t ragflow-analyzer .
    echo '你好世界 Hello World' | docker run --rm -i ragflow-analyzer
    docker run --rm -i ragflow-analyzer --mode index --language English --input jsonl < chunks.jsonl

## ragflow-retrieval — the agent Retrieval over Elasticsearch

`crates/ragflow-retrieval` reproduces upstream's Go retrieval at v1.0.0-rc1
(query building, the ES request, the second KNN pass, scoring, pruning, the
adapter) on top of `ragflow-analyzer`, and implements the runtime's
`RetrievalService` for the `Retrieval` component. The `live` feature adds
Elasticsearch/MySQL/Ollama clients. `retrieval-parity/` runs it against
upstream's Go code on live stores; see its README.

## ragflow-server: RAGFlow's agent endpoints from RAGFlow's own stores

`crates/ragflow-server` serves the agent endpoints of upstream's Go server with no Go process, straight from the
MySQL, Kvrocks and Elasticsearch a RAGFlow deployment already has, using RAGFlow's own tokens:

| endpoint | notes |
|---|---|
| `POST /api/v1/agents/chat/completions` | SSE or one JSON reply; session token or API token |
| `GET/POST /api/v1/agents/{id}/sessions`, `GET/DELETE .../sessions/{sid}`, `DELETE .../sessions` | |
| `POST /api/v1/tasks/{sid}/cancel` | cancels a run in this process or in another one, Go or Rust |
| `POST /api/v1/agentbots/{id}/completions`, `GET .../inputs`, `GET .../logs/{mid}` | also a beta token, which runs only the agent it is bound to |
| `GET /api/v1/agents/{id}/logs/{mid}` | |

It follows `internal/` at the pinned upstream commit: the itsdangerous session token (HMAC-SHA1, 24 hour age, the secret
read from `RAGFLOW_SECRET_KEY` or Kvrocks and never created), API and beta tokens, the community permission policy
(owner, team members, invite rows excluded, sessions private to their creator), canvas and version selection, the
pre-run model check and the two model-resolution paths, the `api_4_conversation*` writes (one transaction each, the
parent row locked `FOR UPDATE`, positions as upstream computes them, the snapshot `buildPersistedAgentDSL` builds), and the
Kvrocks run lease with its Lua scripts. It runs the canvases of the components `ragflow-runtime` implements (`SUPPORTED`)
and says so, before writing anything, when a canvas has others.

Not served: `files` in a request (refused), the OpenAI-compatible mode, dataflow canvases, uploads and attachments, the
admin heartbeat and licence gate, and the `agent:run:*` run records that only UserFillUp resumption reads. Chat is spoken
to OpenAI-compatible endpoints and Ollama; hosted-provider addresses must be publicly routable, as upstream requires.

    cd rust && cargo test                                   # unit and in-memory tests
    docker build -f rust/docker/server.Dockerfile --build-context upstream=/tmp/upstream -t ragflow-server .

Configuration is the environment RAGFlow's deployment already defines (`MYSQL_*`, `KVROCKS_*`, `REDIS_PASSWORD`,
`ES_*`, `ELASTIC_PASSWORD`; see `crates/ragflow-server/src/main.rs`). `server-e2e/` runs the Go server and ragflow-server
against one throwaway copy of a deployment and diffs them.

