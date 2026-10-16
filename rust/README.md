---
why: the fork must sync with infiniflow/ragflow without conflicts; only rust/ and one workflow file are added
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

## The contract

The Rust runtime does not track Go's source; it implements a **versioned
contract**: the canvas DSL and each component's semantics, the SSE event
protocol, and the slice of RAGFlow's stores it reads and writes. The
contract's executable form is the conformance suite under
[`conformance/`](conformance/README.md): checked-in goldens recorded from
upstream's Go code when a contract version was cut, and a runner that needs
no Go.

    rust/conformance/run.sh canvas          # Rust runtime vs canvas goldens (cargo only)
    rust/conformance/run.sh retrieval       # Rust retrieval vs retrieval goldens (Docker)
    rust/conformance/run.sh server          # ragflow-server vs server goldens (Docker)

## Syncing with upstream and cutting a contract version

Everything this fork adds lives in `rust/` plus `.github/workflows/rust-runtime.yml`;
no upstream file is edited, so merging upstream never conflicts:

    git fetch upstream
    git merge upstream/main

Merging upstream does not change what the Rust runtime promises: that is the
contract version it declares. When upstream's semantics should become the new
promise, cut a contract version — `rust/conformance/cut.sh <version> <upstream ref>`
regenerates the goldens from upstream's Go code at that ref (the only step
that runs Go), then the Rust implementation is brought to the new goldens and
declares the version. See `conformance/README.md`.

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
Elasticsearch/MySQL/Ollama clients and the `ragflow-retrieval-conformance`
binary that the retrieval part of the conformance suite runs.

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

### v1 scope

Not served in v1:

* `files` in a request (refused with a clear error: file inputs need RAGFlow's document parsers).
* The OpenAI-compatible agent endpoint (`openai-compatible` requests are refused). It comes later, together with chat
  assistants and search.
* Canvases with components the runtime lacks (anything but Begin, Categorize, Retrieval, Switch, Agent, Message;
  dataflow canvases included): refused before anything is written.
* Chat providers other than OpenAI-compatible endpoints and Ollama: a run that needs another provider fails naming it.
  More providers come in a later ticket. Hosted-provider addresses must be publicly routable, as upstream requires.
* Uploads and attachments, the admin heartbeat and licence gate, and the `agent:run:*` records that only UserFillUp
  resumption reads.

### Deliberate differences from upstream's Go server

| difference | upstream | here | why |
|---|---|---|---|
| beta token bound to another agent on `/agentbots/{id}/completions` | runs it | refused: `API token is not authorized for this agent.` | a bound token should run only its agent |
| storage failure on the agentbot endpoints | the error text carries the storage cause (table, SQL error) | `Internal storage error while accessing the agent.` | the cause is internal detail |
| a deferred Agent's `node_finished` outputs and its Message's inputs | the unopened stream (`{"content": {"Open": null}}`) | the real answer | upstream's value is an artefact |
| `files`, OpenAI-compatible mode | served | refused | v1 scope, above |

Run failures are worded as upstream words them (`agent invoke: [NodeRunError] agent: component "<name>" invoke: ...`,
`**ERROR**: [GraphRunError] ...`), in the stream and in the stored message and `errors` rows. Transport failures
(connection refused, timeouts) carry the Rust HTTP stack's description after upstream's `failed to send request: Post "<url>":`.

    cd rust && cargo test                                   # unit and in-memory tests
    rust/docker/build-server.sh /path/to/upstream-checkout <upstream-commit>

### Environment contract

The names are the ones RAGFlow's own `docker/.env` and compose files already define, so the same env file can feed both
servers. "Required" means the server cannot work against a real deployment without it; the defaults are the names
RAGFlow's compose network gives its services.

| variable | default | required | meaning |
|---|---|---|---|
| `MYSQL_HOST`, `MYSQL_PORT` | `mysql`, `3306` | no | RAGFlow's MySQL |
| `MYSQL_USER`, `MYSQL_DBNAME` | `root`, `rag_flow` | no | |
| `MYSQL_PASSWORD` | empty | yes | |
| `KVROCKS_HOST`, `KVROCKS_PORT` | `kvrocks`, `6379` | no | RAGFlow's Kvrocks (Redis protocol): run leases, cancel flags, the session secret |
| `KVROCKS_DB` | `1` | no | the database number RAGFlow uses |
| `REDIS_PASSWORD` | empty | yes if Kvrocks has a password | |
| `ES_URL` | `http://${ES_HOST}:9200` | no | Elasticsearch base URL; wins over `ES_HOST` |
| `ES_HOST` | `es01` | no | used only when `ES_URL` is unset |
| `ES_USER` | `elastic` | no | |
| `ELASTIC_PASSWORD` | unset (no auth) | yes if Elasticsearch has security on | |
| `RAGFLOW_SECRET_KEY` | unset | no | the itsdangerous session-token secret; when unset it is read from Kvrocks (never created) |
| `RAGFLOW_MODELS_DIR` | `conf/models` (`/ragflow/conf/models` in the image) | no | upstream's provider catalog |
| `RAGFLOW_DICT_PATH` | `/usr/share/infinity/resource` | no | the analyzer's dictionaries (in the image) |
| `RAGFLOW_RES_DIR` | unset | no | the analyzer's resource directory, when it is not under the dictionary path |
| `RAGFLOW_LISTEN` | `0.0.0.0:9380` | no | bind address; the container's `HEALTHCHECK` probes this port |

Secrets belong in an env file or the orchestrator's secret store, never in the image. The server writes to the MySQL it
is given (sessions, logs); give it the deployment's database only on purpose, and it reads Elasticsearch only.

### Image, health, shutdown

`rust/docker/build-server.sh <upstream-repo> <upstream-commit> [image]` builds the image (see `server.Dockerfile`) and
tags it with this checkout's short commit, so a tag names one source tree. The image runs as `nobody`, listens on 9380,
and carries `org.opencontainers.image.revision` and `ragflow.upstream.commit` labels.

* `GET /healthz` answers `200 {"status":"ok"}` with no credentials and touches no store (liveness only: a store outage
  fails requests, not the container). `ragflow-server healthcheck` is the same probe for `HEALTHCHECK`.
* SIGTERM or Ctrl-C stops accepting and drains in-flight requests.

Deploying beside a RAGFlow, splitting traffic and rolling back: [DEPLOY.md](DEPLOY.md).

The server part of the conformance suite (`conformance/server/`) runs ragflow-server against the fixture stores and
compares what it streams and writes with the goldens recorded from upstream's Go server.
