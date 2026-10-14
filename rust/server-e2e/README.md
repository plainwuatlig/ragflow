---
why: ragflow-server claims upstream's Go server's behaviour on RAGFlow's own stores; this harness runs both against one throwaway copy of a deployment and diffs what they stream and write
lane: pull
verify: rust/server-e2e/run.sh PROBES.json CANVAS_ID  (exit 0 when no check FAILs; KNOWN rows are ruled differences)
invalidates-when: upstream changes the agent endpoints' events, the api_4_conversation* writes, the permission policy or the lease scripts; or ragflow-server's configuration variables change
---

# rust/server-e2e: Go server vs ragflow-server on one throwaway copy

What it compares, per run of `run.sh`:

* **Streams.** Every probe question goes through `POST /agents/chat/completions` on both servers with the
  same API token; the SSE events (ids and timestamps normalised, consecutive `message` frames merged) and the
  final answer must be equal, node by node.
* **Rows.** The `api_4_conversation`, `_message` and `_reference` rows each server leaves (ids normalised) must be
  equal, including the DSL snapshot, positions and the reference JSON.
* **Prompts.** The stub model logs every request; the two servers' requests (model, messages, sampling,
  stream options) must be equal, which pins history windows and prompt assembly.
* **Endpoints.** Session list and get answer alike from both servers over the same rows, and Go's endpoints
  show Rust-written sessions like Go-written ones; create, delete, batch delete, errors, non-streaming replies.
* **Authentication.** A matrix of good and bad tokens (expired, from the future, wrong secret, superuser, API token
  spellings, beta token on a standard endpoint, cookie fallback) answers alike.
* **Cancellation.** A slow run on one server is cancelled through the other, both ways and within one server.
* **Failures.** A model that answers HTTP 500 to the classifier or to the answering Agent, and a dataset the caller may
  not use: the event kinds, the text shown, the message rows and the `errors` column must be equal.
* **Team access and the beta binding.** An invited-but-not-accepted member is refused by both; a normal member runs
  the canvas but not another member's session. A beta token bound to another agent is refused by ragflow-server and
  run by upstream (a ruled difference).

## Pieces

| file | role |
|---|---|
| `stack.sh` | throwaway MySQL (a `mysqldump` of the live one), Kvrocks (one key: the session secret), an Elasticsearch view that forwards only searches (`esproxy/`), empty MinIO and NATS for the Go server's start-up, and the stub model |
| `go.sh` | upstream's Go server (admin + API) from `rust/docker/go-server.Dockerfile`, inside the release image's filesystem |
| `rust.sh` | ragflow-server from `rust/docker/server.Dockerfile` |
| `prepare.py` | points every OpenAI instance in the **copy** at the stub, adds API and beta tokens, copies the agent once, writes the stub's routing rules from the probe plan |
| `stub_llm.py` | deterministic OpenAI-compatible model: answers depend only on the prompt, so any difference in what the servers send shows up in the reply |
| `e2e.py` | the comparison |
| `run.sh` | all of it, then tear-down |

Nothing writes to the live deployment: MySQL is read through `mysqldump`, Kvrocks through one `GET`, Elasticsearch
only through `esproxy`. The copy's hosted-model keys are replaced before a server starts.

## Run

    mkdir -p /tmp/upstream && git archive <commit> go.mod go.sum internal cmd conf build.sh Dockerfile ragflow_deps \
      | tar -x -C /tmp/upstream && git show <commit>:docker/service_conf.yaml.template > /tmp/service_conf.yaml.template
    docker build -f rust/docker/go-server.Dockerfile --build-context upstream=/tmp/upstream -t ragflow-go-server:<commit> rust/docker
    docker build -f rust/docker/server.Dockerfile   --build-context upstream=/tmp/upstream -t ragflow-rust-server:dev .
    GO_SERVER_IMAGE=ragflow-go-server:<commit> GO_CONF_DIR=/tmp/upstream/conf \
    GO_TEMPLATE=/tmp/service_conf.yaml.template RUST_SERVER_IMAGE=ragflow-rust-server:dev \
      rust/server-e2e/run.sh my-probes.json <canvas id>

`probes.example.json` shows the plan: conversations of turns (each conversation is one session on each server) and
`rules`, `[substring, category]` pairs that route the stub's classification so the probes reach the branches they
mean to. The plan for a real agent holds that agent's questions, so keep it out of the repository.

`LIVE_ENV_FILE`, `LIVE_NETWORK`, `LIVE_MYSQL`, `LIVE_KVROCKS` and `LIVE_ES_URL` name the deployment being copied
(defaults are the stock docker-compose names). `E2E_TZ` sets the time zone of every container.

## Ruled differences

* A deferred Agent's `node_finished` outputs, and the consuming Message's `inputs`: upstream reports the unopened
  stream (`{"content": {"Open": null}}`), ragflow-server the real answer.
* A beta token bound to one agent cannot run another through `/agentbots/{id}/completions`.
* A storage failure on the agentbot endpoints: upstream's reply carries the cause (table and SQL error), ragflow-server's
  is generic. Deliberate: the cause is internal detail. The harness renames a table away on the copy to provoke it.
* `files` in a request are refused by ragflow-server (v1 scope).

Run failures (a failing classifier or answering model, a refused dataset) must read the same on both, in the stream and
in the stored rows; those checks are plain PASS, not known.
