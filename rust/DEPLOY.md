---
why: ragflow-server is meant to replace RAGFlow's Go server for the agent endpoints only, behind the same hostname, with a one-step way back
lane: pull
verify: curl -fsS http://<ragflow-server>:9380/healthz
invalidates-when: ragflow-server's routes change (the nginx locations below mirror its router), or its environment contract changes
---

# Running ragflow-server beside a RAGFlow

ragflow-server serves only the agent endpoints (README, "ragflow-server"). Everything else (UI, authoring, ingestion)
stays on RAGFlow's own server, so the two run side by side and a reverse proxy splits traffic by path. It reads and
writes the MySQL, Kvrocks and Elasticsearch RAGFlow already uses; there is no migration and no second datastore.

## 1. Check the schema first

ragflow-server follows a pinned upstream commit (`ragflow.upstream.commit` label). Before pointing it at a deployment on
a different RAGFlow release, compare the tables and columns it touches (`api_4_conversation`, `api_4_conversation_message`, `api_4_conversation_reference`, `user_canvas`,
`user_canvas_version`, `api_token`, `user`, `user_tenant`, `tenant`, `knowledgebase`, `document`, and the model tables
`tenant_model_provider`, `tenant_model_instance`, `tenant_model`; read the SQL in
`crates/ragflow-server/src/store.rs` and `sessions.rs` for the exact columns) with the deployment's schema, read-only. A missing column fails the request, not
the process; a column the server writes that the UI reads differently can corrupt what the UI shows. Do not migrate a
live database to satisfy it.

## 2. Run it

Join RAGFlow's compose network so the stores resolve by their service names, and reuse RAGFlow's env file:

```yaml
services:
  ragflow-server:
    image: <registry>/ragflow-server:<commit>
    env_file: [<path to RAGFlow's docker/.env>]
    environment:
      RAGFLOW_LISTEN: 0.0.0.0:9380
    ports: ["127.0.0.1:9380:9380"]
    networks: [ragflow]
    restart: unless-stopped
networks:
  ragflow:
    name: <RAGFlow's compose network>
    external: true
```

The variables are listed in README, "Environment contract". Check it:

    curl -fsS http://127.0.0.1:9380/healthz                        # {"status":"ok"}
    curl -s  http://127.0.0.1:9380/api/v1/agents/<id>/sessions      # no token: an error envelope, not a list

## 3. Split traffic by path

The locations below are exactly the routes the server registers (`router()` in `crates/ragflow-server/src/http.rs`);
anything else, including other `/api/v1/agents/...` routes, must reach RAGFlow's server. Place them in the server block
that proxies the rest to RAGFlow (placeholders in angle brackets):

```nginx
# Rust: agent completions, sessions, logs, task cancel, agentbots
location = /api/v1/agents/chat/completions { proxy_pass http://<ragflow-server>:9380; include <proxy-sse.conf>; }
location ~ ^/api/v1/agents/[^/]+/(sessions|sessions/[^/]+|logs/[^/]+)$ { proxy_pass http://<ragflow-server>:9380; include <proxy-sse.conf>; }
location ~ ^/api/v1/tasks/[^/]+/cancel$ { proxy_pass http://<ragflow-server>:9380; include <proxy-sse.conf>; }
location ~ ^/api/v1/agentbots/[^/]+/(completions|inputs|logs/[^/]+)$ { proxy_pass http://<ragflow-server>:9380; include <proxy-sse.conf>; }
# everything else: RAGFlow
location / { proxy_pass http://<ragflow>:80; ... }
```

`<proxy-sse.conf>` carries what streaming needs: `proxy_http_version 1.1; proxy_buffering off; proxy_read_timeout 300s;`
and the usual `Host` / `X-Forwarded-*` headers. Regex locations beat the prefix `location /`, so no ordering is needed.

## 4. Roll back

Revert the vhost (remove the four locations), `nginx -t`, reload. Traffic returns to RAGFlow's server at once; sessions
ragflow-server wrote are ordinary `api_4_conversation` rows and stay readable there. Then stop the container.
Run leases and cancel flags live in Kvrocks and expire on their own; an in-flight run on ragflow-server ends when the
container stops.
