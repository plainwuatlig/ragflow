#!/usr/bin/env bash
# Throwaway copies of a RAGFlow deployment's stores, with upstream's Go server
# and ragflow-server running against them, for the end-to-end comparison.
#
#   stack.sh up        build the copies, start both servers
#   stack.sh down      remove every container, volume and network it made
#   stack.sh status
#
# Nothing here writes to the live deployment: the MySQL copy comes from a
# mysqldump, the Kvrocks copy takes one key (the session secret), and
# Elasticsearch is reached through esproxy, which forwards only searches.
#
# Environment (defaults are the stock docker-compose names):
#   LIVE_ENV_FILE    the deployment's docker/.env (MYSQL_PASSWORD,
#                    REDIS_PASSWORD, ELASTIC_PASSWORD are read, never printed)
#   LIVE_NETWORK     compose network that reaches mysql/kvrocks/es01
#   LIVE_MYSQL       mysql container                 (ragflow-mysql-1)
#   LIVE_KVROCKS     kvrocks container               (ragflow-kvrocks-1)
#   LIVE_ES_URL      Elasticsearch base URL          (http://es01:9200)
#   GO_SERVER_IMAGE  image holding /ragflow_server   (ragflow-go-server:dev)
#   GO_RUNTIME_IMAGE official release image that supplies libraries/conf
#   GO_CONF_DIR      the conf/ tree of the commit GO_SERVER_IMAGE was built from
#   RUST_SERVER_IMAGE image built from rust/docker/server.Dockerfile
#   E2E_STATE        where generated secrets and logs go     (./state)
#   E2E_TZ           TZ of every container                   (UTC)
set -euo pipefail
cd "$(dirname "$0")"
here="$PWD"

LIVE_ENV_FILE="${LIVE_ENV_FILE:-$HOME/tools/ragflow/docker/.env}"
LIVE_NETWORK="${LIVE_NETWORK:-ragflow_ragflow}"
LIVE_MYSQL="${LIVE_MYSQL:-ragflow-mysql-1}"
LIVE_KVROCKS="${LIVE_KVROCKS:-ragflow-kvrocks-1}"
LIVE_ES_URL="${LIVE_ES_URL:-http://es01:9200}"
GO_SERVER_IMAGE="${GO_SERVER_IMAGE:-ragflow-go-server:dev}"
GO_RUNTIME_IMAGE="${GO_RUNTIME_IMAGE:-infiniflow/ragflow:v1.0.0-rc1}"
RUST_SERVER_IMAGE="${RUST_SERVER_IMAGE:-ragflow-rust-server:dev}"
STATE="${E2E_STATE:-$here/state}"
E2E_TZ="${E2E_TZ:-UTC}"
NET=rfe2e
SUBNET=11.250.0.0/24
P=rfe2e

# 11.250.0.0/24 is deliberately a publicly routable range: the Go server
# refuses model endpoints on private addresses, and the stub model must be
# reachable like a hosted one. The bridge is never routed anywhere.
ip_mysql=11.250.0.10 ip_kv=11.250.0.11 ip_es=11.250.0.12 ip_stub=11.250.0.13
ip_minio=11.250.0.14 ip_nats=11.250.0.15
ip_goadmin=11.250.0.20 ip_goapi=11.250.0.21 ip_rust=11.250.0.22

live_var() { grep -E "^$1=" "$LIVE_ENV_FILE" | head -1 | cut -d= -f2- | tr -d "\"'"; }
rand() { head -c 24 /dev/urandom | base64 | tr -d '/+=' | head -c 24; }
log() { printf '[stack] %s\n' "$*" >&2; }

load_secrets() {
  mkdir -p "$STATE"
  chmod 700 "$STATE"
  if [ ! -s "$STATE/secrets.env" ]; then
    umask 077
    {
      echo "E2E_MYSQL_PASSWORD=$(rand)"
      echo "E2E_REDIS_PASSWORD=$(rand)"
      echo "E2E_MINIO_PASSWORD=$(rand)"
    } > "$STATE/secrets.env"
  fi
  # shellcheck disable=SC1091
  . "$STATE/secrets.env"
}

mysql_copy() {
  docker exec -i -e MYSQL_PWD="$E2E_MYSQL_PASSWORD" $P-mysql mysql -uroot --default-character-set=utf8mb4 "$@"
}
kv_copy() {
  docker exec -i $P-kvrocks redis-cli -a "$E2E_REDIS_PASSWORD" --no-auth-warning "$@"
}

wait_for() { # wait_for NAME CMD...
  local name="$1"; shift
  for _ in $(seq 1 120); do "$@" >/dev/null 2>&1 && return 0; sleep 1; done
  log "$name did not come up"; return 1
}

up() {
  load_secrets
  docker network inspect $NET >/dev/null 2>&1 || docker network create --subnet $SUBNET $NET >/dev/null

  log "mysql copy"
  docker rm -f $P-mysql >/dev/null 2>&1 || true
  docker run -d --name $P-mysql --network $NET --ip $ip_mysql -e TZ="$E2E_TZ" \
    -e MYSQL_ROOT_PASSWORD="$E2E_MYSQL_PASSWORD" -e MYSQL_DATABASE=rag_flow \
    mysql:8.0.40 --max_allowed_packet=1073741824 >/dev/null
  wait_for mysql mysql_copy -e 'select 1'
  docker exec -e MYSQL_PWD="$(live_var MYSQL_PASSWORD)" "$LIVE_MYSQL" \
    mysqldump -uroot --single-transaction --no-tablespaces --set-gtid-purged=OFF rag_flow \
    | mysql_copy rag_flow

  log "kvrocks copy"
  docker rm -f $P-kvrocks >/dev/null 2>&1 || true
  docker run -d --name $P-kvrocks --network $NET --ip $ip_kv --user 0:0 -e TZ="$E2E_TZ" \
    -e REDIS_PASSWORD="$E2E_REDIS_PASSWORD" \
    -v "$here/../../docker/kvrocks-entrypoint.sh:/etc/kvrocks/entrypoint.sh:ro" \
    --entrypoint /bin/sh apache/kvrocks:2.16.0 /etc/kvrocks/entrypoint.sh >/dev/null
  wait_for kvrocks kv_copy ping
  # The session secret, copied unseen; Rust only ever reads it.
  secret="$(docker exec "$LIVE_KVROCKS" redis-cli -a "$(live_var REDIS_PASSWORD)" --no-auth-warning \
    -n 1 GET ragflow:system:secret_key)"
  if [ -n "$secret" ]; then
    printf '%s' "$secret" | kv_copy -n 1 -x SET ragflow:system:secret_key >/dev/null
  fi
  unset secret

  log "search-only Elasticsearch view"
  docker rm -f $P-es >/dev/null 2>&1 || true
  if ! docker image inspect $P-esproxy:dev >/dev/null 2>&1; then
    (cd esproxy && CGO_ENABLED=0 go build -o esproxy.bin main.go \
      && printf 'FROM scratch\nCOPY esproxy.bin /esproxy\nENTRYPOINT ["/esproxy"]\n' > Dockerfile.local \
      && docker build -q -f Dockerfile.local -t $P-esproxy:dev . >/dev/null; rm -f esproxy.bin Dockerfile.local)
  fi
  docker run -d --name $P-es --network $NET --ip $ip_es \
    -e ES_UPSTREAM="$LIVE_ES_URL" -e ES_USER=elastic -e ES_PASSWORD="$(live_var ELASTIC_PASSWORD)" \
    $P-esproxy:dev >/dev/null
  docker network connect "$LIVE_NETWORK" $P-es

  log "object store and queue the Go server wants at start-up (empty)"
  docker rm -f $P-minio $P-nats >/dev/null 2>&1 || true
  docker run -d --name $P-minio --network $NET --ip $ip_minio -e TZ="$E2E_TZ" \
    -e MINIO_ROOT_USER=rag_flow -e MINIO_ROOT_PASSWORD="$E2E_MINIO_PASSWORD" \
    pgsty/silo:RELEASE.2026-08-06T00-00-00Z server /data >/dev/null
  docker run -d --name $P-nats --network $NET --ip $ip_nats nats:2.14.2 -js >/dev/null

  log "stub model"
  docker rm -f $P-stub >/dev/null 2>&1 || true
  mkdir -p "$STATE/stub"
  : > "$STATE/stub/requests.jsonl"
  chmod 666 "$STATE/stub/requests.jsonl"
  [ -s "$STATE/stub/rules.json" ] || echo '[]' > "$STATE/stub/rules.json"
  docker run -d --name $P-stub --network $NET --ip $ip_stub -v "$here/stub_llm.py:/stub_llm.py:ro" \
    -v "$STATE/stub:/stub" python:3.11-slim python /stub_llm.py --port 8000 \
    --log /stub/requests.jsonl --rules /stub/rules.json >/dev/null
}

down() {
  for c in goapi goadmin rust stub nats minio es kvrocks mysql; do docker rm -fv $P-$c >/dev/null 2>&1 || true; done
  docker network rm $NET >/dev/null 2>&1 || true
}

status() { docker ps --filter "name=$P-" --format '{{.Names}}\t{{.Status}}'; }

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) echo "usage: $0 up|down|status" >&2; exit 2 ;;
esac
