#!/usr/bin/env bash
# Starts upstream's Go server (admin + API processes) against the stack's
# throwaway stores.   go.sh up|down
set -euo pipefail
cd "$(dirname "$0")"
here="$PWD"
STATE="${E2E_STATE:-$here/state}"
GO_SERVER_IMAGE="${GO_SERVER_IMAGE:-ragflow-go-server:dev}"
GO_RUNTIME_IMAGE="${GO_RUNTIME_IMAGE:-infiniflow/ragflow:v1.0.0-rc1}"
# The conf/ tree of the commit GO_SERVER_IMAGE was built from (model catalog,
# service_conf template).
GO_CONF_DIR="${GO_CONF_DIR:?GO_CONF_DIR is the conf/ directory of the Go server's source tree}"
GO_TEMPLATE="${GO_TEMPLATE:?GO_TEMPLATE is that tree's docker/service_conf.yaml.template}"
E2E_TZ="${E2E_TZ:-UTC}"
NET=rfe2e
P=rfe2e
# shellcheck disable=SC1091
. "$STATE/secrets.env"

case "${1:-}" in
  down)
    for c in goapi; do docker rm -fv $P-$c >/dev/null 2>&1 || true; done
    exit 0 ;;
  up) ;;
  *) echo "usage: $0 up|down" >&2; exit 2 ;;
esac

mkdir -p "$STATE/go"
cid="$(docker create "$GO_SERVER_IMAGE" /ragflow_server)"
docker cp "$cid:/ragflow_server" "$STATE/go/ragflow_server" >/dev/null
docker rm "$cid" >/dev/null
chmod 755 "$STATE/go/ragflow_server"

MYSQL_HOST=$P-mysql MYSQL_PASSWORD="$E2E_MYSQL_PASSWORD" \
MINIO_HOST=$P-minio MINIO_USER=rag_flow MINIO_PASSWORD="$E2E_MINIO_PASSWORD" \
ES_HOST=$P-es ELASTIC_PASSWORD=unused \
KVROCKS_HOST=$P-kvrocks REDIS_PASSWORD="$E2E_REDIS_PASSWORD" \
NATS_HOST=$P-nats \
  python3 "$here/render_conf.py" "$GO_TEMPLATE" > "$STATE/go/service_conf.yaml"
chmod 600 "$STATE/go/service_conf.yaml"

common=(--network $NET -e TZ="$E2E_TZ" -e DOC_ENGINE=elasticsearch -e DB_TYPE=mysql
  -e DEEPDOC_MODEL_DIR=/ragflow/rag/res/deepdoc
  -e KVROCKS_HOST=$P-kvrocks -e REDIS_PASSWORD="$E2E_REDIS_PASSWORD"
  -v "$STATE/go/ragflow_server:/ragflow/bin/ragflow_server:ro"
  -v "$GO_CONF_DIR:/ragflow/conf:ro"
  -v "$STATE/go/service_conf.yaml:/ragflow/conf/service_conf.yaml:ro"
  -w /ragflow)

# Like the release image, one container runs the admin server (the API
# server's license heartbeat) beside the API server.
docker rm -fv $P-goapi >/dev/null 2>&1 || true
docker run -d --name $P-goapi --ip 11.250.0.21 --entrypoint /bin/bash "${common[@]}" "$GO_RUNTIME_IMAGE" -c '
  /ragflow/bin/ragflow_server --admin --config /ragflow/conf/service_conf.yaml &
  sleep 5
  exec /ragflow/bin/ragflow_server --api --config /ragflow/conf/service_conf.yaml' >/dev/null
