#!/usr/bin/env bash
# Starts ragflow-server against the stack's throwaway stores.   rust.sh up|down
set -euo pipefail
cd "$(dirname "$0")"
STATE="${E2E_STATE:-$PWD/state}"
RUST_SERVER_IMAGE="${RUST_SERVER_IMAGE:-ragflow-rust-server:dev}"
E2E_TZ="${E2E_TZ:-UTC}"
P=rfe2e
# shellcheck disable=SC1091
. "$STATE/secrets.env"
case "${1:-}" in
  down) docker rm -fv $P-rust >/dev/null 2>&1 || true; exit 0 ;;
  up) ;;
  *) echo "usage: $0 up|down" >&2; exit 2 ;;
esac
docker rm -fv $P-rust >/dev/null 2>&1 || true
docker run -d --name $P-rust --network $P --ip 11.250.0.22 -e TZ="$E2E_TZ" \
  -e MYSQL_HOST=$P-mysql -e MYSQL_PASSWORD="$E2E_MYSQL_PASSWORD" \
  -e KVROCKS_HOST=$P-kvrocks -e REDIS_PASSWORD="$E2E_REDIS_PASSWORD" \
  -e ES_URL=http://$P-es:9200 -e ELASTIC_PASSWORD=unused \
  "$RUST_SERVER_IMAGE" >/dev/null
