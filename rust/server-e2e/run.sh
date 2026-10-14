#!/usr/bin/env bash
# Builds the throwaway stack, starts both servers, prepares the copy, runs the
# comparison, and (unless --keep) tears everything down.
#
#   run.sh PROBES.json CANVAS_ID [--keep]
#
# Needs the environment documented in README.md (GO_SERVER_IMAGE,
# GO_CONF_DIR, GO_TEMPLATE, RUST_SERVER_IMAGE, LIVE_*), and a Docker daemon
# that reaches the deployment being copied.
set -euo pipefail
cd "$(dirname "$0")"
probes="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
canvas="$2"
keep="${3:-}"
export E2E_STATE="${E2E_STATE:-$PWD/state}"
mkdir -p "$E2E_STATE"

cleanup() {
  if [ "$keep" != "--keep" ]; then
    ./rust.sh down || true
    ./go.sh down || true
    ./stack.sh down || true
    rm -rf "$E2E_STATE/go" "$E2E_STATE/secrets.env"
  fi
}
trap cleanup EXIT

./stack.sh up
./go.sh up
./rust.sh up
sleep 20 # the Go server needs a moment after its container starts
cp "$probes" "$E2E_STATE/plan.json"
python3 prepare.py --state "$E2E_STATE" --canvas "$canvas" --probes "$E2E_STATE/plan.json"
python3 e2e.py --state "$E2E_STATE" --canvas "$canvas" --probes "$E2E_STATE/plan.json" "${@:4}"
