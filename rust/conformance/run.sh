#!/usr/bin/env bash
# The conformance suite: the Rust implementation against the checked-in
# goldens of one contract version. Needs no Go.
#
#   rust/conformance/run.sh canvas    [VERSION] [--only <substr>]   cargo only
#   rust/conformance/run.sh retrieval [VERSION]                     Docker (fixture stores + analyzer image)
#   rust/conformance/run.sh server    [VERSION]                     Docker (fixture stores + ragflow-server image)
#   rust/conformance/run.sh all       [VERSION]
#
# VERSION defaults to v1 (rust/conformance/contract/<VERSION>). Exit 0 when
# no check FAILs; KNOWN rows are the ruled differences listed in README.md.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
here="$root/rust/conformance"
part="${1:?part: canvas|retrieval|server|all}"
shift
version="v1"
if [ "${1:-}" != "" ] && [ "${1#--}" = "$1" ]; then
  version="$1"
  shift
fi
contract="$here/contract/$version"
[ -d "$contract" ] || { echo "no contract $version at $contract" >&2; exit 2; }

canvas() {
  (cd "$root/rust" && cargo run --quiet -p ragflow-conformance -- canvas check \
    "$contract/canvas" --out "$root/rust/target/conformance/$version/canvas" "$@")
}

retrieval() {
  "$here/server/stack.sh" up
  trap '"$here/server/stack.sh" down' EXIT
  "$here/retrieval.sh" check "$contract"
}

server() {
  "$here/server/stack.sh" up
  trap '"$here/server/stack.sh" down' EXIT
  "$here/server/rust.sh" up
  "$here/server/check.sh" "$contract"
}

case "$part" in
  canvas) canvas "$@" ;;
  retrieval) retrieval ;;
  server) server ;;
  all)
    canvas
    "$here/server/stack.sh" up
    trap '"$here/server/stack.sh" down' EXIT
    "$here/retrieval.sh" check "$contract"
    "$here/server/rust.sh" up
    "$here/server/check.sh" "$contract"
    ;;
  *) echo "usage: $0 canvas|retrieval|server|all [VERSION] [--only <substr>]" >&2; exit 2 ;;
esac
