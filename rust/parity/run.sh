#!/usr/bin/env bash
# Runs every case in rust/parity/corpus through upstream's Go runtime
# (oracle-go) and the Rust runtime, and prints PASS/FAIL per case.
# Extra arguments go to `ragflow-parity corpus` (e.g. --only switch_).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$root/rust/target/parity"
mkdir -p "$out"
# go.mod says `go 1.27`, which GOTOOLCHAIN=auto resolves to the
# non-existent release name "go1.27"; name a real patch release instead.
export GOTOOLCHAIN="${GOTOOLCHAIN:-go1.27.2}"
# CGO off selects upstream's own pure-Go stub of the C++ tokenizer
# (internal/binding/rag_analyzer_nocgo.go); no v1 component on this path
# tokenizes.
(cd "$root" && CGO_ENABLED=0 go build -o "$out/oracle-go" ./rust/parity/oracle-go)
cd "$root/rust"
cargo run --quiet -p ragflow-parity -- corpus "$root/rust/parity/corpus" \
  --oracle "$out/oracle-go" --conf "$root/conf/models" --out "$out" "$@"
