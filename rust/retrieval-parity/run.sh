#!/usr/bin/env bash
# Retrieval parity: runs every case through upstream's Go retrieval
# (oracle-go) and the Rust retrieval against the same live Elasticsearch and
# MySQL, with one recorded query embedding per query fed to both, then
# compares chunk ids and order, scores (1e-9) and formalized_content.
#
#   rust/retrieval-parity/run.sh CASES.jsonl OUT_DIR
#
# CASES.jsonl: one {"name": ..., "args": {<Retrieval tool args>}} per line.
# Environment (credentials are passed to the containers, never printed):
#   PARITY_IMAGE    image built from rust/docker/retrieval-parity.Dockerfile
#   DOCKER_NETWORK  network from which ES and MySQL are reachable
#   ES_URL ES_USER ES_PASSWORD
#   MYSQL_HOST MYSQL_PORT MYSQL_USER MYSQL_PASSWORD MYSQL_DB
#   OLLAMA_URL EMBED_MODEL   used once to record OUT_DIR/vectors.json
#                            (skipped when that file already exists)
# Both programs only read: oracle-go reaches ES through a proxy that forwards
# only GET / and POST */_search; the Rust side only issues _search.
set -euo pipefail
cases="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
out="$2"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
image="${PARITY_IMAGE:-ragflow-retrieval-parity}"

run() {
  docker run --rm -i --network "$DOCKER_NETWORK" \
    -e ES_URL -e ES_USER -e ES_PASSWORD \
    -e MYSQL_HOST -e MYSQL_PORT -e MYSQL_USER -e MYSQL_PASSWORD -e MYSQL_DB \
    -v "$cases:/in/cases.jsonl:ro" -v "$out:/out" --user "$(id -u):$(id -g)" \
    "$image" "$@"
}

if [ ! -s "$out/vectors.json" ]; then
  run ragflow-retrieval-parity embed --cases /in/cases.jsonl \
    --ollama "$OLLAMA_URL" --model "$EMBED_MODEL" > "$out/vectors.json"
fi
run sh -c '/oracle/bin/oracle-go -vectors /out/vectors.json -out /out/go.jsonl < /in/cases.jsonl' \
  > "$out/go.log" 2>&1
run ragflow-retrieval-parity run --cases /in/cases.jsonl --vectors /out/vectors.json \
  > "$out/rust.jsonl" 2> "$out/rust.stderr"
run ragflow-retrieval-parity compare --go /out/go.jsonl --rust /out/rust.jsonl | tee "$out/report.md"
