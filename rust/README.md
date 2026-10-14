---
why: the fork must rebase onto infiniflow/ragflow without conflicts; only rust/ and one workflow file are added
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

## Staying on upstream

Everything this fork adds lives in `rust/` plus `.github/workflows/rust-runtime.yml`;
no upstream file is edited, so a rebase never conflicts:

    git fetch upstream
    git rebase upstream/main

## ragflow-analyzer — upstream's C++ text analyzer over FFI

`crates/ragflow-analyzer` binds `internal/binding/cpp` (the analyzer the Go server
links through cgo) and ships a CLI that prints, per stdin line,
`{"text", "tokens", "fine_grained"}`. The C++ archive is built by upstream's own
`./build.sh --cpp` inside upstream's builder image, so the crate is not a default
workspace member; build it with Docker (linux/amd64):

    docker build -f rust/docker/analyzer.Dockerfile -t ragflow-analyzer .
    echo '你好世界 Hello World' | docker run --rm -i ragflow-analyzer
    docker run --rm -i ragflow-analyzer --mode index --language English --input jsonl < chunks.jsonl
