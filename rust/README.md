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
