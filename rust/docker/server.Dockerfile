# syntax=docker/dockerfile:1
#
# ragflow-server: the agent endpoints served from RAGFlow's own stores. The
# binary links the C++ text analyzer that upstream's `./build.sh --cpp` builds
# (retrieval tokenizes with it), so it is built in upstream's builder image
# against an upstream source tree, and runs on a plain Ubuntu with the
# release image's analyzer dictionaries and upstream's provider catalog.
#
#   mkdir -p /tmp/upstream && git archive <commit> build.sh go.mod go.sum internal conf \
#     | tar -x -C /tmp/upstream
#   docker build -f rust/docker/server.Dockerfile \
#     --build-context upstream=/tmp/upstream -t ragflow-server .
#
# Run it beside RAGFlow's stores with the environment listed in
# rust/crates/ragflow-server/src/main.rs.

ARG BUILDER_IMAGE=infiniflow/github_action_runner:latest@sha256:b712330a93b685625e4e4e0f3092b255eb96db9120a01ebfd3c392f7e644d47c
ARG RAGFLOW_IMAGE=infiniflow/ragflow:v1.0.0-rc1@sha256:31ccf0336d8910f6d90ac7d4fff766cc0483ec10042b906a377f27bfe09f818b
ARG RUST_IMAGE=rust:1.90.0-slim-bookworm@sha256:64232e656c058f4468e8d024e990acff04f0fd5a5c0a88a574dc37773d7325c9
ARG RUNTIME_IMAGE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

FROM ${RUST_IMAGE} AS rust-toolchain
FROM ${RAGFLOW_IMAGE} AS ragflow-dicts

FROM ${BUILDER_IMAGE} AS builder
USER root
SHELL ["/bin/bash", "-c"]
WORKDIR /upstream
COPY --from=upstream . .
RUN git config --global safe.directory "*" && ./build.sh --cpp
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH} \
    RAG_ANALYZER_LIB_DIR=/upstream/internal/binding/cpp/cmake-build-release \
    RAGFLOW_DICT_PATH=/usr/share/infinity/resource
COPY --from=ragflow-dicts /usr/share/infinity/resource /usr/share/infinity/resource
COPY rust /rust
WORKDIR /rust
RUN --mount=type=cache,id=ragflow-server-cargo,target=/usr/local/cargo/registry \
    cargo build --release --locked -p ragflow-server --features retrieval \
    && cargo test --release --locked -p ragflow-server \
    && mkdir -p /out && cp target/release/ragflow-server /out/

FROM ${RUNTIME_IMAGE}
COPY --from=ragflow-dicts /usr/share/infinity/resource /usr/share/infinity/resource
COPY --from=builder /out/ragflow-server /usr/local/bin/ragflow-server
COPY --from=builder /upstream/conf/models /ragflow/conf/models
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*
ENV RAGFLOW_DICT_PATH=/usr/share/infinity/resource \
    RAGFLOW_MODELS_DIR=/ragflow/conf/models
WORKDIR /work
USER 65534:65534
EXPOSE 9380
ENTRYPOINT ["/usr/local/bin/ragflow-server"]
