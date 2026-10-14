# syntax=docker/dockerfile:1
#
# Upstream's Go server, built exactly as upstream's Dockerfile builds it
# (go-builder stage: ./build.sh --cpp, then ./build.sh --go), for the
# end-to-end comparison in rust/server-e2e. Only the binary leaves the build.
#
#   mkdir -p /tmp/upstream && git archive <commit> go.mod go.sum internal cmd conf \
#       build.sh Dockerfile ragflow_deps | tar -x -C /tmp/upstream
#   docker build -f rust/docker/go-server.Dockerfile --build-context upstream=/tmp/upstream \
#       -t ragflow-go-server:<commit> rust/docker
#
# The result is a scratch image holding /ragflow_server; rust/server-e2e/run.sh
# copies it into the official release image, which supplies the runtime
# libraries, the model catalog and the dictionaries.

ARG BUILDER_IMAGE=infiniflow/github_action_runner:latest@sha256:b712330a93b685625e4e4e0f3092b255eb96db9120a01ebfd3c392f7e644d47c

FROM ${BUILDER_IMAGE} AS go-builder
USER root
SHELL ["/bin/bash", "-c"]
WORKDIR /ragflow
ENV GOMODCACHE=/root/.cache/gomod GOCACHE=/root/.cache/gobuild GOTOOLCHAIN=local

COPY --from=upstream go.mod go.sum ./
RUN --mount=type=cache,id=ragflow-go-server-gomod,target=/root/.cache/gomod,sharing=locked \
    --mount=type=cache,id=ragflow-go-server-gobuild,target=/root/.cache/gobuild \
    GOPROXY=${GOPROXY:-https://proxy.golang.org,direct} go mod download

COPY --from=upstream internal internal
COPY --from=upstream cmd cmd
COPY --from=upstream conf conf
COPY --from=upstream ragflow_deps ragflow_deps
COPY --from=upstream build.sh Dockerfile ./

ARG ORT_VERSION=1.29.0
RUN set -eux; \
    mkdir -p "${HOME}/ragflow-native-libs/onnxruntime/static_lib"; \
    _src="/opt/ragflow-native-libs/onnxruntime/static_lib"; \
    _v="$(ls -d "${_src}"/onnxruntime-linux-x64-static_lib-"${ORT_VERSION}"-* 2>/dev/null | head -1)"; \
    if [ -z "${_v}" ]; then \
      _v="$(ls -d "${_src}"/onnxruntime-linux-x64-static_lib-* 2>/dev/null | head -1)"; \
    fi; \
    if [ -n "${_v}" ]; then cp -r "${_v}" "${HOME}/ragflow-native-libs/onnxruntime/static_lib/"; fi

RUN git config --global safe.directory "*" && ./build.sh --cpp

RUN --mount=type=cache,id=ragflow-go-server-gomod,target=/root/.cache/gomod,sharing=locked \
    --mount=type=cache,id=ragflow-go-server-gobuild,target=/root/.cache/gobuild \
    GOPROXY=${GOPROXY:-https://proxy.golang.org,direct} ./build.sh --go

FROM scratch
COPY --from=go-builder /ragflow/bin/ragflow_server /ragflow_server
