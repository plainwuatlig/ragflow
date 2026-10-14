#!/usr/bin/env bash
# Build the ragflow-server image from this checkout and an upstream RAGFlow
# commit (the one the server follows), tagged with this checkout's commit.
#
#   rust/docker/build-server.sh <upstream-repo-dir> <upstream-commit> [image-name]
#
# Prints the image reference. The tree must be clean, so a tag names one commit.
set -euo pipefail

upstream_repo=${1:?usage: build-server.sh <upstream-repo-dir> <upstream-commit> [image-name]}
upstream_commit=${2:?upstream commit}
image=${3:-ragflow-server}

root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
if [ -n "$(git -C "$root" status --porcelain --untracked-files=no)" ]; then
  echo "build-server.sh: working tree is dirty" >&2
  exit 1
fi
revision=$(git -C "$root" rev-parse --short=12 HEAD)

ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
git -C "$upstream_repo" archive "$upstream_commit" build.sh go.mod go.sum internal conf | tar -x -C "$ctx"

docker build -f "$root/rust/docker/server.Dockerfile" \
  --build-context "upstream=$ctx" \
  --build-arg "REVISION=$revision" --build-arg "UPSTREAM_COMMIT=$upstream_commit" \
  -t "$image:$revision" "$root"
echo "$image:$revision"
