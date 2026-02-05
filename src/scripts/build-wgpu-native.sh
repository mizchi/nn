#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:-}"
if [[ -z "$OUTPUT_PATH" ]]; then
  echo "usage: $0 <output-stamp>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PKG_DIR/.." && pwd)"

if [[ "$OUTPUT_PATH" != /* ]]; then
  OUTPUT_PATH="$PKG_DIR/$OUTPUT_PATH"
fi

if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" submodule update --init --recursive deps/wgpu-native
fi

pushd "$REPO_DIR/deps/wgpu-native" >/dev/null
make lib-native-release
popd >/dev/null

mkdir -p "$(dirname "$OUTPUT_PATH")"
: > "$OUTPUT_PATH"
