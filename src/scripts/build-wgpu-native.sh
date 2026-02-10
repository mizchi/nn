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
WGPU_NATIVE_DIR="$REPO_DIR/deps/wgpu-native"
WGPU_NATIVE_REV="ba4deb5d935652f40c7e051b15cbb5d097219941"

if [[ "$OUTPUT_PATH" != /* ]]; then
  OUTPUT_PATH="$PKG_DIR/$OUTPUT_PATH"
fi

if git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$REPO_DIR" submodule update --init --recursive deps/wgpu-native || true
fi

if [[ ! -f "$WGPU_NATIVE_DIR/Makefile" ]]; then
  rm -rf "$WGPU_NATIVE_DIR"
  mkdir -p "$REPO_DIR/deps"
  git clone "https://github.com/gfx-rs/wgpu-native" "$WGPU_NATIVE_DIR"
  git -C "$WGPU_NATIVE_DIR" checkout "$WGPU_NATIVE_REV"
  git -C "$WGPU_NATIVE_DIR" submodule update --init --recursive
fi

if [[ ! -f "$WGPU_NATIVE_DIR/Makefile" ]]; then
  echo "wgpu-native source is unavailable at $WGPU_NATIVE_DIR" >&2
  exit 1
fi

pushd "$WGPU_NATIVE_DIR" >/dev/null
make lib-native-release
popd >/dev/null

mkdir -p "$(dirname "$OUTPUT_PATH")"
: > "$OUTPUT_PATH"
