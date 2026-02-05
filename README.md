# mizchi/wgpu.mbt

Minimal WebGPU contract layer for MoonBit.

## Goals

- Common API for WebGPU (browser) and wgpu-native (native)
- Small, strict contract surface for 2D rendering and compute
- Backend implementations live in separate packages

## Status

Contract layer only. All backend functions return `NotImplemented` for now.

## Packages

- `mizchi/wgpu` : core contract (no surface)
- `mizchi/wgpu/web` : browser surface API (Canvas)
- `mizchi/wgpu/native` : native surface API (window handle)
- `mizchi/wgpu/nn` : minimal 2-layer MLP planning + CPU reference
- `mizchi/wgpu/glfw` : GLFW contract (native stub, may be split out)
- `mizchi/wgpu/browser` : browser entry sample (WebGPU readback)
- `mizchi/wgpu/mnist` : MNIST loader (native)
- `mizchi/wgpu/train` : MNIST training entry (native)
- `mizchi/wgpu/infer` : MNIST inference entry (native)

## Web I/O note

`@web.device_read_buffer_bytes` is async. Buffers you read back from must include
`COPY_SRC` usage (the helper in `@nn` already plans output buffers with it).

Browser entry supports `/?mode=loss` to run a minimal GPU forward + softmax loss demo.

## Enum helpers

Enum constructors are read-only across packages, so `wgpu` exposes helper values/functions like
`texture_format_rgba8_unorm` and `feature_other("name")`. The helper list is intentionally minimal
and should grow only when the backend supports it.

## Quick Commands

```bash
just           # check + test
just fmt       # format code
just check     # type check
just test      # run tests
just e2e       # run Playwright e2e (builds browser entry)
just e2e-install # install Playwright Chromium
just mnist-download # download MNIST (official URL + mirror fallback)
just mnist-train # run MNIST training (native, ~95% test acc)
just mnist-infer # run MNIST inference with saved weights (native)
```
