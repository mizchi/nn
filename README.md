# mizchi/wgpu.mbt

Minimal WebGPU contract layer for MoonBit.

## Goals

- Common API for WebGPU (browser) and wgpu-native (native)
- Small, strict contract surface for 2D rendering and compute
- Backend implementations live in separate packages

## Status

- JS backend: contract only (NotImplemented).
- Native backend: minimal compute path implemented for MLP inference (wgpu-native).

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
Use `/?mode=train` for a single GPU train step vs CPU reference.
Use `/?mode=mnist&limit=1024&test_limit=10000&epochs=1&batch=128&lr=0.1&shuffle=1&init=he` to train MNIST in browser.
For benchmarking, add `bench=1` and size params
(`input`, `hidden`, `output`, `batch`, `warmup`, `iters`, `seed`).

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
just bench-cpu # run CPU loss benchmark (native)
just bench-gpu # run WebGPU loss benchmark (Playwright)
just wgpu-native-build # build wgpu-native (native target)
just serve     # build browser entry + serve repo root
just mnist-download # download MNIST (official URL + mirror fallback)
just mnist-train # run MNIST training (native, ~95% test acc)
just mnist-infer # run MNIST inference with saved weights (native)
just mnist-infer --json # emit JSON lines for inference
just mnist-train --json # emit JSON lines for training
just mnist-infer --backend cpu --bench # benchmark CPU inference
just mnist-infer --backend gpu --bench # benchmark GPU inference
just mnist-train --backend cpu --bench # benchmark CPU training
just mnist-train --backend gpu --bench # benchmark GPU training
just mnist-train --backend gpu --bench --bench-no-readback # GPU training without per-step readback
just mnist-train --epochs 5 --limit 2048 # quick training on a subset
```

## Benchmarks (MNIST subset)

Measured on 2026-02-05 with `limit=1024`, `epochs=5`, `batch=128`, `lr=0.1`.
GPU uses dataset segmentation (`segment_samples=42752`) because of the
`max_*_buffer_binding_size` (128MB) limit.

MoonBit (this repo):

```
# CPU
moon run --target native src/train -- --backend cpu --epochs 5 --limit 1024 --bench
bench: backend=cpu train_ms=11000
epoch 5 loss=1.0267586708068848 acc=0.7705078125
result: backend=cpu split=test loss=1.0727334022521973 acc=0.729200005531311

# GPU
moon run --target native src/train -- --backend gpu --epochs 5 --limit 1024 --bench
bench: backend=gpu train_ms=1000
epoch 5 loss=1.0264246463775635 acc=0.767578125
result: backend=gpu split=test loss=1.0608453750610352 acc=0.7218000292778015

# GPU (no readback)
moon run --target native src/train -- --backend gpu --epochs 5 --limit 1024 --bench --bench-no-readback
bench: backend=gpu train_ms=1000
result: backend=gpu split=test loss=1.0608453750610352 acc=0.7218000292778015
```

PyTorch baseline (~/sandbox/torch-mnist):

```
cd ~/sandbox/torch-mnist
uv run python main.py --epochs 5 --limit 1024 --device cpu
bench: backend=pytorch device=cpu train_ms=36
result: split=test loss=1.196183 acc=0.751400

uv run python main.py --epochs 5 --limit 1024 --device mps
bench: backend=pytorch device=mps train_ms=140
result: split=test loss=1.196183 acc=0.751400
```

Notes:
- These numbers are for a small subset and are not directly comparable across
  runtimes; they are included only for rough, local comparison.
- PyTorch uses the MNIST data from `data/mnist` in this repo by default.

## Benchmarks (MNIST full)

Measured on 2026-02-05 with full dataset, `epochs=20`, `batch=128`, `lr=0.1`.
GPU uses dataset segmentation (`segment_samples=42752`) because of the
`max_*_buffer_binding_size` (128MB) limit.

```
# GPU
moon run --target native src/train -- --backend gpu --bench
bench: backend=gpu train_ms=121000
result: backend=gpu split=test loss=0.15878579020500183 acc=0.953000009059906

# GPU (no readback)
moon run --target native src/train -- --backend gpu --bench --bench-no-readback
bench: backend=gpu train_ms=114000
result: backend=gpu split=test loss=0.15878579020500183 acc=0.953000009059906
```

PyTorch baseline (~/sandbox/torch-mnist):

```
cd ~/sandbox/torch-mnist
uv run python main.py --epochs 20 --device mps
bench: backend=pytorch device=mps train_ms=10282
result: split=test loss=0.079558 acc=0.976200
```

## Native (wgpu-native)

`mnist-infer` and `mnist-train --backend gpu` use the wgpu-native backend
(compute only). Build the native library once before running:

```bash
just wgpu-native-build
```
