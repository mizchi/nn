# mizchi/wgpu

Minimal WebGPU contract layer for MoonBit.

```mbt check
///|
test {
  let opts = @wgpu.request_adapter_options_default()
  inspect(opts.force_fallback_adapter, content="false")
}
```

## Packages

- core: `@wgpu`
- web surface: `@web`
- native surface: `@native`
- nn: `@nn` (planning + CPU reference, Float/f32)
- glfw: `@glfw`
- browser entry: `@browser` (WebGPU readback sample)
- mnist loader: `@mnist` (native)
- training entry: `@train` (native main)
- inference entry: `@infer` (native main)

## Web I/O note

`@web.device_read_buffer_bytes` is async. Ensure the source buffer includes
`COPY_SRC` usage for readback.

## E2E

```
just e2e
```

Browser query parameter (optional):
`/?values=1.25,-2.5,3.75,0`
`/?values=0.5,-1&repeat=3`
`/?seed=7&count=6&repeat=2`
`/?seed=7&count=4&offset=10`
`/?mode=loss` (loss + softmax + logits)
`/?mode=train` (single train step)
`/?mode=mnist&limit=1024&test_limit=10000&epochs=1&batch=128&lr=0.1&shuffle=1&init=he`
`/?mode=loss&bench=1&input=784&hidden=128&output=10&batch=128&warmup=20&iters=200`

Serve (for MNIST in browser):
```
just serve
```

Bench:
```
just bench-cpu
just bench-gpu
```

## MNIST

```
just mnist-download
just mnist-train
just mnist-infer
```

- `just mnist-download` tries the official URL first and falls back to a mirror
- `just mnist-train` trains a 784-128-10 MLP for 20 epochs (target ~95% test acc)
- `just mnist-train --backend gpu` runs training on wgpu-native (compute)
- `just mnist-train --epochs 5 --limit 2048` runs a short training on a subset
- `just mnist-infer` evaluates the saved weights (`data/mnist/mlp_784_128_10.bin`) with wgpu-native

Examples:
```
just mnist-infer --limit 1000
just mnist-infer --weights data/mnist/mlp_784_128_10.bin
just mnist-infer --batch 256
just mnist-infer --json
just mnist-train --json
just mnist-infer --backend cpu --bench
just mnist-infer --backend gpu --bench
just mnist-train --backend cpu --bench
just mnist-train --backend gpu --bench
just mnist-train --backend gpu --bench --bench-no-readback
just mnist-train --epochs 5 --limit 2048
```

## Native (wgpu-native)

Build the native library once before running native targets:

```
just wgpu-native-build
```
