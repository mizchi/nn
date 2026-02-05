# mizchi/wgpu

Minimal WebGPU contract layer for MoonBit.

```mbt test
let opts = @wgpu.request_adapter_options_default()
inspect(opts.force_fallback_adapter, content="false")
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

## MNIST

```
just mnist-download
just mnist-train
just mnist-infer
```

- `just mnist-download` tries the official URL first and falls back to a mirror
- `just mnist-train` trains a 784-128-10 MLP for 20 epochs (target ~95% test acc)
- `just mnist-infer` evaluates the saved weights (`data/mnist/mlp_784_128_10.bin`)

Examples:
```
just mnist-infer --limit 1000
just mnist-infer --weights data/mnist/mlp_784_128_10.bin
```
