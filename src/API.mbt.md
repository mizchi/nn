# API Documentation

This file contains executable doc tests using `mbt test` blocks.

## request_adapter_options_default

```mbt test
let opts = @wgpu.request_adapter_options_default()
inspect(opts.power_preference, content="Default")
inspect(opts.force_fallback_adapter, content="false")
```

## limits_default

```mbt test
let limits = @wgpu.limits_default()
inspect(limits.max_bind_groups, content="0")
inspect(limits.max_texture_dimension_2d, content="0")
```

## buffer_usage_or

```mbt test
let usage = @wgpu.buffer_usage_or(
  @wgpu.buffer_usage_copy_dst,
  @wgpu.buffer_usage_storage,
)
inspect(usage, content="136")
```

## Surface APIs

Surface-related APIs are provided by separate packages:

- `@web` for browser Canvas surfaces
- `@native` for native window surfaces

Auxiliary packages:

- `@nn` for minimal 2-layer MLP planning utilities
- `@glfw` for GLFW contract (native stub)

`@nn` also includes a CPU reference forward pass to validate sizes and outputs.

`@nn` provides byte conversion helpers for upload/readback and initialization policies
(`mlp_init_params_with_policy`). Parameters/inputs/outputs use `Float` (f32) to align
with GPU buffers.

`@nn` also includes MNIST IDX parsers, a CPU training loop,
`mlp_eval_from_bytes` for inference from serialized parameters, and
loss planning helpers (`mlp_loss_shader_plan`, `mlp_plan_loss_buffers`,
`mlp_loss_dispatch_x`) for GPU softmax + cross-entropy.

`@web` includes a minimal I/O helper (`queue_write_bytes` / async `device_read_buffer_bytes`)
to bridge Bytes into the core buffer API.

## Web I/O example

```mbt
/// pseudo (async function)
pub async fn readback_example(
  device : @wgpu.Device,
  queue : @wgpu.Queue,
  buffer : @wgpu.Buffer,
  size : Int,
  input : Bytes
) -> Result[Bytes, @wgpu.WgpuError] {
  match @web.queue_write_bytes(queue, buffer, 0, input) {
    Ok(_) => @web.device_read_buffer_bytes(device, buffer, size)
    Err(err) => Err(err)
  }
}
```

Browser entry sample: `mizchi/wgpu/browser` (calls WebGPU + `@web.device_read_buffer_bytes` and
publishes results to `window.__E2E_RESULT__`).

## Enum helpers

Enum constructors are not directly accessible across packages, so helper values/functions are provided.
The helper list is intentionally minimal.

```mbt test
inspect(@wgpu.power_preference_high_performance, content="HighPerformance")
inspect(@wgpu.texture_format_rgba8_unorm, content="Rgba8Unorm")
inspect(@wgpu.feature_other("foo"), content="Other(\"foo\")")
```

## Descriptor helpers

```mbt test
let usage = @wgpu.buffer_usage_or(@wgpu.buffer_usage_storage, @wgpu.buffer_usage_copy_dst)
let desc = @wgpu.buffer_descriptor(16, usage, false, None)
inspect(desc.size, content="16")
```
