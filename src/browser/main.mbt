///|
using @wgpu {
  type Device,
  type Buffer,
}

///|
extern "js" fn navigator_gpu() -> @core.Any =
  #| () => navigator.gpu

///|
extern "js" fn request_adapter_js(gpu : @core.Any) -> @js_async.Promise[@core.Any] =
  #| (gpu) => gpu.requestAdapter()

///|
extern "js" fn request_device_js(adapter : @core.Any) -> @js_async.Promise[@core.Any] =
  #| (adapter) => adapter.requestDevice()

///|
extern "js" fn create_storage_buffer_js(
  device : @core.Any,
  size : Int
) -> @core.Any =
  #| (device, size) => device.createBuffer({
  #|   size,
  #|   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  #| })

///|
extern "js" fn queue_write_f32_js(
  device : @core.Any,
  buffer : @core.Any,
  data : Array[Float]
) -> Unit =
  #| (device, buffer, data) => {
  #|   const f32 = new Float32Array(data)
  #|   device.queue.writeBuffer(buffer, 0, f32)
  #| }

///|
extern "js" fn queue_write_u32_js(
  device : @core.Any,
  buffer : @core.Any,
  data : Array[Int]
) -> Unit =
  #| (device, buffer, data) => {
  #|   const u32 = new Uint32Array(data)
  #|   device.queue.writeBuffer(buffer, 0, u32)
  #| }

///|
extern "js" fn bytes_to_f32_array_js(bytes : Bytes) -> Array[Float] =
  #| (bytes) => Array.from(
  #|   new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
  #| )

///|
extern "js" fn create_compute_pipeline_js(
  device : @core.Any,
  wgsl : String
) -> @core.Any =
  #| (device, wgsl) => device.createComputePipeline({
  #|   layout: "auto",
  #|   compute: {
  #|     module: device.createShaderModule({ code: wgsl }),
  #|     entryPoint: "main",
  #|   },
  #| })

///|
extern "js" fn create_bind_group_js(
  device : @core.Any,
  pipeline : @core.Any,
  buffers : Array[@core.Any]
) -> @core.Any =
  #| (device, pipeline, buffers) => device.createBindGroup({
  #|   layout: pipeline.getBindGroupLayout(0),
  #|   entries: buffers.map((buffer, i) => ({
  #|     binding: i,
  #|     resource: { buffer },
  #|   })),
  #| })

///|
extern "js" fn dispatch_compute_js(
  device : @core.Any,
  pipeline : @core.Any,
  bind_group : @core.Any,
  dispatch_x : Int
) -> Unit =
  #| (device, pipeline, bindGroup, dispatchX) => {
  #|   const encoder = device.createCommandEncoder()
  #|   const pass = encoder.beginComputePass()
  #|   pass.setPipeline(pipeline)
  #|   pass.setBindGroup(0, bindGroup)
  #|   pass.dispatchWorkgroups(dispatchX)
  #|   pass.end()
  #|   device.queue.submit([encoder.finish()])
  #| }

///|
extern "js" fn query_values_js() -> Array[Float] =
  #| () => {
  #|   const search = globalThis.location?.search ?? ""
  #|   if (!search) return []
  #|   const params = new URLSearchParams(search)
  #|   const raw = params.get("values")
  #|   if (!raw) return []
  #|   return raw
  #|     .split(",")
  #|     .map((v) => Number(v))
  #|     .filter((v) => Number.isFinite(v))
  #| }

///|
extern "js" fn query_repeat_js() -> Int =
  #| () => {
  #|   const search = globalThis.location?.search ?? ""
  #|   if (!search) return 1
  #|   const params = new URLSearchParams(search)
  #|   const raw = params.get("repeat")
  #|   const value = raw ? Number(raw) : 1
  #|   if (!Number.isFinite(value) || value <= 0) return 1
  #|   return Math.floor(value)
  #| }

///|
extern "js" fn query_seed_js() -> Int =
  #| () => {
  #|   const search = globalThis.location?.search ?? ""
  #|   if (!search) return -1
  #|   const params = new URLSearchParams(search)
  #|   const raw = params.get("seed")
  #|   const value = raw === null ? NaN : Number(raw)
  #|   if (!Number.isFinite(value)) return -1
  #|   return Math.floor(value)
  #| }

///|
extern "js" fn query_count_js() -> Int =
  #| () => {
  #|   const search = globalThis.location?.search ?? ""
  #|   if (!search) return 4
  #|   const params = new URLSearchParams(search)
  #|   const raw = params.get("count")
  #|   const value = raw === null ? NaN : Number(raw)
  #|   if (!Number.isFinite(value) || value <= 0) return 4
  #|   return Math.floor(value)
  #| }

///|
extern "js" fn query_offset_js() -> Int =
  #| () => {
  #|   const search = globalThis.location?.search ?? ""
  #|   if (!search) return 0
  #|   const params = new URLSearchParams(search)
  #|   const raw = params.get("offset")
  #|   const value = raw === null ? NaN : Number(raw)
  #|   if (!Number.isFinite(value)) return 0
  #|   return Math.floor(value)
  #| }

///|
extern "js" fn query_mode_js() -> String =
  #| () => {
  #|   const search = globalThis.location?.search ?? ""
  #|   if (!search) return ""
  #|   const params = new URLSearchParams(search)
  #|   return params.get("mode") ?? ""
  #| }

///|
extern "js" fn set_e2e_result_js(
  ok : Bool,
  expected : Array[Float],
  actual : Array[Float],
  reason : String?
) -> Unit =
  #| (ok, expected, actual, reason) => {
  #|   const result = { ok: !!ok, expected, actual, reason, done: true }
  #|   globalThis.__E2E_RESULT__ = result
  #|   const el = globalThis.document?.getElementById?.("status")
  #|   if (el) el.textContent = JSON.stringify(result, null, 2)
  #| }

fn is_null_or_undefined(value : @core.Any) -> Bool {
  @core.typeof_(value) == "undefined" || @core.is_null(value)
}

fn arrays_close(expected : Array[Float], actual : Array[Float]) -> Bool {
  if expected.length() != actual.length() {
    return false
  }
  let mut ok = true
  let mut i = 0
  while i < expected.length() {
    if !Float::is_close(expected[i], actual[i]) {
      ok = false
      break
    }
    i = i + 1
  }
  ok
}

fn arrays_close_eps(expected : Array[Float], actual : Array[Float], eps : Float) -> Bool {
  if expected.length() != actual.length() {
    return false
  }
  let mut ok = true
  let mut i = 0
  while i < expected.length() {
    let diff = (expected[i] - actual[i]).abs()
    if diff > eps {
      ok = false
      break
    }
    i = i + 1
  }
  ok
}

fn softmax_batch(
  logits : Array[Float],
  batch : Int,
  output : Int
) -> Array[Float] {
  let total = batch * output
  let probs = Array::make(total, Float::from_int(0))
  let mut b = 0
  while b < batch {
    let base = b * output
    let mut max_val = logits[base]
    let mut k = 1
    while k < output {
      let v = logits[base + k]
      if v > max_val {
        max_val = v
      }
      k = k + 1
    }
    let mut sum = Float::from_int(0)
    k = 0
    while k < output {
      let v = @math.expf(logits[base + k] - max_val)
      probs[base + k] = v
      sum = sum + v
      k = k + 1
    }
    k = 0
    while k < output {
      probs[base + k] = probs[base + k] / sum
      k = k + 1
    }
    b = b + 1
  }
  probs
}

fn loss_from_probs(
  probs : Array[Float],
  labels : Array[Int],
  output : Int
) -> Float {
  let eps = Float::from_int(1) / Float::from_int(1000000)
  let mut sum = Float::from_int(0)
  let mut b = 0
  while b < labels.length() {
    let label = labels[b]
    let p = probs[b * output + label] + eps
    sum = sum + -@math.lnf(p)
    b = b + 1
  }
  sum / Float::from_int(labels.length())
}

fn repeat_values(values : Array[Float], repeat : Int) -> Array[Float] {
  if repeat <= 1 || values.length() == 0 {
    return values
  }
  let total = values.length() * repeat
  Array::makei(total, fn(i) { values[i % values.length()] })
}

fn seeded_value(seed : Int, index : Int, offset : Int) -> Float {
  let raw = (index + seed + offset) % 23
  let centered = raw - 11
  Float::from_int(centered) / Float::from_int(10)
}

fn seeded_values(seed : Int, count : Int, offset : Int) -> Array[Float] {
  let total = if count <= 0 { 1 } else { count }
  Array::makei(total, fn(i) { seeded_value(seed, i, offset) })
}

///|
pub async fn run_mlp_loss() -> Unit {
  let gpu = navigator_gpu()
  if is_null_or_undefined(gpu) {
    set_e2e_result_js(false, [], [], Some("navigator.gpu is not available"))
    return
  }
  let adapter = request_adapter_js(gpu).wait()
  if is_null_or_undefined(adapter) {
    set_e2e_result_js(false, [], [], Some("requestAdapter returned null"))
    return
  }
  let device_any = request_device_js(adapter).wait()
  let device : Device = device_any.cast()

  let spec = match @nn.mlp_spec_new(4, 3, 2, 2) {
    Ok(s) => s
    Err(err) => {
      set_e2e_result_js(false, [], [], Some("spec error: " + err.to_string()))
      return
    }
  }

  let f = Float::from_int
  let inputs : Array[Float] = [
    f(1),
    f(0),
    f(0),
    f(1),
    f(0),
    f(1),
    f(1),
    f(0),
  ]
  let labels : Array[Int] = [0, 1]

  let params = match @nn.mlp_init_params_with_policy(
    spec,
    0,
    @nn.mlp_init_policy_deterministic,
  ) {
    Ok(p) => p
    Err(err) => {
      set_e2e_result_js(false, [], [], Some("init error: " + err.to_string()))
      return
    }
  }

  let forward = match @nn.mlp_forward(spec, params, inputs) {
    Ok(r) => r
    Err(err) => {
      set_e2e_result_js(false, [], [], Some("forward error: " + err.to_string()))
      return
    }
  }
  let expected_logits = forward.output
  let expected_probs = softmax_batch(
    expected_logits,
    spec.batch_size,
    spec.output_size,
  )
  let expected_loss = loss_from_probs(
    expected_probs,
    labels,
    spec.output_size,
  )
  let expected : Array[Float] = [expected_loss]
  expected.append(expected_probs[:])
  expected.append(expected_logits[:])

  let workgroup_size = 64
  let shader_plan = match @nn.mlp_shader_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      set_e2e_result_js(false, expected, [], Some(err.to_string()))
      return
    }
  }
  let loss_plan = match @nn.mlp_loss_shader_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      set_e2e_result_js(false, expected, [], Some(err.to_string()))
      return
    }
  }
  let buffer_plan = match @nn.mlp_plan_buffers(spec) {
    Ok(p) => p
    Err(err) => {
      set_e2e_result_js(false, expected, [], Some(err.to_string()))
      return
    }
  }
  let loss_buffers = match @nn.mlp_plan_loss_buffers(spec) {
    Ok(p) => p
    Err(err) => {
      set_e2e_result_js(false, expected, [], Some(err.to_string()))
      return
    }
  }
  let dispatch = match @nn.mlp_dispatch_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      set_e2e_result_js(false, expected, [], Some(err.to_string()))
      return
    }
  }
  let loss_dispatch = match @nn.mlp_loss_dispatch_x(spec, workgroup_size) {
    Ok(v) => v
    Err(err) => {
      set_e2e_result_js(false, expected, [], Some(err.to_string()))
      return
    }
  }
  let loss_reduce_dispatch = match @nn.mlp_loss_reduce_dispatch_x(
    spec,
    workgroup_size,
  ) {
    Ok(v) => v
    Err(err) => {
      set_e2e_result_js(false, expected, [], Some(err.to_string()))
      return
    }
  }

  let input_buf = create_storage_buffer_js(device_any, buffer_plan.input_bytes)
  let hidden_buf = create_storage_buffer_js(device_any, buffer_plan.hidden_bytes)
  let output_buf_any = create_storage_buffer_js(
    device_any,
    buffer_plan.output_bytes,
  )
  let output_buf : Buffer = output_buf_any.cast()
  let weight1_buf = create_storage_buffer_js(device_any, buffer_plan.weight1_bytes)
  let bias1_buf = create_storage_buffer_js(device_any, buffer_plan.bias1_bytes)
  let weight2_buf = create_storage_buffer_js(device_any, buffer_plan.weight2_bytes)
  let bias2_buf = create_storage_buffer_js(device_any, buffer_plan.bias2_bytes)
  let labels_buf = create_storage_buffer_js(device_any, loss_buffers.labels_bytes)
  let loss_buf_any = create_storage_buffer_js(device_any, loss_buffers.loss_bytes)
  let probs_buf_any = create_storage_buffer_js(device_any, loss_buffers.probs_bytes)
  let loss_sum_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.loss_sum_bytes,
  )
  let loss_sum_buf : Buffer = loss_sum_buf_any.cast()
  let probs_buf : Buffer = probs_buf_any.cast()

  queue_write_f32_js(device_any, input_buf, inputs)
  queue_write_f32_js(device_any, weight1_buf, params.weight1)
  queue_write_f32_js(device_any, bias1_buf, params.bias1)
  queue_write_f32_js(device_any, weight2_buf, params.weight2)
  queue_write_f32_js(device_any, bias2_buf, params.bias2)
  queue_write_u32_js(device_any, labels_buf, labels)

  let layer1_pipeline =
    create_compute_pipeline_js(device_any, shader_plan.layer1_wgsl)
  let layer2_pipeline =
    create_compute_pipeline_js(device_any, shader_plan.layer2_wgsl)
  let loss_pipeline =
    create_compute_pipeline_js(device_any, loss_plan.loss_wgsl)
  let loss_reduce_pipeline =
    create_compute_pipeline_js(device_any, loss_plan.loss_reduce_wgsl)

  let layer1_bind = create_bind_group_js(device_any, layer1_pipeline, [
    input_buf,
    weight1_buf,
    bias1_buf,
    hidden_buf,
  ])
  let layer2_bind = create_bind_group_js(device_any, layer2_pipeline, [
    hidden_buf,
    weight2_buf,
    bias2_buf,
    output_buf_any,
  ])
  let loss_bind = create_bind_group_js(device_any, loss_pipeline, [
    output_buf_any,
    labels_buf,
    loss_buf_any,
    probs_buf_any,
  ])
  let loss_reduce_bind = create_bind_group_js(device_any, loss_reduce_pipeline, [
    loss_buf_any,
    loss_sum_buf_any,
  ])

  dispatch_compute_js(
    device_any,
    layer1_pipeline,
    layer1_bind,
    dispatch.layer1_dispatch_x,
  )
  dispatch_compute_js(
    device_any,
    layer2_pipeline,
    layer2_bind,
    dispatch.layer2_dispatch_x,
  )
  dispatch_compute_js(device_any, loss_pipeline, loss_bind, loss_dispatch)
  dispatch_compute_js(
    device_any,
    loss_reduce_pipeline,
    loss_reduce_bind,
    loss_reduce_dispatch,
  )

  let loss_read = @web.device_read_buffer_bytes(
    device,
    loss_sum_buf,
    loss_buffers.loss_sum_bytes,
  )
  match loss_read {
    Ok(loss_bytes) => {
      let losses = bytes_to_f32_array_js(loss_bytes)
      let avg = losses[0] / Float::from_int(spec.batch_size)
      let probs_read = @web.device_read_buffer_bytes(
        device,
        probs_buf,
        loss_buffers.probs_bytes,
      )
      match probs_read {
        Ok(prob_bytes) => {
          let probs = bytes_to_f32_array_js(prob_bytes)
          let actual_probs = probs
          let actual_loss : Array[Float] = [avg]
          let eps = Float::from_int(1) / Float::from_int(10000)
          let ok_probs = arrays_close_eps(expected_probs, actual_probs, eps)
          let loss_diff = (expected_loss - actual_loss[0]).abs()
          let ok_loss = loss_diff < eps
          let logits_read = @web.device_read_buffer_bytes(
            device,
            output_buf,
            buffer_plan.output_bytes,
          )
          match logits_read {
            Ok(logit_bytes) => {
              let actual_logits = bytes_to_f32_array_js(logit_bytes)
              let ok_logits = arrays_close_eps(expected_logits, actual_logits, eps)
              let ok = ok_probs && ok_loss && ok_logits
              let actual : Array[Float] = actual_loss
              actual.append(actual_probs[:])
              actual.append(actual_logits[:])
              let reason =
                if ok {
                  None
                } else {
                  Some(
                    "loss/probs/logits mismatch: loss=" +
                    ok_loss.to_string() +
                    " diff=" +
                    loss_diff.to_string() +
                    " probs=" +
                    ok_probs.to_string() +
                    " logits=" +
                    ok_logits.to_string(),
                  )
                }
              set_e2e_result_js(ok, expected, actual, reason)
            }
            Err(err) =>
              set_e2e_result_js(false, expected, [], Some(err.to_string()))
          }
        }
        Err(err) =>
          set_e2e_result_js(false, expected, [], Some(err.to_string()))
      }
    }
    Err(err) =>
      set_e2e_result_js(false, expected, [], Some(err.to_string()))
  }
}

///|
pub async fn run_readback() -> Unit {
  let gpu = navigator_gpu()
  if is_null_or_undefined(gpu) {
    set_e2e_result_js(false, [], [], Some("navigator.gpu is not available"))
    return
  }
  let adapter = request_adapter_js(gpu).wait()
  if is_null_or_undefined(adapter) {
    set_e2e_result_js(false, [], [], Some("requestAdapter returned null"))
    return
  }
  let device_any = request_device_js(adapter).wait()
  let device : Device = device_any.cast()
  let f = Float::from_int
  let parsed = query_values_js()
  let repeat = query_repeat_js()
  let seed = query_seed_js()
  let count = query_count_js()
  let offset = query_offset_js()
  let default_values : Array[Float] = [
    f(5) / f(4),
    f(-5) / f(2),
    f(15) / f(4),
    f(0),
  ]
  let base =
    if parsed.length() == 0 {
      if seed >= 0 { seeded_values(seed, count, offset) } else { default_values }
    } else {
      parsed
    }
  let expected = repeat_values(base, repeat)
  let size = expected.length() * 4
  let buffer_any = create_storage_buffer_js(device_any, size)
  let buffer : Buffer = buffer_any.cast()
  queue_write_f32_js(device_any, buffer_any, expected)
  let read_result = @web.device_read_buffer_bytes(device, buffer, size)
  match read_result {
    Ok(bytes) => {
      let actual = bytes_to_f32_array_js(bytes)
      let ok = arrays_close(expected, actual)
      set_e2e_result_js(ok, expected, actual, None)
    }
    Err(err) =>
      set_e2e_result_js(false, expected, [], Some(err.to_string()))
  }
}

///|
fn main {
  if query_mode_js() == "loss" {
    @async.run_async_main(run_mlp_loss)
  } else {
    @async.run_async_main(run_readback)
  }
}
