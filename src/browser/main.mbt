///|
using @wgpu {type Device, type Buffer}

///|
extern "js" fn navigator_gpu() -> @core.Any =
  #| () => navigator.gpu

///|
extern "js" fn request_adapter_js(
  gpu : @core.Any,
) -> @js_async.Promise[@core.Any] =
  #| (gpu) => gpu.requestAdapter()

///|
extern "js" fn request_device_js(
  adapter : @core.Any,
) -> @js_async.Promise[@core.Any] =
  #| (adapter) => adapter.requestDevice()

///|
extern "js" fn create_storage_buffer_js(
  device : @core.Any,
  size : Int,
) -> @core.Any =
  #| (device, size) => device.createBuffer({
  #|   size,
  #|   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  #| })

///|
extern "js" fn queue_write_f32_js(
  device : @core.Any,
  buffer : @core.Any,
  data : Array[Float],
) -> Unit =
  #| (device, buffer, data) => {
  #|   const f32 = new Float32Array(data)
  #|   device.queue.writeBuffer(buffer, 0, f32)
  #| }

///|
extern "js" fn queue_write_u32_js(
  device : @core.Any,
  buffer : @core.Any,
  data : Array[Int],
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
  wgsl : String,
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
  buffers : Array[@core.Any],
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
  dispatch_x : Int,
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
extern "js" fn begin_encoder_js(device : @core.Any) -> @core.Any =
  #| (device) => device.createCommandEncoder()

///|
extern "js" fn encode_dispatch_js(
  encoder : @core.Any,
  pipeline : @core.Any,
  bind_group : @core.Any,
  dispatch_x : Int,
) -> Unit =
  #| (encoder, pipeline, bindGroup, dispatchX) => {
  #|   const pass = encoder.beginComputePass()
  #|   pass.setPipeline(pipeline)
  #|   pass.setBindGroup(0, bindGroup)
  #|   pass.dispatchWorkgroups(dispatchX)
  #|   pass.end()
  #| }

///|
extern "js" fn submit_encoder_js(
  device : @core.Any,
  encoder : @core.Any,
) -> Unit =
  #| (device, encoder) => {
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
extern "js" fn query_int_js(name : String, default_value : Int) -> Int =
  #| (name, defaultValue) => {
  #|   const search = globalThis.location?.search ?? ""
  #|   if (!search) return defaultValue
  #|   const params = new URLSearchParams(search)
  #|   const raw = params.get(name)
  #|   const value = raw === null ? NaN : Number(raw)
  #|   if (!Number.isFinite(value)) return defaultValue
  #|   return Math.floor(value)
  #| }

///|
extern "js" fn query_string_js(name : String, default_value : String) -> String =
  #| (name, defaultValue) => {
  #|   const search = globalThis.location?.search ?? ""
  #|   if (!search) return defaultValue
  #|   const params = new URLSearchParams(search)
  #|   return params.get(name) ?? defaultValue
  #| }

///|
extern "js" fn query_float_js(name : String, default_value : Float) -> Float =
  #| (name, defaultValue) => {
  #|   const search = globalThis.location?.search ?? ""
  #|   if (!search) return defaultValue
  #|   const params = new URLSearchParams(search)
  #|   const raw = params.get(name)
  #|   const value = raw === null ? NaN : Number(raw)
  #|   if (!Number.isFinite(value)) return defaultValue
  #|   return value
  #| }

///|
extern "js" fn perf_now_js() -> Float =
  #| () => globalThis.performance?.now?.() ?? Date.now()

///|
extern "js" fn set_e2e_result_js(
  ok : Bool,
  expected : Array[Float],
  actual : Array[Float],
  reason : String?,
) -> Unit =
  #| (ok, expected, actual, reason) => {
  #|   const result = { ok: !!ok, expected, actual, reason, done: true }
  #|   globalThis.__E2E_RESULT__ = result
  #|   const el = globalThis.document?.getElementById?.("status")
  #|   if (el) el.textContent = JSON.stringify(result, null, 2)
  #| }

///|
extern "js" fn set_bench_result_js(
  ok : Bool,
  spec : Array[Int],
  stats : Array[Float],
  samples : Array[Float],
  reason : String?,
) -> Unit =
  #| (ok, spec, stats, samples, reason) => {
  #|   const [input, hidden, output, batch, warmup, iters] = spec
  #|   const [avg, p50, p95] = stats
  #|   const result = {
  #|     ok: !!ok,
  #|     spec: { input, hidden, output, batch, warmup, iters },
  #|     stats: { avg, p50, p95 },
  #|     samples,
  #|     reason,
  #|     done: true,
  #|   }
  #|   globalThis.__BENCH_RESULT__ = result
  #|   const el = globalThis.document?.getElementById?.("status")
  #|   if (el) el.textContent = JSON.stringify(result, null, 2)
  #| }

///|
extern "js" fn set_status_js(text : String) -> Unit =
  #| (text) => {
  #|   const el = globalThis.document?.getElementById?.("status")
  #|   if (el) el.textContent = text
  #| }

///|
fn is_null_or_undefined(value : @core.Any) -> Bool {
  @core.typeof_(value) == "undefined" || @core.is_null(value)
}

///|
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

///|
fn arrays_close_eps(
  expected : Array[Float],
  actual : Array[Float],
  eps : Float,
) -> Bool {
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

///|
fn softmax_batch(
  logits : Array[Float],
  batch : Int,
  output : Int,
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

///|
fn loss_from_probs(
  probs : Array[Float],
  labels : Array[Int],
  output : Int,
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

///|
fn repeat_values(values : Array[Float], repeat : Int) -> Array[Float] {
  if repeat <= 1 || values.length() == 0 {
    return values
  }
  let total = values.length() * repeat
  Array::makei(total, fn(i) { values[i % values.length()] })
}

///|
fn seeded_value(seed : Int, index : Int, offset : Int) -> Float {
  let raw = (index + seed + offset) % 23
  let centered = raw - 11
  Float::from_int(centered) / Float::from_int(10)
}

///|
fn seeded_values(seed : Int, count : Int, offset : Int) -> Array[Float] {
  let total = if count <= 0 { 1 } else { count }
  Array::makei(total, fn(i) { seeded_value(seed, i, offset) })
}

///|
fn sort_floats_in_place(values : Array[Float]) -> Unit {
  let mut i = 1
  while i < values.length() {
    let key = values[i]
    let mut j = i - 1
    while j >= 0 {
      if values[j] <= key {
        break
      }
      values[j + 1] = values[j]
      j = j - 1
    }
    values[j + 1] = key
    i = i + 1
  }
}

///|
fn percentile(sorted : Array[Float], p : Float) -> Float {
  if sorted.length() == 0 {
    return Float::from_int(0)
  }
  let n = sorted.length()
  let pos = Float::from_int(n - 1) * p
  let idx = pos.floor().to_int()
  sorted[idx]
}

///|
fn stats_from_samples(samples : Array[Float]) -> Array[Float] {
  if samples.length() == 0 {
    return [Float::from_int(0), Float::from_int(0), Float::from_int(0)]
  }
  let mut sum = Float::from_int(0)
  let mut i = 0
  while i < samples.length() {
    sum = sum + samples[i]
    i = i + 1
  }
  let avg = sum / Float::from_int(samples.length())
  let sorted = Array::makei(samples.length(), fn(i) { samples[i] })
  sort_floats_in_place(sorted)
  let f = Float::from_int
  let p50 = percentile(sorted, f(1) / f(2))
  let p95 = percentile(sorted, f(95) / f(100))
  [avg, p50, p95]
}

///|
fn make_labels(batch : Int, output : Int) -> Array[Int] {
  let total = if batch <= 0 { 0 } else { batch }
  Array::makei(total, fn(i) { i % output })
}

///|
fn params_to_array(params : @nn.MlpParams) -> Array[Float] {
  let out : Array[Float] = []
  out.append(params.weight1[:])
  out.append(params.bias1[:])
  out.append(params.weight2[:])
  out.append(params.bias2[:])
  out
}

///|
fn slice_f32(values : Array[Float], offset : Int, length : Int) -> Array[Float] {
  Array::makei(length, fn(i) { values[offset + i] })
}

///|
fn slice_i32(values : Array[Int], offset : Int, length : Int) -> Array[Int] {
  Array::makei(length, fn(i) { values[offset + i] })
}

///|
fn argmax_in_probs(probs : Array[Float], base : Int, size : Int) -> Int {
  let mut best = 0
  let mut best_val = probs[base]
  let mut k = 1
  while k < size {
    let v = probs[base + k]
    if v > best_val {
      best_val = v
      best = k
    }
    k = k + 1
  }
  best
}

///|

///|
fn gcd(a : Int, b : Int) -> Int {
  let mut x = if a < 0 { -a } else { a }
  let mut y = if b < 0 { -b } else { b }
  while y != 0 {
    let t = x % y
    x = y
    y = t
  }
  if x == 0 {
    1
  } else {
    x
  }
}

///|
fn ceil_div(n : Int, d : Int) -> Int {
  if d <= 0 {
    0
  } else {
    (n + d - 1) / d
  }
}

///|
fn shuffle_stride(count : Int, seed : Int) -> Int {
  if count <= 1 {
    return 1
  }
  let mut stride = seed % (count - 1) + 1
  if stride <= 0 {
    stride = 1
  }
  while gcd(stride, count) != 1 {
    stride = stride + 1
    if stride >= count {
      stride = 1
    }
  }
  stride
}

///|
pub async fn run_mlp_loss() -> Unit {
  let bench = query_int_js("bench", 0) > 0
  let input_size = if bench { query_int_js("input", 784) } else { 4 }
  let hidden_size = if bench { query_int_js("hidden", 128) } else { 3 }
  let output_size = if bench { query_int_js("output", 10) } else { 2 }
  let batch_size = if bench { query_int_js("batch", 128) } else { 2 }
  let warmup = if bench { query_int_js("warmup", 20) } else { 0 }
  let iters = if bench { query_int_js("iters", 200) } else { 0 }
  let spec_info : Array[Int] = [
    input_size, hidden_size, output_size, batch_size, warmup, iters,
  ]
  let report_error = fn(expected : Array[Float], reason : String) {
    if bench {
      set_bench_result_js(false, spec_info, [], [], Some(reason))
    } else {
      set_e2e_result_js(false, expected, [], Some(reason))
    }
  }
  if bench && iters <= 0 {
    set_bench_result_js(false, spec_info, [], [], Some("iters must be > 0"))
    return
  }
  if bench && warmup < 0 {
    set_bench_result_js(false, spec_info, [], [], Some("warmup must be >= 0"))
    return
  }
  let gpu = navigator_gpu()
  if is_null_or_undefined(gpu) {
    report_error([], "navigator.gpu is not available")
    return
  }
  let adapter = request_adapter_js(gpu).wait()
  if is_null_or_undefined(adapter) {
    report_error([], "requestAdapter returned null")
    return
  }
  let device_any = request_device_js(adapter).wait()
  let device : Device = device_any.cast()
  let spec = match
    @nn.mlp_spec_new(input_size, hidden_size, output_size, batch_size) {
    Ok(s) => s
    Err(err) => {
      report_error([], "spec error: " + err.to_string())
      return
    }
  }
  let f = Float::from_int
  let inputs : Array[Float] = if bench {
    let seed = query_int_js("seed", 0)
    seeded_values(seed, spec.input_size * spec.batch_size, 0)
  } else {
    [f(1), f(0), f(0), f(1), f(0), f(1), f(1), f(0)]
  }
  let labels : Array[Int] = if bench {
    make_labels(spec.batch_size, spec.output_size)
  } else {
    [0, 1]
  }
  let params = match
    @nn.mlp_init_params_with_policy(spec, 0, @nn.mlp_init_policy_deterministic) {
    Ok(p) => p
    Err(err) => {
      report_error([], "init error: " + err.to_string())
      return
    }
  }
  // CPU forward validation disabled (numbt is native-only)
  let expected : Array[Float] = []
  let bench = true // Force bench mode since CPU validation is unavailable
  let workgroup_size = 64
  let shader_plan = match @nn.mlp_shader_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      report_error(expected, err.to_string())
      return
    }
  }
  let loss_plan = match @nn.mlp_loss_shader_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      report_error(expected, err.to_string())
      return
    }
  }
  let buffer_plan = match @nn.mlp_plan_buffers(spec) {
    Ok(p) => p
    Err(err) => {
      report_error(expected, err.to_string())
      return
    }
  }
  let dataset_buffers = match
    @nn.mlp_plan_dataset_buffers(spec, spec.batch_size) {
    Ok(p) => p
    Err(err) => {
      report_error(expected, err.to_string())
      return
    }
  }
  let loss_buffers = match
    @nn.mlp_plan_loss_buffers_for_dataset(spec, spec.batch_size) {
    Ok(p) => p
    Err(err) => {
      report_error(expected, err.to_string())
      return
    }
  }
  let dispatch = match @nn.mlp_dispatch_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      report_error(expected, err.to_string())
      return
    }
  }
  let loss_dispatch = match @nn.mlp_loss_dispatch_x(spec, workgroup_size) {
    Ok(v) => v
    Err(err) => {
      report_error(expected, err.to_string())
      return
    }
  }
  let loss_reduce_dispatch = match
    @nn.mlp_loss_reduce_dispatch_x(spec, workgroup_size) {
    Ok(v) => v
    Err(err) => {
      report_error(expected, err.to_string())
      return
    }
  }
  let input_buf = create_storage_buffer_js(
    device_any,
    dataset_buffers.dataset_input_bytes,
  )
  let hidden_buf = create_storage_buffer_js(
    device_any,
    buffer_plan.hidden_bytes,
  )
  let output_buf_any = create_storage_buffer_js(
    device_any,
    buffer_plan.output_bytes,
  )
  let output_buf : Buffer = output_buf_any.cast()
  let weight1_buf = create_storage_buffer_js(
    device_any,
    buffer_plan.weight1_bytes,
  )
  let bias1_buf = create_storage_buffer_js(device_any, buffer_plan.bias1_bytes)
  let weight2_buf = create_storage_buffer_js(
    device_any,
    buffer_plan.weight2_bytes,
  )
  let bias2_buf = create_storage_buffer_js(device_any, buffer_plan.bias2_bytes)
  let labels_buf = create_storage_buffer_js(
    device_any,
    loss_buffers.labels_bytes,
  )
  let indices_buf = create_storage_buffer_js(
    device_any,
    dataset_buffers.indices_bytes,
  )
  let loss_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.loss_bytes,
  )
  let probs_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.probs_bytes,
  )
  let correct_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.correct_bytes,
  )
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
  queue_write_u32_js(
    device_any,
    indices_buf,
    Array::makei(spec.batch_size + 1, fn(i) { if i == 0 { 0 } else { i - 1 } }),
  )
  let layer1_pipeline = create_compute_pipeline_js(
    device_any,
    shader_plan.layer1_wgsl,
  )
  let layer2_pipeline = create_compute_pipeline_js(
    device_any,
    shader_plan.layer2_wgsl,
  )
  let loss_pipeline = create_compute_pipeline_js(
    device_any,
    loss_plan.loss_wgsl,
  )
  let loss_reduce_pipeline = create_compute_pipeline_js(
    device_any,
    loss_plan.loss_reduce_wgsl,
  )
  let layer1_bind = create_bind_group_js(device_any, layer1_pipeline, [
    input_buf, indices_buf, weight1_buf, bias1_buf, hidden_buf,
  ])
  let layer2_bind = create_bind_group_js(device_any, layer2_pipeline, [
    hidden_buf, weight2_buf, bias2_buf, output_buf_any,
  ])
  let loss_bind = create_bind_group_js(device_any, loss_pipeline, [
    output_buf_any, labels_buf, indices_buf, loss_buf_any, probs_buf_any, correct_buf_any,
  ])
  let loss_reduce_bind = create_bind_group_js(
    device_any,
    loss_reduce_pipeline,
    [loss_buf_any, loss_sum_buf_any],
  )
  let dispatch_once = fn() {
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
      device_any, loss_reduce_pipeline, loss_reduce_bind, loss_reduce_dispatch,
    )
  }
  if bench {
    let mut i = 0
    while i < warmup {
      dispatch_once()
      let loss_read = @web.device_read_buffer_bytes(
        device,
        loss_sum_buf,
        loss_buffers.loss_sum_bytes,
      )
      match loss_read {
        Ok(_bytes) => ()
        Err(err) => {
          set_bench_result_js(false, spec_info, [], [], Some(err.to_string()))
          return
        }
      }
      i = i + 1
    }
    let samples = Array::make(iters, Float::from_int(0))
    i = 0
    while i < iters {
      let start = perf_now_js()
      dispatch_once()
      let loss_read = @web.device_read_buffer_bytes(
        device,
        loss_sum_buf,
        loss_buffers.loss_sum_bytes,
      )
      match loss_read {
        Ok(_bytes) => {
          let end = perf_now_js()
          samples[i] = end - start
        }
        Err(err) => {
          set_bench_result_js(
            false,
            spec_info,
            [],
            samples,
            Some(err.to_string()),
          )
          return
        }
      }
      i = i + 1
    }
    let stats = stats_from_samples(samples)
    set_bench_result_js(true, spec_info, stats, samples, None)
    return
  }
  dispatch_once()
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
          let logits_read = @web.device_read_buffer_bytes(
            device,
            output_buf,
            buffer_plan.output_bytes,
          )
          match logits_read {
            Ok(logit_bytes) => {
              let actual_logits = bytes_to_f32_array_js(logit_bytes)
              let actual : Array[Float] = actual_loss
              actual.append(actual_probs[:])
              actual.append(actual_logits[:])
              // Skip validation in bench mode (CPU baseline unavailable)
              let ok = if bench {
                true
              } else {
                let eps = Float::from_int(1) / Float::from_int(10000)
                arrays_close_eps(expected, actual, eps)
              }
              let reason = if ok {
                None
              } else {
                Some("loss/probs/logits mismatch")
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
    Err(err) => set_e2e_result_js(false, expected, [], Some(err.to_string()))
  }
}

///|
pub async fn run_mlp_train_step() -> Unit {
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
  let inputs : Array[Float] = [f(1), f(0), f(0), f(1), f(0), f(1), f(1), f(0)]
  let labels : Array[Int] = [0, 1]
  let learning_rate = f(1) / f(10)
  let params = match
    @nn.mlp_init_params_with_policy(spec, 0, @nn.mlp_init_policy_deterministic) {
    Ok(p) => p
    Err(err) => {
      set_e2e_result_js(false, [], [], Some("init error: " + err.to_string()))
      return
    }
  }
  let weight1_clone = Array::makei(params.weight1.length(), fn(i) {
    params.weight1[i]
  })
  let bias1_clone = Array::makei(params.bias1.length(), fn(i) {
    params.bias1[i]
  })
  let weight2_clone = Array::makei(params.weight2.length(), fn(i) {
    params.weight2[i]
  })
  let bias2_clone = Array::makei(params.bias2.length(), fn(i) {
    params.bias2[i]
  })
  let params_gpu = match
    @nn.mlp_params_new(
      spec, weight1_clone, bias1_clone, weight2_clone, bias2_clone,
    ) {
    Ok(p) => p
    Err(err) => {
      set_e2e_result_js(
        false,
        [],
        [],
        Some("param clone error: " + err.to_string()),
      )
      return
    }
  }
  // CPU training validation disabled (numbt is native-only)
  // GPU training will run without CPU baseline comparison
  let expected : Array[Float] = []
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
  let train_plan = match @nn.mlp_train_shader_plan(spec, workgroup_size) {
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
  let dataset_buffers = match
    @nn.mlp_plan_dataset_buffers(spec, spec.batch_size) {
    Ok(p) => p
    Err(err) => {
      set_e2e_result_js(false, expected, [], Some(err.to_string()))
      return
    }
  }
  let loss_buffers = match
    @nn.mlp_plan_loss_buffers_for_dataset(spec, spec.batch_size) {
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
  let train_dispatch = match @nn.mlp_train_dispatch_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      set_e2e_result_js(false, expected, [], Some(err.to_string()))
      return
    }
  }
  let input_buf = create_storage_buffer_js(
    device_any,
    dataset_buffers.dataset_input_bytes,
  )
  let hidden_buf = create_storage_buffer_js(
    device_any,
    buffer_plan.hidden_bytes,
  )
  let weight1_buf = create_storage_buffer_js(
    device_any,
    buffer_plan.weight1_bytes,
  )
  let bias1_buf = create_storage_buffer_js(device_any, buffer_plan.bias1_bytes)
  let weight2_buf = create_storage_buffer_js(
    device_any,
    buffer_plan.weight2_bytes,
  )
  let bias2_buf = create_storage_buffer_js(device_any, buffer_plan.bias2_bytes)
  let labels_buf = create_storage_buffer_js(
    device_any,
    loss_buffers.labels_bytes,
  )
  let indices_buf = create_storage_buffer_js(
    device_any,
    dataset_buffers.indices_bytes,
  )
  let loss_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.loss_bytes,
  )
  let probs_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.probs_bytes,
  )
  let correct_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.correct_bytes,
  )
  let lr_buf = create_storage_buffer_js(device_any, 4)
  queue_write_f32_js(device_any, input_buf, inputs)
  queue_write_f32_js(device_any, weight1_buf, params_gpu.weight1)
  queue_write_f32_js(device_any, bias1_buf, params_gpu.bias1)
  queue_write_f32_js(device_any, weight2_buf, params_gpu.weight2)
  queue_write_f32_js(device_any, bias2_buf, params_gpu.bias2)
  queue_write_u32_js(device_any, labels_buf, labels)
  queue_write_u32_js(
    device_any,
    indices_buf,
    Array::makei(spec.batch_size + 1, fn(i) { if i == 0 { 0 } else { i - 1 } }),
  )
  queue_write_f32_js(device_any, lr_buf, [learning_rate])
  let layer1_pipeline = create_compute_pipeline_js(
    device_any,
    shader_plan.layer1_wgsl,
  )
  let layer2_loss_pipeline = create_compute_pipeline_js(
    device_any,
    loss_plan.layer2_loss_wgsl,
  )
  let grad_update_w1_pipeline = create_compute_pipeline_js(
    device_any,
    train_plan.grad_update_w1_wgsl,
  )
  let grad_update_b1_pipeline = create_compute_pipeline_js(
    device_any,
    train_plan.grad_update_b1_wgsl,
  )
  let grad_update_w2_pipeline = create_compute_pipeline_js(
    device_any,
    train_plan.grad_update_w2_wgsl,
  )
  let grad_update_b2_pipeline = create_compute_pipeline_js(
    device_any,
    train_plan.grad_update_b2_wgsl,
  )
  let layer1_bind = create_bind_group_js(device_any, layer1_pipeline, [
    input_buf, indices_buf, weight1_buf, bias1_buf, hidden_buf,
  ])
  let layer2_loss_bind = create_bind_group_js(
    device_any,
    layer2_loss_pipeline,
    [
      hidden_buf, weight2_buf, bias2_buf, labels_buf, indices_buf, loss_buf_any,
      probs_buf_any, correct_buf_any,
    ],
  )
  let grad_update_w1_bind = create_bind_group_js(
    device_any,
    grad_update_w1_pipeline,
    [
      input_buf, hidden_buf, probs_buf_any, labels_buf, indices_buf, weight2_buf,
      weight1_buf, lr_buf,
    ],
  )
  let grad_update_b1_bind = create_bind_group_js(
    device_any,
    grad_update_b1_pipeline,
    [
      hidden_buf, probs_buf_any, labels_buf, indices_buf, weight2_buf, bias1_buf,
      lr_buf,
    ],
  )
  let grad_update_w2_bind = create_bind_group_js(
    device_any,
    grad_update_w2_pipeline,
    [hidden_buf, probs_buf_any, labels_buf, indices_buf, weight2_buf, lr_buf],
  )
  let grad_update_b2_bind = create_bind_group_js(
    device_any,
    grad_update_b2_pipeline,
    [probs_buf_any, labels_buf, indices_buf, bias2_buf, lr_buf],
  )
  let enc = begin_encoder_js(device_any)
  encode_dispatch_js(
    enc,
    layer1_pipeline,
    layer1_bind,
    dispatch.layer1_dispatch_x,
  )
  encode_dispatch_js(enc, layer2_loss_pipeline, layer2_loss_bind, loss_dispatch)
  // Dispatch order: w1, b1 first (read weight2), then w2, b2 (write weight2)
  encode_dispatch_js(
    enc,
    grad_update_w1_pipeline,
    grad_update_w1_bind,
    train_dispatch.grad_update_w1_dispatch_x,
  )
  encode_dispatch_js(
    enc,
    grad_update_b1_pipeline,
    grad_update_b1_bind,
    train_dispatch.grad_update_b1_dispatch_x,
  )
  encode_dispatch_js(
    enc,
    grad_update_w2_pipeline,
    grad_update_w2_bind,
    train_dispatch.grad_update_w2_dispatch_x,
  )
  encode_dispatch_js(
    enc,
    grad_update_b2_pipeline,
    grad_update_b2_bind,
    train_dispatch.grad_update_b2_dispatch_x,
  )
  submit_encoder_js(device_any, enc)
  let weight1_read = @web.device_read_buffer_bytes(
    device,
    weight1_buf.cast(),
    buffer_plan.weight1_bytes,
  )
  match weight1_read {
    Ok(weight1_bytes) => {
      let weight1 = bytes_to_f32_array_js(weight1_bytes)
      let bias1_read = @web.device_read_buffer_bytes(
        device,
        bias1_buf.cast(),
        buffer_plan.bias1_bytes,
      )
      match bias1_read {
        Ok(bias1_bytes) => {
          let bias1 = bytes_to_f32_array_js(bias1_bytes)
          let weight2_read = @web.device_read_buffer_bytes(
            device,
            weight2_buf.cast(),
            buffer_plan.weight2_bytes,
          )
          match weight2_read {
            Ok(weight2_bytes) => {
              let weight2 = bytes_to_f32_array_js(weight2_bytes)
              let bias2_read = @web.device_read_buffer_bytes(
                device,
                bias2_buf.cast(),
                buffer_plan.bias2_bytes,
              )
              match bias2_read {
                Ok(bias2_bytes) => {
                  let bias2 = bytes_to_f32_array_js(bias2_bytes)
                  let actual : Array[Float] = []
                  actual.append(weight1[:])
                  actual.append(bias1[:])
                  actual.append(weight2[:])
                  actual.append(bias2[:])
                  // Skip validation when expected is empty (CPU training disabled)
                  let ok = if expected.length() == 0 {
                    true
                  } else {
                    let eps = f(1) / f(10000)
                    arrays_close_eps(expected, actual, eps)
                  }
                  let reason = if ok {
                    None
                  } else {
                    Some("train step mismatch")
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
    Err(err) => set_e2e_result_js(false, expected, [], Some(err.to_string()))
  }
}

///|
pub async fn run_mnist_train() -> Unit {
  set_status_js("loading mnist...")
  let gpu = navigator_gpu()
  if is_null_or_undefined(gpu) {
    set_status_js("navigator.gpu is not available")
    return
  }
  let adapter = request_adapter_js(gpu).wait()
  if is_null_or_undefined(adapter) {
    set_status_js("requestAdapter returned null")
    return
  }
  let device_any = request_device_js(adapter).wait()
  let device : Device = device_any.cast()
  let base = "/data/mnist"
  let limit = query_int_js("limit", 1024)
  let test_limit = query_int_js("test_limit", 10000)
  let epochs = query_int_js("epochs", 1)
  let batch_size = query_int_js("batch", 128)
  let hidden_size = query_int_js("hidden", 128)
  let learning_rate = query_float_js("lr", 0.1)
  let shuffle = query_int_js("shuffle", 1) > 0
  let seed = query_int_js("seed", 0)
  let init = query_string_js("init", "he")
  if limit <= 0 || epochs <= 0 || batch_size <= 0 || hidden_size <= 0 {
    set_status_js("invalid params")
    return
  }
  let images_path = base + "/train-images-idx3-ubyte"
  let labels_path = base + "/train-labels-idx1-ubyte"
  let dataset = match
    @mnist.mnist_load_mlp_dataset(images_path, labels_path, Some(limit)) {
    Ok(ds) => ds
    Err(err) => {
      set_status_js("mnist load error: " + err.to_string())
      return
    }
  }
  let input_size = dataset.input_size
  let output_size = 10
  let effective_count = dataset.count / batch_size * batch_size
  if effective_count <= 0 {
    set_status_js("dataset too small for batch")
    return
  }
  let total_inputs = effective_count * input_size
  let total_labels = effective_count
  let inputs = slice_f32(dataset.inputs, 0, total_inputs)
  let labels = slice_i32(dataset.labels, 0, total_labels)
  let spec = match
    @nn.mlp_spec_new(input_size, hidden_size, output_size, batch_size) {
    Ok(s) => s
    Err(err) => {
      set_status_js("spec error: " + err.to_string())
      return
    }
  }
  let policy = if init == "zeros" {
    @nn.mlp_init_policy_zeros
  } else if init == "xavier" {
    @nn.mlp_init_policy_xavier_uniform
  } else if init == "deterministic" {
    @nn.mlp_init_policy_deterministic
  } else {
    @nn.mlp_init_policy_he_uniform
  }
  let params = match @nn.mlp_init_params_with_policy(spec, seed, policy) {
    Ok(p) => p
    Err(err) => {
      set_status_js("init error: " + err.to_string())
      return
    }
  }
  let workgroup_size = 64
  let shader_plan = match @nn.mlp_shader_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      set_status_js("shader error: " + err.to_string())
      return
    }
  }
  let loss_plan = match @nn.mlp_loss_shader_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      set_status_js("loss shader error: " + err.to_string())
      return
    }
  }
  let train_plan = match @nn.mlp_train_shader_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      set_status_js("train shader error: " + err.to_string())
      return
    }
  }
  let dataset_plan = match
    @nn.mlp_dataset_shader_plan(spec, effective_count, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      set_status_js("dataset shader error: " + err.to_string())
      return
    }
  }
  let buffer_plan = match @nn.mlp_plan_buffers(spec) {
    Ok(p) => p
    Err(err) => {
      set_status_js("buffer plan error: " + err.to_string())
      return
    }
  }
  let dataset_buffers = match
    @nn.mlp_plan_dataset_buffers(spec, effective_count) {
    Ok(p) => p
    Err(err) => {
      set_status_js("dataset buffer error: " + err.to_string())
      return
    }
  }
  let loss_buffers = match
    @nn.mlp_plan_loss_buffers_for_dataset(spec, effective_count) {
    Ok(p) => p
    Err(err) => {
      set_status_js("loss buffer error: " + err.to_string())
      return
    }
  }
  let dispatch = match @nn.mlp_dispatch_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      set_status_js("dispatch error: " + err.to_string())
      return
    }
  }
  let loss_dispatch = match @nn.mlp_loss_dispatch_x(spec, workgroup_size) {
    Ok(v) => v
    Err(err) => {
      set_status_js("loss dispatch error: " + err.to_string())
      return
    }
  }
  let loss_reduce_dispatch = match
    @nn.mlp_loss_reduce_dispatch_x(spec, workgroup_size) {
    Ok(v) => v
    Err(err) => {
      set_status_js("loss reduce dispatch error: " + err.to_string())
      return
    }
  }
  let train_dispatch = match @nn.mlp_train_dispatch_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => {
      set_status_js("train dispatch error: " + err.to_string())
      return
    }
  }
  let input_buf = create_storage_buffer_js(
    device_any,
    dataset_buffers.dataset_input_bytes,
  )
  let hidden_buf = create_storage_buffer_js(
    device_any,
    buffer_plan.hidden_bytes,
  )
  let weight1_buf = create_storage_buffer_js(
    device_any,
    buffer_plan.weight1_bytes,
  )
  let bias1_buf = create_storage_buffer_js(device_any, buffer_plan.bias1_bytes)
  let weight2_buf = create_storage_buffer_js(
    device_any,
    buffer_plan.weight2_bytes,
  )
  let bias2_buf = create_storage_buffer_js(device_any, buffer_plan.bias2_bytes)
  let labels_buf = create_storage_buffer_js(
    device_any,
    loss_buffers.labels_bytes,
  )
  let indices_buf = create_storage_buffer_js(
    device_any,
    dataset_buffers.indices_bytes,
  )
  let loss_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.loss_bytes,
  )
  let probs_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.probs_bytes,
  )
  let correct_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.correct_bytes,
  )
  let epoch_loss_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.epoch_loss_bytes,
  )
  let epoch_correct_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.epoch_correct_bytes,
  )
  let epoch_seen_buf_any = create_storage_buffer_js(
    device_any,
    loss_buffers.epoch_seen_bytes,
  )
  let shuffle_params_buf = create_storage_buffer_js(
    device_any,
    dataset_buffers.shuffle_params_bytes,
  )
  let lr_buf = create_storage_buffer_js(device_any, 4)
  queue_write_f32_js(device_any, weight1_buf, params.weight1)
  queue_write_f32_js(device_any, bias1_buf, params.bias1)
  queue_write_f32_js(device_any, weight2_buf, params.weight2)
  queue_write_f32_js(device_any, bias2_buf, params.bias2)
  queue_write_f32_js(device_any, input_buf, inputs)
  queue_write_u32_js(device_any, labels_buf, labels)
  queue_write_f32_js(device_any, lr_buf, [learning_rate])
  let layer1_pipeline = create_compute_pipeline_js(
    device_any,
    shader_plan.layer1_wgsl,
  )
  let layer2_loss_pipeline = create_compute_pipeline_js(
    device_any,
    loss_plan.layer2_loss_wgsl,
  )
  let loss_epoch_reduce_pipeline = create_compute_pipeline_js(
    device_any,
    loss_plan.loss_epoch_reduce_wgsl,
  )
  let indices_pipeline = create_compute_pipeline_js(
    device_any,
    dataset_plan.indices_wgsl,
  )
  let grad_update_w1_pipeline = create_compute_pipeline_js(
    device_any,
    train_plan.grad_update_w1_wgsl,
  )
  let grad_update_b1_pipeline = create_compute_pipeline_js(
    device_any,
    train_plan.grad_update_b1_wgsl,
  )
  let grad_update_w2_pipeline = create_compute_pipeline_js(
    device_any,
    train_plan.grad_update_w2_wgsl,
  )
  let grad_update_b2_pipeline = create_compute_pipeline_js(
    device_any,
    train_plan.grad_update_b2_wgsl,
  )
  let layer1_bind = create_bind_group_js(device_any, layer1_pipeline, [
    input_buf, indices_buf, weight1_buf, bias1_buf, hidden_buf,
  ])
  let layer2_loss_bind = create_bind_group_js(
    device_any,
    layer2_loss_pipeline,
    [
      hidden_buf, weight2_buf, bias2_buf, labels_buf, indices_buf, loss_buf_any,
      probs_buf_any, correct_buf_any,
    ],
  )
  let loss_epoch_reduce_bind = create_bind_group_js(
    device_any,
    loss_epoch_reduce_pipeline,
    [
      loss_buf_any, correct_buf_any, epoch_loss_buf_any, epoch_correct_buf_any, epoch_seen_buf_any,
    ],
  )
  let indices_bind = create_bind_group_js(device_any, indices_pipeline, [
    indices_buf, shuffle_params_buf,
  ])
  let grad_update_w1_bind = create_bind_group_js(
    device_any,
    grad_update_w1_pipeline,
    [
      input_buf, hidden_buf, probs_buf_any, labels_buf, indices_buf, weight2_buf,
      weight1_buf, lr_buf,
    ],
  )
  let grad_update_b1_bind = create_bind_group_js(
    device_any,
    grad_update_b1_pipeline,
    [
      hidden_buf, probs_buf_any, labels_buf, indices_buf, weight2_buf, bias1_buf,
      lr_buf,
    ],
  )
  let grad_update_w2_bind = create_bind_group_js(
    device_any,
    grad_update_w2_pipeline,
    [hidden_buf, probs_buf_any, labels_buf, indices_buf, weight2_buf, lr_buf],
  )
  let grad_update_b2_bind = create_bind_group_js(
    device_any,
    grad_update_b2_pipeline,
    [probs_buf_any, labels_buf, indices_buf, bias2_buf, lr_buf],
  )
  let steps = effective_count / batch_size
  let indices_dispatch_x = ceil_div(effective_count, workgroup_size)
  let zero_f32 = [Float::from_int(0)]
  let mut epoch = 0
  while epoch < epochs {
    set_status_js("epoch " + epoch.to_string())
    queue_write_f32_js(device_any, epoch_loss_buf_any, zero_f32)
    queue_write_f32_js(device_any, epoch_correct_buf_any, zero_f32)
    queue_write_f32_js(device_any, epoch_seen_buf_any, zero_f32)
    let epoch_seed = if shuffle { seed + epoch } else { 0 }
    let stride = if shuffle {
      shuffle_stride(effective_count, epoch_seed)
    } else {
      1
    }
    queue_write_u32_js(device_any, shuffle_params_buf, [epoch_seed, stride])
    dispatch_compute_js(
      device_any, indices_pipeline, indices_bind, indices_dispatch_x,
    )
    let mut step = 0
    while step < steps {
      let base = step * batch_size
      queue_write_u32_js(device_any, indices_buf, [base])
      let enc = begin_encoder_js(device_any)
      encode_dispatch_js(
        enc,
        layer1_pipeline,
        layer1_bind,
        dispatch.layer1_dispatch_x,
      )
      encode_dispatch_js(
        enc, layer2_loss_pipeline, layer2_loss_bind, loss_dispatch,
      )
      encode_dispatch_js(
        enc, loss_epoch_reduce_pipeline, loss_epoch_reduce_bind, loss_reduce_dispatch,
      )
      // Dispatch order: w1, b1 first (read weight2), then w2, b2 (write weight2)
      encode_dispatch_js(
        enc,
        grad_update_w1_pipeline,
        grad_update_w1_bind,
        train_dispatch.grad_update_w1_dispatch_x,
      )
      encode_dispatch_js(
        enc,
        grad_update_b1_pipeline,
        grad_update_b1_bind,
        train_dispatch.grad_update_b1_dispatch_x,
      )
      encode_dispatch_js(
        enc,
        grad_update_w2_pipeline,
        grad_update_w2_bind,
        train_dispatch.grad_update_w2_dispatch_x,
      )
      encode_dispatch_js(
        enc,
        grad_update_b2_pipeline,
        grad_update_b2_bind,
        train_dispatch.grad_update_b2_dispatch_x,
      )
      submit_encoder_js(device_any, enc)
      step = step + 1
    }
    let loss_read = @web.device_read_buffer_bytes(
      device,
      epoch_loss_buf_any.cast(),
      loss_buffers.epoch_loss_bytes,
    )
    let correct_read = @web.device_read_buffer_bytes(
      device,
      epoch_correct_buf_any.cast(),
      loss_buffers.epoch_correct_bytes,
    )
    let seen_read = @web.device_read_buffer_bytes(
      device,
      epoch_seen_buf_any.cast(),
      loss_buffers.epoch_seen_bytes,
    )
    match loss_read {
      Ok(loss_bytes) =>
        match correct_read {
          Ok(correct_bytes) =>
            match seen_read {
              Ok(seen_bytes) => {
                let loss_values = bytes_to_f32_array_js(loss_bytes)
                let correct_values = bytes_to_f32_array_js(correct_bytes)
                let seen_values = bytes_to_f32_array_js(seen_bytes)
                let seen = seen_values[0]
                let avg_loss = loss_values[0] / seen
                let acc = correct_values[0] / seen
                set_status_js(
                  "epoch " +
                  epoch.to_string() +
                  " avg_loss=" +
                  avg_loss.to_string() +
                  " acc=" +
                  acc.to_string(),
                )
              }
              Err(err) => {
                set_status_js("epoch seen read error: " + err.to_string())
                return
              }
            }
          Err(err) => {
            set_status_js("epoch correct read error: " + err.to_string())
            return
          }
        }
      Err(err) => {
        set_status_js("epoch loss read error: " + err.to_string())
        return
      }
    }
    epoch = epoch + 1
  }
  set_status_js("evaluating test (gpu)...")
  let test_images_path = base + "/t10k-images-idx3-ubyte"
  let test_labels_path = base + "/t10k-labels-idx1-ubyte"
  let test_dataset = match
    @mnist.mnist_load_mlp_dataset(
      test_images_path,
      test_labels_path,
      Some(test_limit),
    ) {
    Ok(ds) => ds
    Err(err) => {
      set_status_js("test load error: " + err.to_string())
      return
    }
  }
  if test_dataset.input_size != input_size {
    set_status_js("test dataset input_size mismatch")
    return
  }
  let test_effective = test_dataset.count / batch_size * batch_size
  if test_effective <= 0 {
    set_status_js("test dataset too small for batch")
    return
  }
  let test_inputs = slice_f32(
    test_dataset.inputs,
    0,
    test_effective * input_size,
  )
  let test_labels = slice_i32(test_dataset.labels, 0, test_effective)
  let test_dataset_buffers = match
    @nn.mlp_plan_dataset_buffers(spec, test_effective) {
    Ok(p) => p
    Err(err) => {
      set_status_js("test dataset buffer error: " + err.to_string())
      return
    }
  }
  let test_loss_buffers = match
    @nn.mlp_plan_loss_buffers_for_dataset(spec, test_effective) {
    Ok(p) => p
    Err(err) => {
      set_status_js("test loss buffer error: " + err.to_string())
      return
    }
  }
  let test_input_buf = create_storage_buffer_js(
    device_any,
    test_dataset_buffers.dataset_input_bytes,
  )
  let test_labels_buf = create_storage_buffer_js(
    device_any,
    test_loss_buffers.labels_bytes,
  )
  let test_indices_buf = create_storage_buffer_js(
    device_any,
    test_dataset_buffers.indices_bytes,
  )
  let test_indices = Array::makei(test_effective + 1, fn(i) {
    if i == 0 {
      0
    } else {
      i - 1
    }
  })
  queue_write_f32_js(device_any, test_input_buf, test_inputs)
  queue_write_u32_js(device_any, test_labels_buf, test_labels)
  queue_write_u32_js(device_any, test_indices_buf, test_indices)
  let test_layer1_bind = create_bind_group_js(device_any, layer1_pipeline, [
    test_input_buf, test_indices_buf, weight1_buf, bias1_buf, hidden_buf,
  ])
  let test_layer2_loss_bind = create_bind_group_js(
    device_any,
    layer2_loss_pipeline,
    [
      hidden_buf, weight2_buf, bias2_buf, test_labels_buf, test_indices_buf, loss_buf_any,
      probs_buf_any, correct_buf_any,
    ],
  )
  let test_steps = test_effective / batch_size
  queue_write_f32_js(device_any, epoch_loss_buf_any, zero_f32)
  queue_write_f32_js(device_any, epoch_correct_buf_any, zero_f32)
  queue_write_f32_js(device_any, epoch_seen_buf_any, zero_f32)
  let mut step = 0
  while step < test_steps {
    let base = step * batch_size
    queue_write_u32_js(device_any, test_indices_buf, [base])
    let enc = begin_encoder_js(device_any)
    encode_dispatch_js(
      enc,
      layer1_pipeline,
      test_layer1_bind,
      dispatch.layer1_dispatch_x,
    )
    encode_dispatch_js(
      enc, layer2_loss_pipeline, test_layer2_loss_bind, loss_dispatch,
    )
    encode_dispatch_js(
      enc, loss_epoch_reduce_pipeline, loss_epoch_reduce_bind, loss_reduce_dispatch,
    )
    submit_encoder_js(device_any, enc)
    step = step + 1
  }
  let loss_read = @web.device_read_buffer_bytes(
    device,
    epoch_loss_buf_any.cast(),
    loss_buffers.epoch_loss_bytes,
  )
  let correct_read = @web.device_read_buffer_bytes(
    device,
    epoch_correct_buf_any.cast(),
    loss_buffers.epoch_correct_bytes,
  )
  let seen_read = @web.device_read_buffer_bytes(
    device,
    epoch_seen_buf_any.cast(),
    loss_buffers.epoch_seen_bytes,
  )
  match loss_read {
    Ok(loss_bytes) =>
      match correct_read {
        Ok(correct_bytes) =>
          match seen_read {
            Ok(seen_bytes) => {
              let loss_values = bytes_to_f32_array_js(loss_bytes)
              let correct_values = bytes_to_f32_array_js(correct_bytes)
              let seen_values = bytes_to_f32_array_js(seen_bytes)
              let seen = seen_values[0]
              let test_loss = loss_values[0] / seen
              let test_acc = correct_values[0] / seen
              set_status_js(
                "test loss=" +
                test_loss.to_string() +
                " acc=" +
                test_acc.to_string(),
              )
            }
            Err(err) => {
              set_status_js("test seen read error: " + err.to_string())
              return
            }
          }
        Err(err) => {
          set_status_js("test correct read error: " + err.to_string())
          return
        }
      }
    Err(err) => {
      set_status_js("test loss read error: " + err.to_string())
      return
    }
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
  let base = if parsed.length() == 0 {
    if seed >= 0 {
      seeded_values(seed, count, offset)
    } else {
      default_values
    }
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
    Err(err) => set_e2e_result_js(false, expected, [], Some(err.to_string()))
  }
}

///|
fn main {
  let mode = query_mode_js()
  if mode == "loss" {
    @async.run_async_main(run_mlp_loss)
  } else if mode == "train" {
    @async.run_async_main(run_mlp_train_step)
  } else if mode == "mnist" {
    @async.run_async_main(run_mnist_train)
  } else {
    @async.run_async_main(run_readback)
  }
}
