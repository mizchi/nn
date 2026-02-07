///|
fn mnist_path(base : String, name : String) -> String {
  base + "/" + name
}

///|
enum InferBackend {
  Cpu
  Gpu
  Auto
} derive(Show, Eq)

///|
struct InferArgs {
  weights_path : String
  limit : Int?
  batch_size : Int
  json : Bool
  backend : InferBackend
  bench : Bool
} derive(Show, Eq)

///|
fn infer_args_default() -> InferArgs {
  let base = "data/mnist"
  {
    weights_path: mnist_path(base, "mlp_784_128_10.bin"),
    limit: None,
    batch_size: 128,
    json: false,
    backend: Auto,
    bench: false,
  }
}

///|
fn infer_backend_label(backend : InferBackend) -> String {
  match backend {
    Cpu => "cpu"
    Gpu => "gpu"
    Auto => "auto"
  }
}

///|
fn parse_backend(value : String) -> Result[InferBackend, String] {
  if value == "cpu" {
    Ok(Cpu)
  } else if value == "gpu" {
    Ok(Gpu)
  } else if value == "auto" {
    Ok(Auto)
  } else {
    Err("invalid backend: " + value)
  }
}

///|
fn print_usage(program : String) -> Unit {
  println(
    "Usage: " +
    program +
    " [--weights PATH] [--limit N] [--batch N] [--backend cpu|gpu|auto] [--bench] [--json]",
  )
  println(
    "  --weights PATH   weights file (default data/mnist/mlp_784_128_10.bin)",
  )
  println("  --limit N        evaluate first N samples")
  println("  --batch N        batch size (default 128)")
  println("  --backend NAME   select backend (cpu|gpu|auto, default auto)")
  println("  --cpu            alias for --backend cpu")
  println("  --gpu            alias for --backend gpu")
  println("  --bench          print evaluation time")
  println("  --json           emit JSON lines")
}

///|
fn parse_limit(value : String) -> Result[Int, String] {
  let parsed = try? @strconv.parse_int(value)
  match parsed {
    Ok(v) => if v > 0 { Ok(v) } else { Err("limit must be > 0") }
    Err(err) => Err("invalid limit: " + err.to_string())
  }
}

///|
fn parse_batch(value : String) -> Result[Int, String] {
  let parsed = try? @strconv.parse_int(value)
  match parsed {
    Ok(v) => if v > 0 { Ok(v) } else { Err("batch must be > 0") }
    Err(err) => Err("invalid batch: " + err.to_string())
  }
}

///|
fn parse_args(args : Array[String]) -> Result[InferArgs, String] {
  let default = infer_args_default()
  let mut weights_path = default.weights_path
  let mut limit = default.limit
  let mut batch_size = default.batch_size
  let mut json = default.json
  let mut backend = default.backend
  let mut bench = default.bench
  let mut i = 1
  while i < args.length() {
    let arg = args[i]
    if arg == "--help" || arg == "-h" {
      return Err("help")
    }
    if arg == "--weights" {
      if i + 1 >= args.length() {
        return Err("missing value for --weights")
      }
      weights_path = args[i + 1]
      i = i + 2
      continue
    }
    if arg.has_prefix("--weights=") {
      let view = arg.sub(start=10) catch {
        _ => return Err("invalid --weights value")
      }
      weights_path = view.to_string()
      i = i + 1
      continue
    }
    if arg == "--limit" {
      if i + 1 >= args.length() {
        return Err("missing value for --limit")
      }
      match parse_limit(args[i + 1]) {
        Ok(v) => limit = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--limit=") {
      let view = arg.sub(start=8) catch {
        _ => return Err("invalid --limit value")
      }
      let value = view.to_string()
      match parse_limit(value) {
        Ok(v) => limit = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--batch" {
      if i + 1 >= args.length() {
        return Err("missing value for --batch")
      }
      match parse_batch(args[i + 1]) {
        Ok(v) => batch_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--batch=") {
      let view = arg.sub(start=8) catch {
        _ => return Err("invalid --batch value")
      }
      let value = view.to_string()
      match parse_batch(value) {
        Ok(v) => batch_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--backend" {
      if i + 1 >= args.length() {
        return Err("missing value for --backend")
      }
      match parse_backend(args[i + 1]) {
        Ok(v) => backend = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--backend=") {
      let view = arg.sub(start=10) catch {
        _ => return Err("invalid --backend value")
      }
      match parse_backend(view.to_string()) {
        Ok(v) => backend = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--cpu" {
      backend = Cpu
      i = i + 1
      continue
    }
    if arg == "--gpu" {
      backend = Gpu
      i = i + 1
      continue
    }
    if arg == "--bench" {
      bench = true
      i = i + 1
      continue
    }
    if arg == "--json" {
      json = true
      i = i + 1
      continue
    }
    return Err("unknown option: " + arg)
  }
  Ok({ weights_path, limit, batch_size, json, backend, bench })
}

///|
fn wgpu_error_to_string(err : @wgpu.WgpuError) -> String {
  match err {
    NotImplemented => "wgpu not implemented"
    NotSupported => "wgpu not supported"
    Validation(msg) => "wgpu validation: " + msg
  }
}

///|
fn read_u32_le(bytes : Bytes, offset : Int) -> UInt {
  for j = 0, acc = UInt::default()
      j < 4
      j = j + 1, acc = acc | (bytes[offset + j].to_uint() << (j * 8)) {

  } else {
    acc
  }
}

///|
fn read_f32_le(bytes : Bytes, offset : Int) -> Float {
  let bits = read_u32_le(bytes, offset)
  Float::reinterpret_from_uint(bits)
}

///|
fn labels_to_bytes(labels : Array[Int], start : Int, batch_size : Int) -> Bytes {
  Bytes::makei(batch_size * 4, fn(idx) {
    let i = idx / 4
    let j = idx % 4
    let label = labels[start + i]
    let bits = Int::reinterpret_as_uint(label)
    (bits >> (j * 8)).to_byte()
  })
}

///|
fn slice_bytes(bytes : Bytes, start : Int, len : Int) -> Bytes {
  bytes.sub(start~, end=start + len).to_bytes()
}

///|
fn argmax_logits(logits : Array[Float], offset : Int, size : Int) -> Int {
  let mut best = 0
  let mut best_val = logits[offset]
  for k = 1; k < size; k = k + 1 {
    let v = logits[offset + k]
    if v > best_val {
      best_val = v
      best = k
    }
  }
  best
}

///|
fn make_input_batch(
  spec : @nn.MlpSpec,
  dataset : @nn.MlpDataset,
  start : Int,
) -> Array[Float] {
  let total = spec.input_size * spec.batch_size
  let out = Array::make(total, Float::from_int(0))
  let src_offset = start * spec.input_size
  for i = 0; i < total; i = i + 1 {
    out[i] = dataset.inputs[src_offset + i]
  }
  out
}

///|
fn make_labels_slice(
  dataset : @nn.MlpDataset,
  start : Int,
  count : Int,
) -> Array[Int] {
  let out = Array::make(count, 0)
  for i = 0; i < count; i = i + 1 {
    out[i] = dataset.labels[start + i]
  }
  out
}

///|
fn make_dataset_slice(
  dataset : @nn.MlpDataset,
  start : Int,
  count : Int,
) -> Result[@nn.MlpDataset, String] {
  if count <= 0 {
    return Err("slice count must be > 0".to_string())
  }
  if start < 0 || start + count > dataset.count {
    return Err("slice out of range".to_string())
  }
  let total = dataset.input_size * count
  let inputs = Array::make(total, Float::from_int(0))
  let src_offset = start * dataset.input_size
  for i = 0; i < total; i = i + 1 {
    inputs[i] = dataset.inputs[src_offset + i]
  }
  let labels = make_labels_slice(dataset, start, count)
  match @nn.mlp_dataset_new(dataset.input_size, inputs, labels) {
    Ok(ds) => Ok(ds)
    Err(err) => Err(err.to_string())
  }
}

///|
async fn gpu_eval(
  spec : @nn.MlpSpec,
  dataset : @nn.MlpDataset,
  weights_bytes : Bytes,
  params : @nn.MlpParams,
) -> Result[@nn.MlpTrainMetrics, String] {
  let adapter = match
    @wgpu.request_adapter(@wgpu.request_adapter_options_default()) {
    Ok(a) => a
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let device = match
    @wgpu.request_device(adapter, @wgpu.device_descriptor_default()) {
    Ok(d) => d
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let queue = match @wgpu.device_get_queue(device) {
    Ok(q) => q
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let resources = match @nn.mlp_plan_resources(spec) {
    Ok(r) => r
    Err(err) => return Err(err.to_string())
  }
  let loss_resources = match @nn.mlp_plan_loss_resources(spec) {
    Ok(r) => r
    Err(err) => return Err(err.to_string())
  }
  let buffers = match @nn.mlp_plan_buffers(spec) {
    Ok(p) => p
    Err(err) => return Err(err.to_string())
  }
  let loss_buffers = match @nn.mlp_plan_loss_buffers(spec) {
    Ok(p) => p
    Err(err) => return Err(err.to_string())
  }
  let workgroup_size = 64
  let shader_plan = match @nn.mlp_shader_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => return Err(err.to_string())
  }
  let loss_shader_plan = match @nn.mlp_loss_shader_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => return Err(err.to_string())
  }
  let dispatch = match @nn.mlp_dispatch_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => return Err(err.to_string())
  }
  let loss_dispatch_x = match @nn.mlp_loss_dispatch_x(spec, workgroup_size) {
    Ok(v) => v
    Err(err) => return Err(err.to_string())
  }
  let loss_reduce_dispatch_x = match
    @nn.mlp_loss_reduce_dispatch_x(spec, workgroup_size) {
    Ok(v) => v
    Err(err) => return Err(err.to_string())
  }
  let expected = match @nn.mlp_params_bytes_length(spec) {
    Ok(v) => v
    Err(err) => return Err(err.to_string())
  }
  if weights_bytes.length() != expected {
    return Err("weights length mismatch".to_string())
  }
  let input_buf = match @wgpu.device_create_buffer(device, resources.input) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let hidden_buf = match @wgpu.device_create_buffer(device, resources.hidden) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let output_buf = match @wgpu.device_create_buffer(device, resources.output) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let weight1_buf = match
    @wgpu.device_create_buffer(device, resources.weight1) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let bias1_buf = match @wgpu.device_create_buffer(device, resources.bias1) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let weight2_buf = match
    @wgpu.device_create_buffer(device, resources.weight2) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let bias2_buf = match @wgpu.device_create_buffer(device, resources.bias2) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let labels_buf = match
    @wgpu.device_create_buffer(device, loss_resources.labels) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_buf = match @wgpu.device_create_buffer(device, loss_resources.loss) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let probs_buf = match
    @wgpu.device_create_buffer(device, loss_resources.probs) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_sum_buf = match
    @wgpu.device_create_buffer(device, loss_resources.loss_sum) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let layer1_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(shader_plan.layer1_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let layer2_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(shader_plan.layer2_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(loss_shader_plan.loss_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_reduce_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(loss_shader_plan.loss_reduce_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let stage = @wgpu.shader_stage_compute
  let layer1_layout = match
    @wgpu.device_create_bind_group_layout(
      device,
      @wgpu.bind_group_layout_descriptor(
        [
          @wgpu.bind_group_layout_entry(
            0, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            1, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            2, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            3, stage, @wgpu.binding_type_storage_buffer,
          ),
        ],
        None,
      ),
    ) {
    Ok(l) => l
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let layer2_layout = match
    @wgpu.device_create_bind_group_layout(
      device,
      @wgpu.bind_group_layout_descriptor(
        [
          @wgpu.bind_group_layout_entry(
            0, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            1, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            2, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            3, stage, @wgpu.binding_type_storage_buffer,
          ),
        ],
        None,
      ),
    ) {
    Ok(l) => l
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_layout = match
    @wgpu.device_create_bind_group_layout(
      device,
      @wgpu.bind_group_layout_descriptor(
        [
          @wgpu.bind_group_layout_entry(
            0, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            1, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            2, stage, @wgpu.binding_type_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            3, stage, @wgpu.binding_type_storage_buffer,
          ),
        ],
        None,
      ),
    ) {
    Ok(l) => l
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_reduce_layout = match
    @wgpu.device_create_bind_group_layout(
      device,
      @wgpu.bind_group_layout_descriptor(
        [
          @wgpu.bind_group_layout_entry(
            0, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            1, stage, @wgpu.binding_type_storage_buffer,
          ),
        ],
        None,
      ),
    ) {
    Ok(l) => l
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let layer1_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        layer1_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(input_buf, 0, resources.input.size),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(
              weight1_buf,
              0,
              resources.weight1.size,
            ),
          ),
          @wgpu.bind_group_entry(
            2,
            @wgpu.binding_resource_buffer(bias1_buf, 0, resources.bias1.size),
          ),
          @wgpu.bind_group_entry(
            3,
            @wgpu.binding_resource_buffer(hidden_buf, 0, resources.hidden.size),
          ),
        ],
        None,
      ),
    ) {
    Ok(g) => g
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let layer2_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        layer2_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(hidden_buf, 0, resources.hidden.size),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(
              weight2_buf,
              0,
              resources.weight2.size,
            ),
          ),
          @wgpu.bind_group_entry(
            2,
            @wgpu.binding_resource_buffer(bias2_buf, 0, resources.bias2.size),
          ),
          @wgpu.bind_group_entry(
            3,
            @wgpu.binding_resource_buffer(output_buf, 0, resources.output.size),
          ),
        ],
        None,
      ),
    ) {
    Ok(g) => g
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        loss_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(output_buf, 0, resources.output.size),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(
              labels_buf,
              0,
              loss_resources.labels.size,
            ),
          ),
          @wgpu.bind_group_entry(
            2,
            @wgpu.binding_resource_buffer(loss_buf, 0, loss_resources.loss.size),
          ),
          @wgpu.bind_group_entry(
            3,
            @wgpu.binding_resource_buffer(
              probs_buf,
              0,
              loss_resources.probs.size,
            ),
          ),
        ],
        None,
      ),
    ) {
    Ok(g) => g
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_reduce_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        loss_reduce_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(loss_buf, 0, loss_resources.loss.size),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(
              loss_sum_buf,
              0,
              loss_resources.loss_sum.size,
            ),
          ),
        ],
        None,
      ),
    ) {
    Ok(g) => g
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let layer1_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([layer1_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let layer2_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([layer2_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([loss_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_reduce_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([loss_reduce_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let layer1_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        layer1_pipeline_layout,
        @wgpu.programmable_stage(layer1_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let layer2_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        layer2_pipeline_layout,
        @wgpu.programmable_stage(layer2_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        loss_pipeline_layout,
        @wgpu.programmable_stage(loss_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_reduce_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        loss_reduce_pipeline_layout,
        @wgpu.programmable_stage(loss_reduce_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let w1_bytes = slice_bytes(weights_bytes, 0, buffers.weight1_bytes)
  let b1_offset = buffers.weight1_bytes
  let b1_bytes = slice_bytes(weights_bytes, b1_offset, buffers.bias1_bytes)
  let w2_offset = b1_offset + buffers.bias1_bytes
  let w2_bytes = slice_bytes(weights_bytes, w2_offset, buffers.weight2_bytes)
  let b2_offset = w2_offset + buffers.weight2_bytes
  let b2_bytes = slice_bytes(weights_bytes, b2_offset, buffers.bias2_bytes)
  ignore(@wgpu.queue_write_buffer(queue, weight1_buf, 0, w1_bytes.to_array()))
  ignore(@wgpu.queue_write_buffer(queue, bias1_buf, 0, b1_bytes.to_array()))
  ignore(@wgpu.queue_write_buffer(queue, weight2_buf, 0, w2_bytes.to_array()))
  ignore(@wgpu.queue_write_buffer(queue, bias2_buf, 0, b2_bytes.to_array()))
  let total_batches = dataset.count / spec.batch_size
  let effective_count = total_batches * spec.batch_size
  let remainder = dataset.count - effective_count
  if effective_count == 0 {
    return Err("dataset too small for batch size".to_string())
  }
  let mut loss_sum_total = Float::from_int(0)
  let mut correct = 0
  let output_size = spec.output_size
  let output_bytes_len = buffers.output_bytes
  for batch = 0; batch < total_batches; batch = batch + 1 {
    let start = batch * spec.batch_size
    let input = make_input_batch(spec, dataset, start)
    let input_bytes = match @nn.mlp_input_to_bytes(spec, input) {
      Ok(b) => b
      Err(err) => return Err(err.to_string())
    }
    ignore(
      @wgpu.queue_write_buffer(queue, input_buf, 0, input_bytes.to_array()),
    )
    let label_bytes = labels_to_bytes(dataset.labels, start, spec.batch_size)
    ignore(
      @wgpu.queue_write_buffer(queue, labels_buf, 0, label_bytes.to_array()),
    )
    ignore(
      @wgpu.device_dispatch_compute(
        device,
        layer1_pipeline,
        layer1_group,
        dispatch.layer1_dispatch_x,
      ),
    )
    ignore(
      @wgpu.device_dispatch_compute(
        device,
        layer2_pipeline,
        layer2_group,
        dispatch.layer2_dispatch_x,
      ),
    )
    ignore(
      @wgpu.device_dispatch_compute(
        device, loss_pipeline, loss_group, loss_dispatch_x,
      ),
    )
    ignore(
      @wgpu.device_dispatch_compute(
        device, loss_reduce_pipeline, loss_reduce_group, loss_reduce_dispatch_x,
      ),
    )
    let output_bytes = match
      @wgpu.device_read_buffer_bytes(device, output_buf, output_bytes_len) {
      Ok(b) => b
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    let logits = match @nn.mlp_output_bytes_to_array(spec, output_bytes) {
      Ok(v) => v
      Err(err) => return Err(err.to_string())
    }
    for i = 0; i < spec.batch_size; i = i + 1 {
      let pred = argmax_logits(logits, i * output_size, output_size)
      if pred == dataset.labels[start + i] {
        correct = correct + 1
      }
    }
    let loss_sum_bytes = match
      @wgpu.device_read_buffer_bytes(
        device,
        loss_sum_buf,
        loss_buffers.loss_sum_bytes,
      ) {
      Ok(b) => b
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    loss_sum_total = loss_sum_total + read_f32_le(loss_sum_bytes, 0)
  }
  if remainder > 0 {
    let tail = match make_dataset_slice(dataset, effective_count, remainder) {
      Ok(ds) => ds
      Err(err) => return Err(err)
    }
    let tail_metrics = match @nn_cpu.mlp_eval(spec, params, tail) {
      Ok(m) => m
      Err(err) => return Err(err.to_string())
    }
    let remainder_f = Float::from_int(remainder)
    loss_sum_total = loss_sum_total + tail_metrics.loss * remainder_f
    let tail_correct = Float::round(tail_metrics.accuracy * remainder_f).to_int()
    correct = correct + tail_correct
  }
  let total_count_f = Float::from_int(dataset.count)
  let loss = loss_sum_total / total_count_f
  let accuracy = Float::from_int(correct) / total_count_f
  Ok(@nn.mlp_train_metrics(loss, accuracy))
}

///|
fn cpu_eval(
  spec : @nn.MlpSpec,
  params : @nn.MlpParams,
  dataset : @nn.MlpDataset,
) -> Result[@nn.MlpTrainMetrics, String] {
  match @nn_cpu.mlp_eval(spec, params, dataset) {
    Ok(m) => Ok(m)
    Err(err) => Err(err.to_string())
  }
}

///|
enum EvalBackend {
  Gpu
  Cpu
}

///|
fn eval_backend_label(backend : EvalBackend) -> String {
  match backend {
    Gpu => "gpu"
    Cpu => "cpu"
  }
}

///|
struct EvalReport {
  backend : EvalBackend
  metrics : @nn.MlpTrainMetrics
}

///|
fn json_escape(value : String) -> String {
  let sb = StringBuilder::new(size_hint=value.length())
  let len = value.length()
  for i = 0; i < len; i = i + 1 {
    let ch = value.unsafe_get(i)
    if ch == '"' {
      sb.write_string("\\\"")
    } else if ch == '\\' {
      sb.write_string("\\\\")
    } else if ch == '\n' {
      sb.write_string("\\n")
    } else if ch == '\r' {
      sb.write_string("\\r")
    } else if ch == '\t' {
      sb.write_string("\\t")
    } else {
      sb.write_char(ch.unsafe_to_char())
    }
  }
  sb.to_string()
}

///|
fn json_string(value : String) -> String {
  "\"" + json_escape(value) + "\""
}

///|
fn json_field(key : String, value : String) -> String {
  json_string(key) + ":" + value
}

///|
fn json_object(fields : Array[String]) -> String {
  let sb = StringBuilder::new()
  sb.write_char('{')
  for i = 0; i < fields.length(); i = i + 1 {
    if i > 0 {
      sb.write_char(',')
    }
    sb.write_string(fields[i])
  }
  sb.write_char('}')
  sb.to_string()
}

///|
fn print_json_message(kind : String, message : String) -> Unit {
  println(
    json_object([
      json_field("type", json_string(kind)),
      json_field("message", json_string(message)),
    ]),
  )
}

///|
fn print_error_line(json : Bool, message : String) -> Unit {
  if json {
    print_json_message("error", message)
  } else {
    println(message)
  }
}

///|
fn print_warn_line(json : Bool, message : String) -> Unit {
  if json {
    print_json_message("warn", message)
  } else {
    println(message)
  }
}

///|
fn print_eval_config(
  cfg : InferArgs,
  dataset : @nn.MlpDataset,
  spec : @nn.MlpSpec,
  backend : InferBackend,
  remainder : Int,
  remainder_backend : String,
) -> Unit {
  let limit_text = match cfg.limit {
    Some(v) => v.to_string()
    None => "all"
  }
  let model = "mlp(" +
    spec.input_size.to_string() +
    "-" +
    spec.hidden_size.to_string() +
    "-" +
    spec.output_size.to_string() +
    ")"
  let backend_label = infer_backend_label(backend)
  if cfg.json {
    let limit_json = match cfg.limit {
      Some(v) => v.to_string()
      None => "null"
    }
    println(
      json_object([
        json_field("type", json_string("config")),
        json_field("weights", json_string(cfg.weights_path)),
        json_field("split", json_string("test")),
        json_field("samples", dataset.count.to_string()),
        json_field("limit", limit_json),
        json_field("batch", spec.batch_size.to_string()),
        json_field("model", json_string(model)),
        json_field("backend", json_string(backend_label)),
        json_field("remainder", remainder.to_string()),
        json_field("remainder_backend", json_string(remainder_backend)),
      ]),
    )
  } else {
    println(
      "config: weights=" +
      cfg.weights_path +
      " split=test" +
      " samples=" +
      dataset.count.to_string() +
      " limit=" +
      limit_text +
      " batch=" +
      spec.batch_size.to_string() +
      " model=" +
      model +
      " backend=" +
      backend_label +
      " remainder=" +
      remainder.to_string() +
      " remainder_backend=" +
      remainder_backend,
    )
  }
}

///|
fn print_eval_result(
  json : Bool,
  report : EvalReport,
  dataset : @nn.MlpDataset,
) -> Unit {
  let backend = eval_backend_label(report.backend)
  if json {
    println(
      json_object([
        json_field("type", json_string("result")),
        json_field("backend", json_string(backend)),
        json_field("split", json_string("test")),
        json_field("samples", dataset.count.to_string()),
        json_field("loss", report.metrics.loss.to_string()),
        json_field("acc", report.metrics.accuracy.to_string()),
      ]),
    )
  } else {
    println(
      "result: backend=" +
      backend +
      " split=test" +
      " samples=" +
      dataset.count.to_string() +
      " loss=" +
      report.metrics.loss.to_string() +
      " acc=" +
      report.metrics.accuracy.to_string(),
    )
  }
}

///|
fn print_eval_bench(json : Bool, backend : EvalBackend, ms : Float) -> Unit {
  let backend_label = eval_backend_label(backend)
  if json {
    println(
      json_object([
        json_field("type", json_string("bench")),
        json_field("backend", json_string(backend_label)),
        json_field("eval_ms", ms.to_string()),
      ]),
    )
  } else {
    println("bench: backend=" + backend_label + " eval_ms=" + ms.to_string())
  }
}

///|
async fn run_eval(
  spec : @nn.MlpSpec,
  params : @nn.MlpParams,
  dataset : @nn.MlpDataset,
  weights_bytes : Bytes,
  backend : InferBackend,
  json : Bool,
) -> Result[EvalReport, String] {
  match backend {
    Cpu => {
      let cpu_result = cpu_eval(spec, params, dataset)
      match cpu_result {
        Ok(m) => Ok({ backend: Cpu, metrics: m })
        Err(err) => Err(err)
      }
    }
    Gpu => {
      if !@wgpu.is_supported() {
        return Err("wgpu not supported".to_string())
      }
      let gpu_result = gpu_eval(spec, dataset, weights_bytes, params)
      match gpu_result {
        Ok(m) => Ok({ backend: Gpu, metrics: m })
        Err(err) => Err(err)
      }
    }
    Auto => {
      if @wgpu.is_supported() {
        let gpu_result = gpu_eval(spec, dataset, weights_bytes, params)
        match gpu_result {
          Ok(m) => return Ok({ backend: Gpu, metrics: m })
          Err(err) => {
            print_warn_line(json, "gpu eval error: " + err)
            print_warn_line(json, "fallback to cpu")
          }
        }
      }
      let cpu_result = cpu_eval(spec, params, dataset)
      match cpu_result {
        Ok(m) => Ok({ backend: Cpu, metrics: m })
        Err(err) => Err(err)
      }
    }
  }
}

///|
async fn infer_main() -> Unit {
  let args = @env.args()
  let program = if args.length() > 0 { args[0] } else { "mnist-infer" }
  let mut wants_json = false
  for i = 1; i < args.length(); i = i + 1 {
    if args[i] == "--json" {
      wants_json = true
      break
    }
  }
  let cfg_result = parse_args(args)
  let cfg = match cfg_result {
    Ok(c) => c
    Err(msg) => {
      if msg != "help" {
        print_error_line(wants_json, "error: " + msg)
      }
      print_usage(program)
      return
    }
  }
  let base = "data/mnist"
  let weights_path = cfg.weights_path
  let exists_result : Result[Bool, Error] = try? @afs.exists(weights_path)
  match exists_result {
    Ok(true) => ()
    Ok(false) => {
      print_error_line(cfg.json, "weights not found: " + weights_path)
      return
    }
    Err(err) => {
      print_error_line(cfg.json, "exists error: " + err.to_string())
      return
    }
  }
  let test_images = mnist_path(base, "t10k-images-idx3-ubyte")
  let test_labels = mnist_path(base, "t10k-labels-idx1-ubyte")
  let test_dataset = @mnist.mnist_load_mlp_dataset(
    test_images,
    test_labels,
    cfg.limit,
  )
  let test_set = match test_dataset {
    Ok(ds) => ds
    Err(err) => {
      print_error_line(cfg.json, "mnist test load error: " + err.to_string())
      return
    }
  }
  let mut batch_size = cfg.batch_size
  if batch_size <= 0 {
    print_error_line(cfg.json, "batch must be > 0")
    return
  }
  if test_set.count < batch_size {
    batch_size = test_set.count
  }
  if batch_size <= 0 {
    print_error_line(cfg.json, "dataset empty")
    return
  }
  let spec = match @nn.mlp_spec_new(784, 128, 10, batch_size) {
    Ok(s) => s
    Err(err) => {
      print_error_line(cfg.json, "spec error: " + err.to_string())
      return
    }
  }
  let data_result = try? @afs.read_file(weights_path)
  let data = match data_result {
    Ok(d) => d
    Err(err) => {
      print_error_line(cfg.json, "read error: " + err.to_string())
      return
    }
  }
  let bytes = data.binary()
  let params = match @nn.mlp_params_from_bytes(spec, bytes) {
    Ok(p) => p
    Err(err) => {
      print_error_line(cfg.json, "weights error: " + err.to_string())
      return
    }
  }
  let remainder = test_set.count % spec.batch_size
  let remainder_backend = if remainder > 0 { "cpu" } else { "none" }
  print_eval_config(
    cfg,
    test_set,
    spec,
    cfg.backend,
    remainder,
    remainder_backend,
  )
  let eval_start = if cfg.bench { @env.now() } else { 0UL }
  let eval_result = run_eval(
    spec,
    params,
    test_set,
    bytes,
    cfg.backend,
    cfg.json,
  )
  let eval_end = if cfg.bench { @env.now() } else { 0UL }
  match eval_result {
    Ok(report) => {
      if cfg.bench {
        let elapsed = eval_end - eval_start
        let ms = Float::from_uint64(elapsed)
        print_eval_bench(cfg.json, report.backend, ms)
      }
      print_eval_result(cfg.json, report, test_set)
    }
    Err(err) => print_error_line(cfg.json, "eval error: " + err)
  }
}

///|
fn main {
  @async.run_async_main(infer_main)
}
