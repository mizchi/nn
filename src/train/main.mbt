///|
fn mnist_path(base : String, name : String) -> String {
  base + "/" + name
}

///|
enum TrainBackend {
  Cpu
  Gpu
  Auto
} derive(Show, Eq)

///|
struct TrainArgs {
  json : Bool
  backend : TrainBackend
  bench : Bool
  bench_no_readback : Bool
  limit : Int?
  epochs : Int?
} derive(Show, Eq)

///|
fn train_args_default() -> TrainArgs {
  {
    json: false,
    backend: Cpu,
    bench: false,
    bench_no_readback: false,
    limit: None,
    epochs: None,
  }
}

///|
fn train_backend_label(backend : TrainBackend) -> String {
  match backend {
    Cpu => "cpu"
    Gpu => "gpu"
    Auto => "auto"
  }
}

///|
fn parse_backend(value : String) -> Result[TrainBackend, String] {
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
    " [--backend cpu|gpu|auto] [--epochs N] [--limit N] [--bench] [--bench-no-readback] [--json]",
  )
  println("  --backend NAME   select backend (cpu|gpu|auto, default cpu)")
  println("  --epochs N       number of epochs (default 20)")
  println("  --limit N        use first N training samples")
  println("  --cpu            alias for --backend cpu")
  println("  --gpu            alias for --backend gpu")
  println("  --bench          print training time")
  println("  --bench-no-readback skip per-step readback (gpu only)")
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
fn parse_epochs(value : String) -> Result[Int, String] {
  let parsed = try? @strconv.parse_int(value)
  match parsed {
    Ok(v) => if v > 0 { Ok(v) } else { Err("epochs must be > 0") }
    Err(err) => Err("invalid epochs: " + err.to_string())
  }
}

///|
fn parse_args(args : Array[String]) -> Result[TrainArgs, String] {
  let default = train_args_default()
  let mut json = default.json
  let mut backend = default.backend
  let mut bench = default.bench
  let mut bench_no_readback = default.bench_no_readback
  let mut limit = default.limit
  let mut epochs = default.epochs
  let mut i = 1
  while i < args.length() {
    let arg = args[i]
    if arg == "--help" || arg == "-h" {
      return Err("help")
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
    if arg == "--bench-no-readback" {
      bench_no_readback = true
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
      match parse_limit(view.to_string()) {
        Ok(v) => limit = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--epochs" {
      if i + 1 >= args.length() {
        return Err("missing value for --epochs")
      }
      match parse_epochs(args[i + 1]) {
        Ok(v) => epochs = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--epochs=") {
      let view = arg.sub(start=9) catch {
        _ => return Err("invalid --epochs value")
      }
      match parse_epochs(view.to_string()) {
        Ok(v) => epochs = Some(v)
        Err(msg) => return Err(msg)
      }
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
  Ok({ json, backend, bench, bench_no_readback, limit, epochs })
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
fn print_train_config(
  args : TrainArgs,
  cfg : @nn.MlpTrainConfig,
  train_set : @nn.MlpDataset,
  test_set : @nn.MlpDataset,
  spec : @nn.MlpSpec,
  backend : TrainBackend,
  remainder : Int,
  remainder_policy : String,
) -> Unit {
  let model = "mlp(" +
    spec.input_size.to_string() +
    "-" +
    spec.hidden_size.to_string() +
    "-" +
    spec.output_size.to_string() +
    ")"
  let backend_label = train_backend_label(backend)
  let train_limit_json = match args.limit {
    Some(v) => v.to_string()
    None => "null"
  }
  let train_limit_text = match args.limit {
    Some(v) => v.to_string()
    None => "none"
  }
  if args.json {
    println(
      json_object([
        json_field("type", json_string("config")),
        json_field("backend", json_string(backend_label)),
        json_field("train_samples", train_set.count.to_string()),
        json_field("test_samples", test_set.count.to_string()),
        json_field("epochs", cfg.epochs.to_string()),
        json_field("batch", cfg.batch_size.to_string()),
        json_field("lr", cfg.learning_rate.to_string()),
        json_field("shuffle", if cfg.shuffle { "true" } else { "false" }),
        json_field("seed", cfg.seed.to_string()),
        json_field("model", json_string(model)),
        json_field("remainder", remainder.to_string()),
        json_field("remainder_policy", json_string(remainder_policy)),
        json_field("train_limit", train_limit_json),
        json_field(
          "bench_no_readback",
          if args.bench_no_readback {
            "true"
          } else {
            "false"
          },
        ),
      ]),
    )
  } else {
    println(
      "config: train_samples=" +
      train_set.count.to_string() +
      " test_samples=" +
      test_set.count.to_string() +
      " epochs=" +
      cfg.epochs.to_string() +
      " batch=" +
      cfg.batch_size.to_string() +
      " lr=" +
      cfg.learning_rate.to_string() +
      " shuffle=" +
      cfg.shuffle.to_string() +
      " seed=" +
      cfg.seed.to_string() +
      " model=" +
      model +
      " backend=" +
      backend_label +
      " remainder=" +
      remainder.to_string() +
      " remainder_policy=" +
      remainder_policy +
      " train_limit=" +
      train_limit_text +
      " bench_no_readback=" +
      args.bench_no_readback.to_string(),
    )
  }
}

///|
fn print_train_epoch(
  args : TrainArgs,
  epoch : Int,
  metrics : @nn.MlpTrainMetrics,
) -> Unit {
  if args.json {
    println(
      json_object([
        json_field("type", json_string("epoch")),
        json_field("split", json_string("train")),
        json_field("epoch", epoch.to_string()),
        json_field("loss", metrics.loss.to_string()),
        json_field("acc", metrics.accuracy.to_string()),
      ]),
    )
  } else {
    println(
      "epoch " +
      epoch.to_string() +
      " loss=" +
      metrics.loss.to_string() +
      " acc=" +
      metrics.accuracy.to_string(),
    )
  }
}

///|
fn print_train_result(
  args : TrainArgs,
  backend : TrainBackend,
  metrics : @nn.MlpTrainMetrics,
) -> Unit {
  let backend_label = train_backend_label(backend)
  if args.json {
    println(
      json_object([
        json_field("type", json_string("result")),
        json_field("split", json_string("test")),
        json_field("backend", json_string(backend_label)),
        json_field("loss", metrics.loss.to_string()),
        json_field("acc", metrics.accuracy.to_string()),
      ]),
    )
  } else {
    println(
      "result: backend=" +
      backend_label +
      " split=test loss=" +
      metrics.loss.to_string() +
      " acc=" +
      metrics.accuracy.to_string(),
    )
  }
}

///|
fn print_save(args : TrainArgs, path : String) -> Unit {
  if args.json {
    println(
      json_object([
        json_field("type", json_string("save")),
        json_field("path", json_string(path)),
      ]),
    )
  } else {
    println("saved weights: " + path)
  }
}

///|
fn print_train_bench(
  args : TrainArgs,
  backend : TrainBackend,
  ms : Float,
) -> Unit {
  let backend_label = train_backend_label(backend)
  if args.json {
    println(
      json_object([
        json_field("type", json_string("bench")),
        json_field("backend", json_string(backend_label)),
        json_field("train_ms", ms.to_string()),
      ]),
    )
  } else {
    println("bench: backend=" + backend_label + " train_ms=" + ms.to_string())
  }
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
fn bytes_to_f32_array(bytes : Bytes) -> Array[Float] {
  let count = bytes.length() / 4
  let out = Array::make(count, Float::from_int(0))
  for i = 0; i < count; i = i + 1 {
    out[i] = read_f32_le(bytes, i * 4)
  }
  out
}

///|
fn float_array_to_bytes(values : Array[Float]) -> Bytes {
  Bytes::makei(values.length() * 4, fn(idx) {
    let i = idx / 4
    let j = idx % 4
    let bits = Float::reinterpret_as_uint(values[i])
    (bits >> (j * 8)).to_byte()
  })
}

///|
fn labels_array_to_bytes(labels : Array[Int]) -> Bytes {
  Bytes::makei(labels.length() * 4, fn(idx) {
    let i = idx / 4
    let j = idx % 4
    let bits = Int::reinterpret_as_uint(labels[i])
    (bits >> (j * 8)).to_byte()
  })
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
fn lcg_next(state : UInt) -> UInt {
  let a = Int::reinterpret_as_uint(1664525)
  let c = Int::reinterpret_as_uint(1013904223)
  state * a + c
}

///|
fn shuffle_indices(indices : Array[Int], seed : Int) -> Unit {
  let mut state = Int::reinterpret_as_uint(seed)
  let mut i = indices.length() - 1
  while i > 0 {
    state = lcg_next(state)
    let j = (state % Int::reinterpret_as_uint(i + 1)).reinterpret_as_int()
    let tmp = indices[i]
    indices[i] = indices[j]
    indices[j] = tmp
    i = i - 1
  }
}

///|
fn fill_batch(
  dataset : @nn.MlpDataset,
  indices : Array[Int],
  batch_start : Int,
  batch_size : Int,
  input_size : Int,
  out_inputs : Array[Float],
  out_labels : Array[Int],
) -> Unit {
  let mut bi = 0
  while bi < batch_size {
    let idx = indices[batch_start + bi]
    out_labels[bi] = dataset.labels[idx]
    let input_offset = idx * input_size
    let out_offset = bi * input_size
    let mut j = 0
    while j < input_size {
      out_inputs[out_offset + j] = dataset.inputs[input_offset + j]
      j = j + 1
    }
    bi = bi + 1
  }
}

///|
fn buffer_usage_storage_upload() -> @wgpu.BufferUsage {
  @wgpu.buffer_usage_or(@wgpu.buffer_usage_storage, @wgpu.buffer_usage_copy_dst)
}

///|
fn buffer_desc(
  size : Int,
  usage : @wgpu.BufferUsage,
  label : String,
) -> @wgpu.BufferDescriptor {
  @wgpu.buffer_descriptor(size, usage, false, Some(label))
}

///|
async fn gpu_train(
  spec : @nn.MlpSpec,
  params : @nn.MlpParams,
  dataset : @nn.MlpDataset,
  config : @nn.MlpTrainConfig,
  bench_no_readback : Bool,
) -> Result[@nn.MlpTrainResult, String] {
  if config.epochs <= 0 {
    return Err("epochs must be > 0".to_string())
  }
  if config.batch_size <= 0 {
    return Err("batch_size must be > 0".to_string())
  }
  if config.batch_size != spec.batch_size {
    return Err("batch_size mismatch".to_string())
  }
  if dataset.input_size != spec.input_size {
    return Err("dataset input_size mismatch".to_string())
  }
  if dataset.count <= 0 {
    return Err("dataset empty".to_string())
  }
  let total_batches = dataset.count / spec.batch_size
  let effective_count = total_batches * spec.batch_size
  if effective_count <= 0 {
    return Err("dataset too small for batch size".to_string())
  }
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
  let train_buffers = match @nn.mlp_plan_train_buffers(spec) {
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
  let train_shader_plan = match
    @nn.mlp_train_shader_plan(spec, workgroup_size) {
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
  let train_dispatch = match @nn.mlp_train_dispatch_plan(spec, workgroup_size) {
    Ok(p) => p
    Err(err) => return Err(err.to_string())
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
  let grad_usage = @wgpu.buffer_usage_storage
  let upload_usage = buffer_usage_storage_upload()
  let grad_w1_buf = match
    @wgpu.device_create_buffer(
      device,
      buffer_desc(train_buffers.grad_weight1_bytes, grad_usage, "mlp_grad_w1"),
    ) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_b1_buf = match
    @wgpu.device_create_buffer(
      device,
      buffer_desc(train_buffers.grad_bias1_bytes, grad_usage, "mlp_grad_b1"),
    ) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_w2_buf = match
    @wgpu.device_create_buffer(
      device,
      buffer_desc(train_buffers.grad_weight2_bytes, grad_usage, "mlp_grad_w2"),
    ) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_b2_buf = match
    @wgpu.device_create_buffer(
      device,
      buffer_desc(train_buffers.grad_bias2_bytes, grad_usage, "mlp_grad_b2"),
    ) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let lr_buf = match
    @wgpu.device_create_buffer(device, buffer_desc(4, upload_usage, "mlp_lr")) {
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
  let grad_w2_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(train_shader_plan.grad_w2_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_b2_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(train_shader_plan.grad_b2_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_w1_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(train_shader_plan.grad_w1_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_b1_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(train_shader_plan.grad_b1_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_w2_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(train_shader_plan.update_w2_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_b2_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(train_shader_plan.update_b2_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_w1_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(train_shader_plan.update_w1_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_b1_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(train_shader_plan.update_b1_wgsl, None),
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
  let grad_w2_layout = match
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
  let grad_b2_layout = match
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
        ],
        None,
      ),
    ) {
    Ok(l) => l
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_w1_layout = match
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
            3, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            4, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            5, stage, @wgpu.binding_type_storage_buffer,
          ),
        ],
        None,
      ),
    ) {
    Ok(l) => l
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_b1_layout = match
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
            3, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            4, stage, @wgpu.binding_type_storage_buffer,
          ),
        ],
        None,
      ),
    ) {
    Ok(l) => l
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_layout = match
    @wgpu.device_create_bind_group_layout(
      device,
      @wgpu.bind_group_layout_descriptor(
        [
          @wgpu.bind_group_layout_entry(
            0, stage, @wgpu.binding_type_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            1, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            2, stage, @wgpu.binding_type_read_only_storage_buffer,
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
  let grad_w2_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        grad_w2_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(hidden_buf, 0, resources.hidden.size),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(
              probs_buf,
              0,
              loss_resources.probs.size,
            ),
          ),
          @wgpu.bind_group_entry(
            2,
            @wgpu.binding_resource_buffer(
              labels_buf,
              0,
              loss_resources.labels.size,
            ),
          ),
          @wgpu.bind_group_entry(
            3,
            @wgpu.binding_resource_buffer(
              grad_w2_buf,
              0,
              train_buffers.grad_weight2_bytes,
            ),
          ),
        ],
        None,
      ),
    ) {
    Ok(g) => g
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_b2_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        grad_b2_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(
              probs_buf,
              0,
              loss_resources.probs.size,
            ),
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
            @wgpu.binding_resource_buffer(
              grad_b2_buf,
              0,
              train_buffers.grad_bias2_bytes,
            ),
          ),
        ],
        None,
      ),
    ) {
    Ok(g) => g
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_w1_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        grad_w1_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(input_buf, 0, resources.input.size),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(hidden_buf, 0, resources.hidden.size),
          ),
          @wgpu.bind_group_entry(
            2,
            @wgpu.binding_resource_buffer(
              probs_buf,
              0,
              loss_resources.probs.size,
            ),
          ),
          @wgpu.bind_group_entry(
            3,
            @wgpu.binding_resource_buffer(
              labels_buf,
              0,
              loss_resources.labels.size,
            ),
          ),
          @wgpu.bind_group_entry(
            4,
            @wgpu.binding_resource_buffer(
              weight2_buf,
              0,
              resources.weight2.size,
            ),
          ),
          @wgpu.bind_group_entry(
            5,
            @wgpu.binding_resource_buffer(
              grad_w1_buf,
              0,
              train_buffers.grad_weight1_bytes,
            ),
          ),
        ],
        None,
      ),
    ) {
    Ok(g) => g
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_b1_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        grad_b1_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(hidden_buf, 0, resources.hidden.size),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(
              probs_buf,
              0,
              loss_resources.probs.size,
            ),
          ),
          @wgpu.bind_group_entry(
            2,
            @wgpu.binding_resource_buffer(
              labels_buf,
              0,
              loss_resources.labels.size,
            ),
          ),
          @wgpu.bind_group_entry(
            3,
            @wgpu.binding_resource_buffer(
              weight2_buf,
              0,
              resources.weight2.size,
            ),
          ),
          @wgpu.bind_group_entry(
            4,
            @wgpu.binding_resource_buffer(
              grad_b1_buf,
              0,
              train_buffers.grad_bias1_bytes,
            ),
          ),
        ],
        None,
      ),
    ) {
    Ok(g) => g
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_w2_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        update_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(
              weight2_buf,
              0,
              resources.weight2.size,
            ),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(
              grad_w2_buf,
              0,
              train_buffers.grad_weight2_bytes,
            ),
          ),
          @wgpu.bind_group_entry(2, @wgpu.binding_resource_buffer(lr_buf, 0, 4)),
        ],
        None,
      ),
    ) {
    Ok(g) => g
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_b2_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        update_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(bias2_buf, 0, resources.bias2.size),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(
              grad_b2_buf,
              0,
              train_buffers.grad_bias2_bytes,
            ),
          ),
          @wgpu.bind_group_entry(2, @wgpu.binding_resource_buffer(lr_buf, 0, 4)),
        ],
        None,
      ),
    ) {
    Ok(g) => g
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_w1_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        update_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(
              weight1_buf,
              0,
              resources.weight1.size,
            ),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(
              grad_w1_buf,
              0,
              train_buffers.grad_weight1_bytes,
            ),
          ),
          @wgpu.bind_group_entry(2, @wgpu.binding_resource_buffer(lr_buf, 0, 4)),
        ],
        None,
      ),
    ) {
    Ok(g) => g
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_b1_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        update_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(bias1_buf, 0, resources.bias1.size),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(
              grad_b1_buf,
              0,
              train_buffers.grad_bias1_bytes,
            ),
          ),
          @wgpu.bind_group_entry(2, @wgpu.binding_resource_buffer(lr_buf, 0, 4)),
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
  let grad_w2_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([grad_w2_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_b2_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([grad_b2_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_w1_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([grad_w1_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_b1_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([grad_b1_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([update_layout], None),
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
  let grad_w2_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        grad_w2_pipeline_layout,
        @wgpu.programmable_stage(grad_w2_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_b2_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        grad_b2_pipeline_layout,
        @wgpu.programmable_stage(grad_b2_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_w1_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        grad_w1_pipeline_layout,
        @wgpu.programmable_stage(grad_w1_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_b1_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        grad_b1_pipeline_layout,
        @wgpu.programmable_stage(grad_b1_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_w2_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        update_pipeline_layout,
        @wgpu.programmable_stage(update_w2_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_b2_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        update_pipeline_layout,
        @wgpu.programmable_stage(update_b2_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_w1_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        update_pipeline_layout,
        @wgpu.programmable_stage(update_w1_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let update_b1_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        update_pipeline_layout,
        @wgpu.programmable_stage(update_b1_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let w1_bytes = float_array_to_bytes(params.weight1)
  let b1_bytes = float_array_to_bytes(params.bias1)
  let w2_bytes = float_array_to_bytes(params.weight2)
  let b2_bytes = float_array_to_bytes(params.bias2)
  let lr_bytes = float_array_to_bytes([config.learning_rate])
  ignore(@wgpu.queue_write_buffer(queue, weight1_buf, 0, w1_bytes.to_array()))
  ignore(@wgpu.queue_write_buffer(queue, bias1_buf, 0, b1_bytes.to_array()))
  ignore(@wgpu.queue_write_buffer(queue, weight2_buf, 0, w2_bytes.to_array()))
  ignore(@wgpu.queue_write_buffer(queue, bias2_buf, 0, b2_bytes.to_array()))
  ignore(@wgpu.queue_write_buffer(queue, lr_buf, 0, lr_bytes.to_array()))
  let output_bytes_len = buffers.output_bytes
  let loss_sum_bytes_len = loss_buffers.loss_sum_bytes
  let batch_inputs = Array::make(
    spec.input_size * spec.batch_size,
    Float::from_int(0),
  )
  let batch_labels = Array::make(spec.batch_size, 0)
  let indices = Array::makei(effective_count, fn(i) { i })
  let metrics : Array[@nn.MlpTrainMetrics] = []
  let output_size = spec.output_size
  for epoch = 0; epoch < config.epochs; epoch = epoch + 1 {
    if config.shuffle {
      shuffle_indices(indices, config.seed + epoch)
    }
    let mut epoch_loss_sum = Float::from_int(0)
    let mut epoch_correct = 0
    let mut epoch_seen = 0
    for step = 0; step < total_batches; step = step + 1 {
      let batch_start = step * spec.batch_size
      fill_batch(
        dataset,
        indices,
        batch_start,
        spec.batch_size,
        spec.input_size,
        batch_inputs,
        batch_labels,
      )
      let input_bytes = match @nn.mlp_input_to_bytes(spec, batch_inputs) {
        Ok(b) => b
        Err(err) => return Err(err.to_string())
      }
      let label_bytes = labels_array_to_bytes(batch_labels)
      ignore(
        @wgpu.queue_write_buffer(queue, input_buf, 0, input_bytes.to_array()),
      )
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
      ignore(
        @wgpu.device_dispatch_compute(
          device,
          grad_w2_pipeline,
          grad_w2_group,
          train_dispatch.grad_w2_dispatch_x,
        ),
      )
      ignore(
        @wgpu.device_dispatch_compute(
          device,
          grad_b2_pipeline,
          grad_b2_group,
          train_dispatch.grad_b2_dispatch_x,
        ),
      )
      ignore(
        @wgpu.device_dispatch_compute(
          device,
          grad_w1_pipeline,
          grad_w1_group,
          train_dispatch.grad_w1_dispatch_x,
        ),
      )
      ignore(
        @wgpu.device_dispatch_compute(
          device,
          grad_b1_pipeline,
          grad_b1_group,
          train_dispatch.grad_b1_dispatch_x,
        ),
      )
      ignore(
        @wgpu.device_dispatch_compute(
          device,
          update_w2_pipeline,
          update_w2_group,
          train_dispatch.update_w2_dispatch_x,
        ),
      )
      ignore(
        @wgpu.device_dispatch_compute(
          device,
          update_b2_pipeline,
          update_b2_group,
          train_dispatch.update_b2_dispatch_x,
        ),
      )
      ignore(
        @wgpu.device_dispatch_compute(
          device,
          update_w1_pipeline,
          update_w1_group,
          train_dispatch.update_w1_dispatch_x,
        ),
      )
      ignore(
        @wgpu.device_dispatch_compute(
          device,
          update_b1_pipeline,
          update_b1_group,
          train_dispatch.update_b1_dispatch_x,
        ),
      )
      if !bench_no_readback {
        let loss_sum_bytes = match
          @wgpu.device_read_buffer_bytes(
            device, loss_sum_buf, loss_sum_bytes_len,
          ) {
          Ok(b) => b
          Err(err) => return Err(wgpu_error_to_string(err))
        }
        epoch_loss_sum = epoch_loss_sum + read_f32_le(loss_sum_bytes, 0)
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
          if pred == batch_labels[i] {
            epoch_correct = epoch_correct + 1
          }
        }
        epoch_seen = epoch_seen + spec.batch_size
      }
    }
    if !bench_no_readback {
      let epoch_loss = epoch_loss_sum / Float::from_int(epoch_seen)
      let epoch_acc = Float::from_int(epoch_correct) /
        Float::from_int(epoch_seen)
      metrics.push(@nn.mlp_train_metrics(epoch_loss, epoch_acc))
    }
  }
  let weight1_bytes = match
    @wgpu.device_read_buffer_bytes(device, weight1_buf, buffers.weight1_bytes) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let bias1_bytes = match
    @wgpu.device_read_buffer_bytes(device, bias1_buf, buffers.bias1_bytes) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let weight2_bytes = match
    @wgpu.device_read_buffer_bytes(device, weight2_buf, buffers.weight2_bytes) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let bias2_bytes = match
    @wgpu.device_read_buffer_bytes(device, bias2_buf, buffers.bias2_bytes) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let weight1 = bytes_to_f32_array(weight1_bytes)
  let bias1 = bytes_to_f32_array(bias1_bytes)
  let weight2 = bytes_to_f32_array(weight2_bytes)
  let bias2 = bytes_to_f32_array(bias2_bytes)
  let updated_params = match
    @nn.mlp_params_new(spec, weight1, bias1, weight2, bias2) {
    Ok(p) => p
    Err(err) => return Err(err.to_string())
  }
  Ok(@nn.mlp_train_result(updated_params, metrics))
}

///|
struct TrainReport {
  backend : TrainBackend
  result : @nn.MlpTrainResult
}

///|
async fn run_train(
  spec : @nn.MlpSpec,
  params : @nn.MlpParams,
  dataset : @nn.MlpDataset,
  config : @nn.MlpTrainConfig,
  backend : TrainBackend,
  bench_no_readback : Bool,
  json : Bool,
) -> Result[TrainReport, String] {
  match backend {
    Cpu => {
      let train_result = @nn.mlp_train(spec, params, dataset, config)
      match train_result {
        Ok(r) => Ok({ backend: Cpu, result: r })
        Err(err) => Err(err.to_string())
      }
    }
    Gpu => {
      if !@wgpu.is_supported() {
        return Err("wgpu not supported".to_string())
      }
      let train_result = gpu_train(
        spec, params, dataset, config, bench_no_readback,
      )
      match train_result {
        Ok(r) => Ok({ backend: Gpu, result: r })
        Err(err) => Err(err)
      }
    }
    Auto => {
      if @wgpu.is_supported() {
        let train_result = gpu_train(
          spec, params, dataset, config, bench_no_readback,
        )
        match train_result {
          Ok(r) => return Ok({ backend: Gpu, result: r })
          Err(err) => {
            print_warn_line(json, "gpu train error: " + err)
            print_warn_line(json, "fallback to cpu")
          }
        }
      }
      let train_result = @nn.mlp_train(spec, params, dataset, config)
      match train_result {
        Ok(r) => Ok({ backend: Cpu, result: r })
        Err(err) => Err(err.to_string())
      }
    }
  }
}

///|
async fn train_main() -> Unit {
  let args = @env.args()
  let program = if args.length() > 0 { args[0] } else { "mnist-train" }
  let mut wants_json = false
  for i = 1; i < args.length(); i = i + 1 {
    if args[i] == "--json" {
      wants_json = true
      break
    }
  }
  let cfg_result = parse_args(args)
  let args = match cfg_result {
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
  let train_images = mnist_path(base, "train-images-idx3-ubyte")
  let train_labels = mnist_path(base, "train-labels-idx1-ubyte")
  let test_images = mnist_path(base, "t10k-images-idx3-ubyte")
  let test_labels = mnist_path(base, "t10k-labels-idx1-ubyte")
  let train_dataset = @mnist.mnist_load_mlp_dataset(
    train_images,
    train_labels,
    args.limit,
  )
  let test_dataset = @mnist.mnist_load_mlp_dataset(
    test_images,
    test_labels,
    None,
  )
  let train = match train_dataset {
    Ok(ds) => ds
    Err(err) => {
      print_error_line(args.json, "mnist train load error: " + err.to_string())
      return
    }
  }
  let test_set = match test_dataset {
    Ok(ds) => ds
    Err(err) => {
      print_error_line(args.json, "mnist test load error: " + err.to_string())
      return
    }
  }
  let base_cfg = @nn.mlp_train_config(
    20,
    128,
    Float::from_int(1) / Float::from_int(10),
    true,
    0,
  )
  let cfg = match args.epochs {
    Some(v) =>
      @nn.mlp_train_config(
        v,
        base_cfg.batch_size,
        base_cfg.learning_rate,
        base_cfg.shuffle,
        base_cfg.seed,
      )
    None => base_cfg
  }
  let spec = match @nn.mlp_spec_new(784, 128, 10, cfg.batch_size) {
    Ok(s) => s
    Err(err) => {
      print_error_line(args.json, "spec error: " + err.to_string())
      return
    }
  }
  let params = match
    @nn.mlp_init_params_with_policy(
      spec,
      cfg.seed,
      @nn.mlp_init_policy_he_uniform,
    ) {
    Ok(p) => p
    Err(err) => {
      print_error_line(args.json, "init error: " + err.to_string())
      return
    }
  }
  let remainder = train.count % cfg.batch_size
  let remainder_policy = match args.backend {
    Cpu => "full"
    Gpu => "drop"
    Auto => "auto"
  }
  print_train_config(
    args,
    cfg,
    train,
    test_set,
    spec,
    args.backend,
    remainder,
    remainder_policy,
  )
  let train_start = if args.bench { @env.now() } else { 0UL }
  let train_result = run_train(
    spec,
    params,
    train,
    cfg,
    args.backend,
    args.bench_no_readback,
    args.json,
  )
  let train_end = if args.bench { @env.now() } else { 0UL }
  let report = match train_result {
    Ok(r) => r
    Err(err) => {
      print_error_line(args.json, "train error: " + err)
      return
    }
  }
  if args.bench_no_readback {
    match report.backend {
      Gpu =>
        if report.result.metrics.length() == 0 {
          print_warn_line(args.json, "metrics skipped: --bench-no-readback")
        }
      Cpu =>
        print_warn_line(
          args.json,
          "--bench-no-readback ignored for cpu backend",
        )
      Auto => ()
    }
  }
  if args.bench {
    let elapsed = train_end - train_start
    let ms = Float::from_uint64(elapsed)
    print_train_bench(args, report.backend, ms)
  }
  let trained = report.result
  for i = 0; i < trained.metrics.length(); i = i + 1 {
    let m = trained.metrics[i]
    print_train_epoch(args, i + 1, m)
  }
  let eval = @nn.mlp_eval(spec, trained.params, test_set)
  match eval {
    Ok(m) => print_train_result(args, report.backend, m)
    Err(err) => print_error_line(args.json, "eval error: " + err.to_string())
  }
  let exists_result : Result[Bool, Error] = try? @afs.exists(base)
  match exists_result {
    Ok(true) => ()
    Ok(false) => {
      let mkdir_result : Result[Unit, Error] = try? @afs.mkdir(
        base,
        permission=0o755,
        recursive=true,
      )
      match mkdir_result {
        Ok(_) => ()
        Err(err) =>
          print_error_line(args.json, "mkdir error: " + err.to_string())
      }
    }
    Err(err) => print_error_line(args.json, "exists error: " + err.to_string())
  }
  let bytes = @nn.mlp_params_to_bytes(spec, trained.params)
  match bytes {
    Ok(data) => {
      let write_result : Result[Unit, Error] = try? @afs.write_file(
        mnist_path(base, "mlp_784_128_10.bin"),
        data,
        create=0o644,
        truncate=true,
      )
      match write_result {
        Ok(_) => print_save(args, "data/mnist/mlp_784_128_10.bin")
        Err(err) =>
          print_error_line(args.json, "save error: " + err.to_string())
      }
    }
    Err(err) =>
      print_error_line(args.json, "serialize error: " + err.to_string())
  }
}

///|
fn main {
  @async.run_async_main(train_main)
}
