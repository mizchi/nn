///|
fn mnist_path(base : String, name : String) -> String {
  base + "/" + name
}

///|
enum TrainBackend {
  Cpu
  Gpu
  Auto
  GpuTransformer
  Vit
} derive(Show, Eq)

///|
struct TrainArgs {
  json : Bool
  backend : TrainBackend
  bench : Bool
  bench_no_readback : Bool
  use_blas : Bool
  profile : Bool
  batch_size : Int?
  workgroup_size : Int?
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
    use_blas: false,
    profile: false,
    batch_size: None,
    workgroup_size: None,
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
    GpuTransformer => "gpu-transformer"
    Vit => "vit"
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
    " [--backend cpu|gpu|auto] [--epochs N] [--batch N] [--workgroup N] [--limit N] [--bench] [--bench-no-readback] [--blas] [--profile] [--json]",
  )
  println("  --backend NAME   select backend (cpu|gpu|auto, default cpu)")
  println("  --epochs N       number of epochs (default 20)")
  println("  --batch N        batch size (default 128)")
  println("  --workgroup N    workgroup size for GPU (default 64)")
  println("  --limit N        use first N training samples")
  println("  --cpu            alias for --backend cpu")
  println("  --gpu            alias for --backend gpu")
  println("  --vit            train ViT model instead of MLP")
  println("  --blas           use BLAS-optimized CPU training")
  println("  --bench          print training time")
  println("  --bench-no-readback skip per-step readback (gpu only)")
  println("  --profile        per-phase GPU profiling (gpu only)")
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
fn parse_batch(value : String) -> Result[Int, String] {
  let parsed = try? @strconv.parse_int(value)
  match parsed {
    Ok(v) => if v > 0 { Ok(v) } else { Err("batch must be > 0") }
    Err(err) => Err("invalid batch: " + err.to_string())
  }
}

///|
fn parse_workgroup(value : String) -> Result[Int, String] {
  let parsed = try? @strconv.parse_int(value)
  match parsed {
    Ok(v) => if v > 0 { Ok(v) } else { Err("workgroup must be > 0") }
    Err(err) => Err("invalid workgroup: " + err.to_string())
  }
}

///|
fn parse_args(args : Array[String]) -> Result[TrainArgs, String] {
  let default = train_args_default()
  let mut json = default.json
  let mut backend = default.backend
  let mut bench = default.bench
  let mut bench_no_readback = default.bench_no_readback
  let mut use_blas = default.use_blas
  let mut profile = default.profile
  let mut batch_size = default.batch_size
  let mut workgroup_size = default.workgroup_size
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
    if arg == "--gpu-transformer" {
      backend = GpuTransformer
      i = i + 1
      continue
    }
    if arg == "--vit" {
      backend = Vit
      i = i + 1
      continue
    }
    if arg == "--bench" {
      bench = true
      i = i + 1
      continue
    }
    if arg == "--blas" {
      use_blas = true
      i = i + 1
      continue
    }
    if arg == "--profile" {
      profile = true
      i = i + 1
      continue
    }
    if arg == "--batch" {
      if i + 1 >= args.length() {
        return Err("missing value for --batch")
      }
      match parse_batch(args[i + 1]) {
        Ok(v) => batch_size = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--batch=") {
      let view = arg.sub(start=8) catch {
        _ => return Err("invalid --batch value")
      }
      match parse_batch(view.to_string()) {
        Ok(v) => batch_size = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--workgroup" {
      if i + 1 >= args.length() {
        return Err("missing value for --workgroup")
      }
      match parse_workgroup(args[i + 1]) {
        Ok(v) => workgroup_size = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--workgroup=") {
      let view = arg.sub(start=12) catch {
        _ => return Err("invalid --workgroup value")
      }
      match parse_workgroup(view.to_string()) {
        Ok(v) => workgroup_size = Some(v)
        Err(msg) => return Err(msg)
      }
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
  Ok({
    json,
    backend,
    bench,
    bench_no_readback,
    use_blas,
    profile,
    batch_size,
    workgroup_size,
    limit,
    epochs,
  })
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
  workgroup_size : Int,
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
        json_field("workgroup", workgroup_size.to_string()),
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
      " workgroup=" +
      workgroup_size.to_string() +
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
fn slice_f32(values : Array[Float], length : Int) -> Array[Float] {
  if length >= values.length() {
    values
  } else {
    Array::makei(length, fn(i) { values[i] })
  }
}

///|
fn slice_f32_offset(
  values : Array[Float],
  offset : Int,
  length : Int,
) -> Array[Float] {
  if offset == 0 {
    slice_f32(values, length)
  } else {
    Array::makei(length, fn(i) { values[offset + i] })
  }
}

///|
fn slice_i32(values : Array[Int], length : Int) -> Array[Int] {
  if length >= values.length() {
    values
  } else {
    Array::makei(length, fn(i) { values[i] })
  }
}

///|
fn slice_i32_offset(
  values : Array[Int],
  offset : Int,
  length : Int,
) -> Array[Int] {
  if offset == 0 {
    slice_i32(values, length)
  } else {
    Array::makei(length, fn(i) { values[offset + i] })
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
  workgroup_size : Int,
  bench_no_readback : Bool,
  profile : Bool,
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
  if workgroup_size <= 0 {
    return Err("workgroup_size must be > 0".to_string())
  }
  let total_batches = dataset.count / spec.batch_size
  let effective_count = total_batches * spec.batch_size
  if effective_count <= 0 {
    return Err("dataset too small for batch size".to_string())
  }
  let total_inputs = effective_count * spec.input_size
  let all_inputs = slice_f32(dataset.inputs, total_inputs)
  let all_labels = slice_i32(dataset.labels, effective_count)
  let max_storage_buffer_binding_size = 134_217_728
  let max_dataset_samples = max_storage_buffer_binding_size /
    (spec.input_size * 4)
  let max_segment_samples = max_dataset_samples /
    spec.batch_size *
    spec.batch_size
  let segment_samples = if max_segment_samples >= spec.batch_size &&
    effective_count > max_segment_samples {
    max_segment_samples
  } else {
    effective_count
  }
  let use_segments = segment_samples < effective_count
  if use_segments {
    println(
      "gpu: segmenting dataset (segment_samples=" +
      segment_samples.to_string() +
      ")",
    )
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
  let dataset_buffers = match
    @nn.mlp_plan_dataset_buffers(spec, segment_samples) {
    Ok(p) => p
    Err(err) => return Err(err.to_string())
  }
  let loss_resources = match
    @nn.mlp_plan_loss_resources_for_dataset(spec, segment_samples) {
    Ok(r) => r
    Err(err) => return Err(err.to_string())
  }
  let buffers = match @nn.mlp_plan_buffers(spec) {
    Ok(p) => p
    Err(err) => return Err(err.to_string())
  }
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
  let dataset_shader_plan = match
    @nn.mlp_dataset_shader_plan(spec, segment_samples, workgroup_size) {
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
  let upload_usage = buffer_usage_storage_upload()
  // Pre-compute segment info: (segment_offset, segment_effective)
  let segment_count = ceil_div(effective_count, segment_samples)
  let segment_info : Array[(Int, Int)] = []
  for seg = 0; seg < segment_count; seg = seg + 1 {
    let offset = seg * segment_samples
    let remaining = effective_count - offset
    let effective = (if remaining < segment_samples {
        remaining
      } else {
        segment_samples
      }) /
      spec.batch_size *
      spec.batch_size
    if effective > 0 {
      segment_info.push((offset, effective))
    }
  }
  // Create per-segment buffers and upload data ONCE
  let seg_input_bufs : Array[@wgpu.Buffer] = []
  let seg_labels_bufs : Array[@wgpu.Buffer] = []
  let seg_indices_bufs : Array[@wgpu.Buffer] = []
  let seg_dataset_buffers_arr : Array[@nn.MlpDatasetBufferPlan] = []
  let seg_loss_resources_arr : Array[@nn.MlpLossResourcePlan] = []
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
  let shuffle_params_buf = match
    @wgpu.device_create_buffer(
      device,
      buffer_desc(
        dataset_buffers.shuffle_params_bytes,
        upload_usage,
        "mlp_shuffle_params",
      ),
    ) {
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
  let correct_buf = match
    @wgpu.device_create_buffer(device, loss_resources.correct) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let epoch_loss_buf = match
    @wgpu.device_create_buffer(device, loss_resources.epoch_loss) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let epoch_correct_buf = match
    @wgpu.device_create_buffer(device, loss_resources.epoch_correct) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let epoch_seen_buf = match
    @wgpu.device_create_buffer(device, loss_resources.epoch_seen) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let lr_buf = match
    @wgpu.device_create_buffer(device, buffer_desc(4, upload_usage, "mlp_lr")) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let dh_bytes = spec.batch_size * spec.hidden_size * 4
  let dh_buf = match
    @wgpu.device_create_buffer(
      device,
      buffer_desc(dh_bytes, @wgpu.buffer_usage_storage, "mlp_dh"),
    ) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  // Create per-segment buffers and upload data ONCE (before epoch loop)
  for si = 0; si < segment_info.length(); si = si + 1 {
    let (seg_offset, seg_effective) = segment_info[si]
    let seg_db = match @nn.mlp_plan_dataset_buffers(spec, seg_effective) {
      Ok(p) => p
      Err(err) => return Err(err.to_string())
    }
    let seg_lr = match
      @nn.mlp_plan_loss_resources_for_dataset(spec, seg_effective) {
      Ok(r) => r
      Err(err) => return Err(err.to_string())
    }
    let s_input_buf = match
      @wgpu.device_create_buffer(
        device,
        buffer_desc(seg_db.dataset_input_bytes, upload_usage, "mlp_input_seg"),
      ) {
      Ok(b) => b
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    let s_labels_buf = match @wgpu.device_create_buffer(device, seg_lr.labels) {
      Ok(b) => b
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    let s_indices_buf = match
      @wgpu.device_create_buffer(
        device,
        buffer_desc(seg_db.indices_bytes, upload_usage, "mlp_indices_seg"),
      ) {
      Ok(b) => b
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    // Upload data ONCE
    let input_offset = seg_offset * spec.input_size
    let input_len = seg_effective * spec.input_size
    let seg_inputs = slice_f32_offset(all_inputs, input_offset, input_len)
    let seg_labels = slice_i32_offset(all_labels, seg_offset, seg_effective)
    let input_bytes = float_array_to_bytes(seg_inputs)
    let label_bytes = labels_array_to_bytes(seg_labels)
    ignore(
      @wgpu.queue_write_buffer(queue, s_input_buf, 0, input_bytes.to_array()),
    )
    ignore(
      @wgpu.queue_write_buffer(queue, s_labels_buf, 0, label_bytes.to_array()),
    )
    seg_input_bufs.push(s_input_buf)
    seg_labels_bufs.push(s_labels_buf)
    seg_indices_bufs.push(s_indices_buf)
    seg_dataset_buffers_arr.push(seg_db)
    seg_loss_resources_arr.push(seg_lr)
  }
  let layer1_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(shader_plan.layer1_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let layer2_loss_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(loss_shader_plan.layer2_loss_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let indices_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(dataset_shader_plan.indices_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_epoch_reduce_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(
        loss_shader_plan.loss_epoch_reduce_wgsl,
        None,
      ),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let compute_dh_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(train_shader_plan.compute_dh_wgsl, None),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_w1_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(
        train_shader_plan.grad_update_w1_wgsl,
        None,
      ),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_b1_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(
        train_shader_plan.grad_update_b1_wgsl,
        None,
      ),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_w2_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(
        train_shader_plan.grad_update_w2_wgsl,
        None,
      ),
    ) {
    Ok(m) => m
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_b2_module = match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(
        train_shader_plan.grad_update_b2_wgsl,
        None,
      ),
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
  let layer2_loss_layout = match
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
          @wgpu.bind_group_layout_entry(
            6, stage, @wgpu.binding_type_storage_buffer,
          ),
          @wgpu.bind_group_layout_entry(
            7, stage, @wgpu.binding_type_storage_buffer,
          ),
        ],
        None,
      ),
    ) {
    Ok(l) => l
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let indices_layout = match
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
        ],
        None,
      ),
    ) {
    Ok(l) => l
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_epoch_reduce_layout = match
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
  // compute_dh: 4 bindings (hidden[r], dlogit[r], weight2[r], dh[rw])
  let compute_dh_layout = match
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
  // grad_update_w1: 5 bindings (input[r], dh[r], indices[r], weight1[rw], lr[r])
  let grad_update_w1_layout = match
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
          @wgpu.bind_group_layout_entry(
            4, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
        ],
        None,
      ),
    ) {
    Ok(l) => l
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  // grad_update_b1: 3 bindings (dh[r], bias1[rw], lr[r])
  let grad_update_b1_layout = match
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
  // grad_update_w2: 4 bindings (hidden[r], dlogit[r], weight2[rw], lr[r])
  let grad_update_w2_layout = match
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
            3, stage, @wgpu.binding_type_read_only_storage_buffer,
          ),
        ],
        None,
      ),
    ) {
    Ok(l) => l
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  // grad_update_b2: 3 bindings (dlogit[r], bias2[rw], lr[r])
  let grad_update_b2_layout = match
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
  // Per-segment bind groups (segment-specific: input, labels, indices buffers differ)
  let seg_layer1_groups : Array[@wgpu.BindGroup] = []
  let seg_layer2_loss_groups : Array[@wgpu.BindGroup] = []
  let seg_indices_groups : Array[@wgpu.BindGroup] = []
  let seg_compute_dh_groups : Array[@wgpu.BindGroup] = []
  let seg_grad_update_w1_groups : Array[@wgpu.BindGroup] = []
  let seg_grad_update_b1_groups : Array[@wgpu.BindGroup] = []
  let seg_grad_update_w2_groups : Array[@wgpu.BindGroup] = []
  let seg_grad_update_b2_groups : Array[@wgpu.BindGroup] = []
  for si = 0; si < segment_info.length(); si = si + 1 {
    let seg_ib = seg_input_bufs[si]
    let seg_lb = seg_labels_bufs[si]
    let seg_idx = seg_indices_bufs[si]
    let seg_db = seg_dataset_buffers_arr[si]
    let seg_lr = seg_loss_resources_arr[si]
    let g_layer1 = match
      @wgpu.device_create_bind_group(
        device,
        @wgpu.bind_group_descriptor(
          layer1_layout,
          [
            @wgpu.bind_group_entry(
              0,
              @wgpu.binding_resource_buffer(
                seg_ib,
                0,
                seg_db.dataset_input_bytes,
              ),
            ),
            @wgpu.bind_group_entry(
              1,
              @wgpu.binding_resource_buffer(seg_idx, 0, seg_db.indices_bytes),
            ),
            @wgpu.bind_group_entry(
              2,
              @wgpu.binding_resource_buffer(
                weight1_buf,
                0,
                resources.weight1.size,
              ),
            ),
            @wgpu.bind_group_entry(
              3,
              @wgpu.binding_resource_buffer(bias1_buf, 0, resources.bias1.size),
            ),
            @wgpu.bind_group_entry(
              4,
              @wgpu.binding_resource_buffer(
                hidden_buf,
                0,
                resources.hidden.size,
              ),
            ),
          ],
          None,
        ),
      ) {
      Ok(g) => g
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    let g_layer2_loss = match
      @wgpu.device_create_bind_group(
        device,
        @wgpu.bind_group_descriptor(
          layer2_loss_layout,
          [
            @wgpu.bind_group_entry(
              0,
              @wgpu.binding_resource_buffer(
                hidden_buf,
                0,
                resources.hidden.size,
              ),
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
              @wgpu.binding_resource_buffer(seg_lb, 0, seg_lr.labels.size),
            ),
            @wgpu.bind_group_entry(
              4,
              @wgpu.binding_resource_buffer(seg_idx, 0, seg_db.indices_bytes),
            ),
            @wgpu.bind_group_entry(
              5,
              @wgpu.binding_resource_buffer(
                loss_buf,
                0,
                loss_resources.loss.size,
              ),
            ),
            @wgpu.bind_group_entry(
              6,
              @wgpu.binding_resource_buffer(
                probs_buf,
                0,
                loss_resources.probs.size,
              ),
            ),
            @wgpu.bind_group_entry(
              7,
              @wgpu.binding_resource_buffer(
                correct_buf,
                0,
                loss_resources.correct.size,
              ),
            ),
          ],
          None,
        ),
      ) {
      Ok(g) => g
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    let g_indices = match
      @wgpu.device_create_bind_group(
        device,
        @wgpu.bind_group_descriptor(
          indices_layout,
          [
            @wgpu.bind_group_entry(
              0,
              @wgpu.binding_resource_buffer(seg_idx, 0, seg_db.indices_bytes),
            ),
            @wgpu.bind_group_entry(
              1,
              @wgpu.binding_resource_buffer(
                shuffle_params_buf,
                0,
                seg_db.shuffle_params_bytes,
              ),
            ),
          ],
          None,
        ),
      ) {
      Ok(g) => g
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    // compute_dh: hidden[r], dlogit(=probs)[r], weight2[r], dh[rw]
    let g_compute_dh = match
      @wgpu.device_create_bind_group(
        device,
        @wgpu.bind_group_descriptor(
          compute_dh_layout,
          [
            @wgpu.bind_group_entry(
              0,
              @wgpu.binding_resource_buffer(
                hidden_buf,
                0,
                resources.hidden.size,
              ),
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
                weight2_buf,
                0,
                resources.weight2.size,
              ),
            ),
            @wgpu.bind_group_entry(
              3,
              @wgpu.binding_resource_buffer(dh_buf, 0, dh_bytes),
            ),
          ],
          None,
        ),
      ) {
      Ok(g) => g
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    // grad_update_w1: input[r], dh[r], indices[r], weight1[rw], lr[r]
    let g_grad_w1 = match
      @wgpu.device_create_bind_group(
        device,
        @wgpu.bind_group_descriptor(
          grad_update_w1_layout,
          [
            @wgpu.bind_group_entry(
              0,
              @wgpu.binding_resource_buffer(
                seg_ib,
                0,
                seg_db.dataset_input_bytes,
              ),
            ),
            @wgpu.bind_group_entry(
              1,
              @wgpu.binding_resource_buffer(dh_buf, 0, dh_bytes),
            ),
            @wgpu.bind_group_entry(
              2,
              @wgpu.binding_resource_buffer(seg_idx, 0, seg_db.indices_bytes),
            ),
            @wgpu.bind_group_entry(
              3,
              @wgpu.binding_resource_buffer(
                weight1_buf,
                0,
                resources.weight1.size,
              ),
            ),
            @wgpu.bind_group_entry(
              4,
              @wgpu.binding_resource_buffer(lr_buf, 0, 4),
            ),
          ],
          None,
        ),
      ) {
      Ok(g) => g
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    // grad_update_b1: dh[r], bias1[rw], lr[r]
    let g_grad_b1 = match
      @wgpu.device_create_bind_group(
        device,
        @wgpu.bind_group_descriptor(
          grad_update_b1_layout,
          [
            @wgpu.bind_group_entry(
              0,
              @wgpu.binding_resource_buffer(dh_buf, 0, dh_bytes),
            ),
            @wgpu.bind_group_entry(
              1,
              @wgpu.binding_resource_buffer(bias1_buf, 0, resources.bias1.size),
            ),
            @wgpu.bind_group_entry(
              2,
              @wgpu.binding_resource_buffer(lr_buf, 0, 4),
            ),
          ],
          None,
        ),
      ) {
      Ok(g) => g
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    // grad_update_w2: hidden[r], dlogit(=probs)[r], weight2[rw], lr[r]
    let g_grad_w2 = match
      @wgpu.device_create_bind_group(
        device,
        @wgpu.bind_group_descriptor(
          grad_update_w2_layout,
          [
            @wgpu.bind_group_entry(
              0,
              @wgpu.binding_resource_buffer(
                hidden_buf,
                0,
                resources.hidden.size,
              ),
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
                weight2_buf,
                0,
                resources.weight2.size,
              ),
            ),
            @wgpu.bind_group_entry(
              3,
              @wgpu.binding_resource_buffer(lr_buf, 0, 4),
            ),
          ],
          None,
        ),
      ) {
      Ok(g) => g
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    // grad_update_b2: dlogit(=probs)[r], bias2[rw], lr[r]
    let g_grad_b2 = match
      @wgpu.device_create_bind_group(
        device,
        @wgpu.bind_group_descriptor(
          grad_update_b2_layout,
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
              @wgpu.binding_resource_buffer(bias2_buf, 0, resources.bias2.size),
            ),
            @wgpu.bind_group_entry(
              2,
              @wgpu.binding_resource_buffer(lr_buf, 0, 4),
            ),
          ],
          None,
        ),
      ) {
      Ok(g) => g
      Err(err) => return Err(wgpu_error_to_string(err))
    }
    seg_layer1_groups.push(g_layer1)
    seg_layer2_loss_groups.push(g_layer2_loss)
    seg_indices_groups.push(g_indices)
    seg_compute_dh_groups.push(g_compute_dh)
    seg_grad_update_w1_groups.push(g_grad_w1)
    seg_grad_update_b1_groups.push(g_grad_b1)
    seg_grad_update_w2_groups.push(g_grad_w2)
    seg_grad_update_b2_groups.push(g_grad_b2)
  }
  // loss_epoch_reduce_group is shared (no segment-specific buffers)
  let loss_epoch_reduce_group = match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(
        loss_epoch_reduce_layout,
        [
          @wgpu.bind_group_entry(
            0,
            @wgpu.binding_resource_buffer(loss_buf, 0, loss_resources.loss.size),
          ),
          @wgpu.bind_group_entry(
            1,
            @wgpu.binding_resource_buffer(
              correct_buf,
              0,
              loss_resources.correct.size,
            ),
          ),
          @wgpu.bind_group_entry(
            2,
            @wgpu.binding_resource_buffer(
              epoch_loss_buf,
              0,
              loss_resources.epoch_loss.size,
            ),
          ),
          @wgpu.bind_group_entry(
            3,
            @wgpu.binding_resource_buffer(
              epoch_correct_buf,
              0,
              loss_resources.epoch_correct.size,
            ),
          ),
          @wgpu.bind_group_entry(
            4,
            @wgpu.binding_resource_buffer(
              epoch_seen_buf,
              0,
              loss_resources.epoch_seen.size,
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
  let layer2_loss_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([layer2_loss_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let indices_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([indices_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_epoch_reduce_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([loss_epoch_reduce_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let compute_dh_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([compute_dh_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_w1_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([grad_update_w1_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_b1_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([grad_update_b1_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_w2_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([grad_update_w2_layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_b2_pipeline_layout = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([grad_update_b2_layout], None),
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
  let layer2_loss_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        layer2_loss_pipeline_layout,
        @wgpu.programmable_stage(layer2_loss_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let indices_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        indices_pipeline_layout,
        @wgpu.programmable_stage(indices_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let loss_epoch_reduce_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        loss_epoch_reduce_pipeline_layout,
        @wgpu.programmable_stage(loss_epoch_reduce_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let compute_dh_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        compute_dh_pipeline_layout,
        @wgpu.programmable_stage(compute_dh_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_w1_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        grad_update_w1_pipeline_layout,
        @wgpu.programmable_stage(grad_update_w1_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_b1_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        grad_update_b1_pipeline_layout,
        @wgpu.programmable_stage(grad_update_b1_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_w2_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        grad_update_w2_pipeline_layout,
        @wgpu.programmable_stage(grad_update_w2_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let grad_update_b2_pipeline = match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        grad_update_b2_pipeline_layout,
        @wgpu.programmable_stage(grad_update_b2_module, "main"),
        None,
      ),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  // Per-segment indices pipelines and dispatch_x
  let seg_indices_pipelines : Array[@wgpu.ComputePipeline] = []
  let seg_indices_dispatch_x : Array[Int] = []
  for si = 0; si < segment_info.length(); si = si + 1 {
    let (_, seg_effective) = segment_info[si]
    seg_indices_dispatch_x.push(ceil_div(seg_effective, workgroup_size))
    if seg_effective == segment_samples {
      seg_indices_pipelines.push(indices_pipeline)
    } else {
      let small_shader = match
        @nn.mlp_dataset_shader_plan(spec, seg_effective, workgroup_size) {
        Ok(p) => p
        Err(err) => return Err(err.to_string())
      }
      let small_module = match
        @wgpu.device_create_shader_module(
          device,
          @wgpu.shader_module_descriptor(small_shader.indices_wgsl, None),
        ) {
        Ok(m) => m
        Err(err) => return Err(wgpu_error_to_string(err))
      }
      let small_pipeline = match
        @wgpu.device_create_compute_pipeline(
          device,
          @wgpu.compute_pipeline_descriptor(
            indices_pipeline_layout,
            @wgpu.programmable_stage(small_module, "main"),
            None,
          ),
        ) {
        Ok(p) => p
        Err(err) => return Err(wgpu_error_to_string(err))
      }
      seg_indices_pipelines.push(small_pipeline)
    }
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
  let zero_f32_bytes = float_array_to_bytes([Float::from_int(0)])
  let metrics : Array[@nn.MlpTrainMetrics] = []
  // Profile accumulators (in ms via UInt64)
  let mut prof_data_upload = 0UL
  let mut prof_forward = 0UL
  let mut prof_loss_reduce = 0UL
  let mut prof_backward = 0UL
  let mut prof_readback = 0UL
  for epoch = 0; epoch < config.epochs; epoch = epoch + 1 {
    if !bench_no_readback {
      ignore(
        @wgpu.queue_write_buffer(
          queue,
          epoch_loss_buf,
          0,
          zero_f32_bytes.to_array(),
        ),
      )
      ignore(
        @wgpu.queue_write_buffer(
          queue,
          epoch_correct_buf,
          0,
          zero_f32_bytes.to_array(),
        ),
      )
      ignore(
        @wgpu.queue_write_buffer(
          queue,
          epoch_seen_buf,
          0,
          zero_f32_bytes.to_array(),
        ),
      )
    }
    let epoch_seed = if config.shuffle { config.seed + epoch } else { 0 }
    for segment = 0; segment < segment_info.length(); segment = segment + 1 {
      let (_, segment_effective) = segment_info[segment]
      // Only shuffle_params + indices dispatch per epoch/segment (no data re-upload)
      let t_upload_start = if profile { @env.now() } else { 0UL }
      let stride = if config.shuffle && segment_effective == segment_samples {
        shuffle_stride(segment_samples, epoch_seed + segment)
      } else {
        1
      }
      let seed = if config.shuffle && segment_effective == segment_samples {
        epoch_seed + segment
      } else {
        0
      }
      let shuffle_bytes = labels_array_to_bytes([seed, stride])
      ignore(
        @wgpu.queue_write_buffer(
          queue,
          shuffle_params_buf,
          0,
          shuffle_bytes.to_array(),
        ),
      )
      ignore(
        @wgpu.device_dispatch_compute(
          device,
          seg_indices_pipelines[segment],
          seg_indices_groups[segment],
          seg_indices_dispatch_x[segment],
        ),
      )
      if profile {
        @wgpu.device_poll(device, true)
        prof_data_upload = prof_data_upload + (@env.now() - t_upload_start)
      }
      let segment_steps = segment_effective / spec.batch_size
      // Reusable 4-byte buffer for batch_start (avoid per-step allocation)
      let batch_start_bytes : Array[Byte] = [b'\x00', b'\x00', b'\x00', b'\x00']
      for step = 0; step < segment_steps; step = step + 1 {
        let batch_start = step * spec.batch_size
        let bs = Int::reinterpret_as_uint(batch_start)
        batch_start_bytes[0] = (bs >> 0).to_byte()
        batch_start_bytes[1] = (bs >> 8).to_byte()
        batch_start_bytes[2] = (bs >> 16).to_byte()
        batch_start_bytes[3] = (bs >> 24).to_byte()
        let t_step_upload = if profile { @env.now() } else { 0UL }
        ignore(
          @wgpu.queue_write_buffer(
            queue,
            seg_indices_bufs[segment],
            0,
            batch_start_bytes,
          ),
        )
        if profile {
          prof_data_upload = prof_data_upload + (@env.now() - t_step_upload)
        }
        if profile {
          // Profile mode: separate dispatches for per-phase timing
          let t_fwd = @env.now()
          @wgpu.dispatch_batch_begin()
          @wgpu.dispatch_batch_add(
            layer1_pipeline,
            seg_layer1_groups[segment],
            dispatch.layer1_dispatch_x,
          )
          @wgpu.dispatch_batch_add(
            layer2_loss_pipeline,
            seg_layer2_loss_groups[segment],
            loss_dispatch_x,
          )
          @wgpu.dispatch_batch_submit(device)
          @wgpu.device_poll(device, true)
          prof_forward = prof_forward + (@env.now() - t_fwd)
          if !bench_no_readback {
            let t_lr = @env.now()
            @wgpu.dispatch_batch_begin()
            @wgpu.dispatch_batch_add(
              loss_epoch_reduce_pipeline, loss_epoch_reduce_group, loss_reduce_dispatch_x,
            )
            @wgpu.dispatch_batch_submit(device)
            @wgpu.device_poll(device, true)
            prof_loss_reduce = prof_loss_reduce + (@env.now() - t_lr)
          }
          let t_bwd = @env.now()
          @wgpu.dispatch_batch_begin()
          @wgpu.dispatch_batch_add(
            compute_dh_pipeline,
            seg_compute_dh_groups[segment],
            train_dispatch.compute_dh_dispatch_x,
          )
          @wgpu.dispatch_batch_add(
            grad_update_w1_pipeline,
            seg_grad_update_w1_groups[segment],
            train_dispatch.grad_update_w1_dispatch_x,
          )
          @wgpu.dispatch_batch_add(
            grad_update_b1_pipeline,
            seg_grad_update_b1_groups[segment],
            train_dispatch.grad_update_b1_dispatch_x,
          )
          @wgpu.dispatch_batch_add(
            grad_update_w2_pipeline,
            seg_grad_update_w2_groups[segment],
            train_dispatch.grad_update_w2_dispatch_x,
          )
          @wgpu.dispatch_batch_add(
            grad_update_b2_pipeline,
            seg_grad_update_b2_groups[segment],
            train_dispatch.grad_update_b2_dispatch_x,
          )
          @wgpu.dispatch_batch_submit(device)
          @wgpu.device_poll(device, true)
          prof_backward = prof_backward + (@env.now() - t_bwd)
        } else {
          // Normal mode: single batch dispatch for all phases
          @wgpu.dispatch_batch_begin()
          @wgpu.dispatch_batch_add(
            layer1_pipeline,
            seg_layer1_groups[segment],
            dispatch.layer1_dispatch_x,
          )
          @wgpu.dispatch_batch_add(
            layer2_loss_pipeline,
            seg_layer2_loss_groups[segment],
            loss_dispatch_x,
          )
          if !bench_no_readback {
            @wgpu.dispatch_batch_add(
              loss_epoch_reduce_pipeline, loss_epoch_reduce_group, loss_reduce_dispatch_x,
            )
          }
          @wgpu.dispatch_batch_add(
            compute_dh_pipeline,
            seg_compute_dh_groups[segment],
            train_dispatch.compute_dh_dispatch_x,
          )
          @wgpu.dispatch_batch_add(
            grad_update_w1_pipeline,
            seg_grad_update_w1_groups[segment],
            train_dispatch.grad_update_w1_dispatch_x,
          )
          @wgpu.dispatch_batch_add(
            grad_update_b1_pipeline,
            seg_grad_update_b1_groups[segment],
            train_dispatch.grad_update_b1_dispatch_x,
          )
          @wgpu.dispatch_batch_add(
            grad_update_w2_pipeline,
            seg_grad_update_w2_groups[segment],
            train_dispatch.grad_update_w2_dispatch_x,
          )
          @wgpu.dispatch_batch_add(
            grad_update_b2_pipeline,
            seg_grad_update_b2_groups[segment],
            train_dispatch.grad_update_b2_dispatch_x,
          )
          @wgpu.dispatch_batch_submit(device)
        }
      }
    }
    if !bench_no_readback {
      let t_rb = if profile { @env.now() } else { 0UL }
      let loss_bytes = match
        @wgpu.device_read_buffer_bytes(
          device,
          epoch_loss_buf,
          loss_resources.epoch_loss.size,
        ) {
        Ok(b) => b
        Err(err) => return Err(wgpu_error_to_string(err))
      }
      let correct_bytes = match
        @wgpu.device_read_buffer_bytes(
          device,
          epoch_correct_buf,
          loss_resources.epoch_correct.size,
        ) {
        Ok(b) => b
        Err(err) => return Err(wgpu_error_to_string(err))
      }
      let seen_bytes = match
        @wgpu.device_read_buffer_bytes(
          device,
          epoch_seen_buf,
          loss_resources.epoch_seen.size,
        ) {
        Ok(b) => b
        Err(err) => return Err(wgpu_error_to_string(err))
      }
      if profile {
        prof_readback = prof_readback + (@env.now() - t_rb)
      }
      let seen = read_f32_le(seen_bytes, 0)
      if seen > 0 {
        let epoch_loss = read_f32_le(loss_bytes, 0) / seen
        let epoch_acc = read_f32_le(correct_bytes, 0) / seen
        metrics.push(@nn.mlp_train_metrics(epoch_loss, epoch_acc))
      }
    }
  }
  if profile {
    let total = prof_data_upload +
      prof_forward +
      prof_loss_reduce +
      prof_backward +
      prof_readback
    println(
      "profile: forward_ms=" +
      Float::from_uint64(prof_forward).to_string() +
      " backward_ms=" +
      Float::from_uint64(prof_backward).to_string() +
      " loss_reduce_ms=" +
      Float::from_uint64(prof_loss_reduce).to_string() +
      " data_upload_ms=" +
      Float::from_uint64(prof_data_upload).to_string() +
      " readback_ms=" +
      Float::from_uint64(prof_readback).to_string() +
      " total_ms=" +
      Float::from_uint64(total).to_string(),
    )
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
/// BLAS-optimized CPU training using C-side buffers
/// Uses batch BLAS operations for both forward and backward pass
fn blas_train(
  spec : @nn.MlpSpec,
  params : @nn.MlpParams,
  dataset : @nn.MlpDataset,
  config : @nn.MlpTrainConfig,
) -> Result[@nn.MlpTrainResult, String] {
  if config.epochs <= 0 {
    return Err("epochs must be > 0")
  }
  if config.batch_size <= 0 {
    return Err("batch_size must be > 0")
  }
  if dataset.count <= 0 {
    return Err("dataset empty")
  }

  // Allocate training buffers (all computation happens in C memory)
  let bufs = @blas.mlp_train_buffers_create(
    config.batch_size,
    spec.input_size,
    spec.hidden_size,
    spec.output_size,
  )

  // Initialize weights in C buffers
  @blas.mlp_train_buffers_init_weights(
    bufs,
    params.weight1,
    params.bias1,
    params.weight2,
    params.bias2,
  )

  // Working arrays
  let zero = Float::from_int(0)
  let metrics : Array[@nn.MlpTrainMetrics] = []
  let indices = Array::makei(dataset.count, fn(i) { i })

  // Batch arrays for input and labels
  let batch_input = Array::make(config.batch_size * spec.input_size, zero)
  let batch_labels = Array::make(config.batch_size, 0)
  for _epoch = 0; _epoch < config.epochs; _epoch = _epoch + 1 {
    if config.shuffle {
      shuffle_indices_blas(indices, config.seed + _epoch)
    }
    let mut loss_sum = zero
    let mut correct = 0
    let mut start = 0
    while start < dataset.count {
      let end = if start + config.batch_size <= dataset.count {
        start + config.batch_size
      } else {
        // Skip incomplete batch at end
        break
      }
      let batch_count = end - start

      // Prepare batch data
      for bi = 0; bi < batch_count; bi = bi + 1 {
        let idx = indices[start + bi]
        batch_labels[bi] = dataset.labels[idx]
        let input_offset = idx * spec.input_size
        for i = 0; i < spec.input_size; i = i + 1 {
          batch_input[bi * spec.input_size + i] = dataset.inputs[input_offset +
            i]
        }
      }

      // Run complete training step in C
      let (batch_loss, batch_correct) = @blas.mlp_train_step(
        bufs,
        batch_input,
        batch_labels,
        config.learning_rate,
      )
      loss_sum = loss_sum + batch_loss
      correct = correct + batch_correct
      start = end
    }
    let count_f = Float::from_int(dataset.count)
    let loss = loss_sum / count_f
    let accuracy = Float::from_int(correct) / count_f
    metrics.push(@nn.mlp_train_metrics(loss, accuracy))
  }

  // Copy final weights back
  @blas.mlp_train_buffers_get_weights(
    bufs,
    params.weight1,
    params.bias1,
    params.weight2,
    params.bias2,
  )

  // Free C buffers
  @blas.mlp_train_buffers_free(bufs)
  Ok(@nn.mlp_train_result(params, metrics))
}

///|
fn shuffle_indices_blas(indices : Array[Int], seed : Int) -> Unit {
  let a = 1664525
  let c = 1013904223
  let mut state = seed
  let mut i = indices.length() - 1
  while i > 0 {
    state = state * a + c
    let j = if state < 0 { -state } else { state }
    let jmod = j % (i + 1)
    let tmp = indices[i]
    indices[i] = indices[jmod]
    indices[jmod] = tmp
    i = i - 1
  }
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
  workgroup_size : Int,
  backend : TrainBackend,
  bench_no_readback : Bool,
  use_blas : Bool,
  json : Bool,
  profile : Bool,
) -> Result[TrainReport, String] {
  match backend {
    Cpu => {
      let train_result = if use_blas {
        blas_train(spec, params, dataset, config)
      } else {
        match @nn_cpu.mlp_train(spec, params, dataset, config) {
          Ok(r) => Ok(r)
          Err(err) => Err(err.to_string())
        }
      }
      match train_result {
        Ok(r) => Ok({ backend: Cpu, result: r })
        Err(err) => Err(err)
      }
    }
    Gpu => {
      if !@wgpu.is_supported() {
        return Err("wgpu not supported".to_string())
      }
      let train_result = gpu_train(
        spec, params, dataset, config, workgroup_size, bench_no_readback, profile,
      )
      match train_result {
        Ok(r) => Ok({ backend: Gpu, result: r })
        Err(err) => Err(err)
      }
    }
    GpuTransformer => {
      gpu_transformer_test()
      // Return a dummy result since this is a separate test mode
      let dummy_params = match @nn.mlp_init_params(spec, 0) {
        Ok(p) => p
        Err(err) => return Err(err.to_string())
      }
      return Ok({
        backend: GpuTransformer,
        result: @nn.mlp_train_result(dummy_params, []),
      })
    }
    Auto => {
      if @wgpu.is_supported() {
        let train_result = gpu_train(
          spec, params, dataset, config, workgroup_size, bench_no_readback, profile,
        )
        match train_result {
          Ok(r) => return Ok({ backend: Gpu, result: r })
          Err(err) => {
            print_warn_line(json, "gpu train error: " + err)
            print_warn_line(json, "fallback to cpu")
          }
        }
      }
      let train_result = if use_blas {
        blas_train(spec, params, dataset, config)
      } else {
        match @nn_cpu.mlp_train(spec, params, dataset, config) {
          Ok(r) => Ok(r)
          Err(err) => Err(err.to_string())
        }
      }
      match train_result {
        Ok(r) => Ok({ backend: Cpu, result: r })
        Err(err) => Err(err)
      }
    }
    Vit => Err("vit backend uses separate training path")
  }
}

///|
async fn run_vit_train(args : TrainArgs) -> Unit {
  let base = "data/mnist"
  let train_images_path = mnist_path(base, "train-images-idx3-ubyte")
  let train_labels_path = mnist_path(base, "train-labels-idx1-ubyte")
  let test_images_path = mnist_path(base, "t10k-images-idx3-ubyte")
  let test_labels_path = mnist_path(base, "t10k-labels-idx1-ubyte")
  // Load training data
  let train_dataset = @mnist.mnist_load_mlp_dataset(
    train_images_path,
    train_labels_path,
    args.limit,
  )
  let train = match train_dataset {
    Ok(ds) => ds
    Err(err) => {
      print_error_line(args.json, "mnist train load error: " + err.to_string())
      return
    }
  }
  // Load test data
  let test_dataset = @mnist.mnist_load_mlp_dataset(
    test_images_path,
    test_labels_path,
    None,
  )
  let test_set = match test_dataset {
    Ok(ds) => ds
    Err(err) => {
      print_error_line(args.json, "mnist test load error: " + err.to_string())
      return
    }
  }
  let epochs = match args.epochs {
    Some(v) => v
    None => 10
  }
  let batch_size = match args.batch_size {
    Some(v) => v
    None => 64
  }
  let lr = Float::from_double(0.001)
  let config = match @tensor.vit_config(28, 7, 64, 4, 2, 128, 10) {
    Ok(c) => c
    Err(err) => {
      print_error_line(args.json, "vit config error: " + err)
      return
    }
  }
  // Print config
  if args.json {
    println(
      json_object([
        json_field("type", json_string("config")),
        json_field("backend", json_string("vit")),
        json_field("model", json_string("vit(28,7,64,4,2,128,10)")),
        json_field("train_samples", train.count.to_string()),
        json_field("test_samples", test_set.count.to_string()),
        json_field("epochs", epochs.to_string()),
        json_field("batch", batch_size.to_string()),
        json_field("lr", lr.to_string()),
      ]),
    )
  } else {
    println(
      "config: model=vit(28,7,64,4,2,128,10) train_samples=" +
      train.count.to_string() +
      " test_samples=" +
      test_set.count.to_string() +
      " epochs=" +
      epochs.to_string() +
      " batch=" +
      batch_size.to_string() +
      " lr=" +
      lr.to_string() +
      " backend=vit",
    )
  }
  // Convert MlpDataset to Tensor
  let train_images = @tensor.tensor_new(
    @tensor.shape_new_unchecked([train.count, 784]),
    train.inputs,
  ).unwrap()
  let (params, metrics) = @tensor.vit_train_with_eval(
    train_images,
    train.labels,
    train.count,
    config,
    epochs,
    batch_size,
    lr,
    42,
  )
  // Print epoch metrics
  for i = 0; i < metrics.length(); i = i + 1 {
    let (loss, acc) = metrics[i]
    if args.json {
      println(
        json_object([
          json_field("type", json_string("epoch")),
          json_field("split", json_string("train")),
          json_field("epoch", (i + 1).to_string()),
          json_field("loss", loss.to_string()),
          json_field("acc", acc.to_string()),
        ]),
      )
    } else {
      println(
        "epoch " +
        (i + 1).to_string() +
        " loss=" +
        loss.to_string() +
        " acc=" +
        acc.to_string(),
      )
    }
  }
  // Evaluate on test set
  let test_count = test_set.count
  let test_images = @tensor.tensor_new(
    @tensor.shape_new_unchecked([test_count, 784]),
    test_set.inputs,
  ).unwrap()
  // Evaluate in batches to avoid OOM
  let eval_batch = 256
  let mut total_loss = Float::from_int(0)
  let mut total_correct = 0
  let mut offset = 0
  while offset < test_count {
    let bs = if offset + eval_batch <= test_count {
      eval_batch
    } else {
      test_count - offset
    }
    let batch_data = Array::make(bs * 784, Float::from_int(0))
    let batch_labels : Array[Int] = Array::make(bs, 0)
    let tc = test_images.contiguous()
    for b = 0; b < bs; b = b + 1 {
      let src = (offset + b) * 784
      let dst = b * 784
      for d = 0; d < 784; d = d + 1 {
        batch_data[dst + d] = tc.data[src + d]
      }
      batch_labels[b] = test_set.labels[offset + b]
    }
    let batch_tensor = @tensor.tensor_new(
      @tensor.shape_new_unchecked([bs, 784]),
      batch_data,
    ).unwrap()
    match @tensor.vit_eval(batch_tensor, params, config, batch_labels, bs) {
      Ok((loss, acc)) => {
        total_loss = total_loss + loss * Float::from_int(bs)
        let correct_in_batch = ((acc * Float::from_int(bs)).to_double() + 0.5)
          |> Double::to_int
        total_correct = total_correct + correct_in_batch
      }
      Err(err) => {
        print_error_line(args.json, "eval error: " + err)
        return
      }
    }
    offset = offset + bs
  }
  let test_loss = total_loss / Float::from_int(test_count)
  let test_acc = Float::from_int(total_correct) / Float::from_int(test_count)
  if args.json {
    println(
      json_object([
        json_field("type", json_string("result")),
        json_field("split", json_string("test")),
        json_field("backend", json_string("vit")),
        json_field("loss", test_loss.to_string()),
        json_field("acc", test_acc.to_string()),
      ]),
    )
  } else {
    println(
      "result: backend=vit split=test loss=" +
      test_loss.to_string() +
      " acc=" +
      test_acc.to_string(),
    )
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
  if args.backend is Vit {
    run_vit_train(args)
    return
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
  let batch_size = match args.batch_size {
    Some(v) => v
    None => base_cfg.batch_size
  }
  let epochs = match args.epochs {
    Some(v) => v
    None => base_cfg.epochs
  }
  let cfg = @nn.mlp_train_config(
    epochs,
    batch_size,
    base_cfg.learning_rate,
    base_cfg.shuffle,
    base_cfg.seed,
  )
  let workgroup_size = match args.workgroup_size {
    Some(v) => v
    None => 64
  }
  if workgroup_size <= 0 {
    print_error_line(args.json, "workgroup_size must be > 0")
    return
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
    GpuTransformer => "n/a"
    Vit => "n/a"
  }
  print_train_config(
    args,
    cfg,
    train,
    test_set,
    spec,
    workgroup_size,
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
    workgroup_size,
    args.backend,
    args.bench_no_readback,
    args.use_blas,
    args.json,
    args.profile,
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
      Gpu | GpuTransformer =>
        if report.result.metrics.length() == 0 {
          print_warn_line(args.json, "metrics skipped: --bench-no-readback")
        }
      Cpu =>
        print_warn_line(
          args.json,
          "--bench-no-readback ignored for cpu backend",
        )
      Auto | Vit => ()
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
  let eval = @nn_cpu.mlp_eval(spec, trained.params, test_set)
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
