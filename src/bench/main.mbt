///|
struct BenchArgs {
  input_size : Int
  hidden_size : Int
  output_size : Int
  batch_size : Int
  warmup : Int
  iters : Int
} derive(Show, Eq)

///|
fn bench_args_default() -> BenchArgs {
  {
    input_size: 784,
    hidden_size: 128,
    output_size: 10,
    batch_size: 128,
    warmup: 20,
    iters: 200,
  }
}

///|
fn print_usage(program : String) -> Unit {
  println(
    "Usage: " +
    program +
    " [--input N] [--hidden N] [--output N] [--batch N] [--warmup N] [--iters N]",
  )
  println("  --input N    input size (default 784)")
  println("  --hidden N   hidden size (default 128)")
  println("  --output N   output size (default 10)")
  println("  --batch N    batch size (default 128)")
  println("  --warmup N   warmup iterations (default 20)")
  println("  --iters N    measure iterations (default 200)")
}

///|
fn parse_positive_int(name : String, value : String) -> Result[Int, String] {
  let parsed = try? @strconv.parse_int(value)
  match parsed {
    Ok(v) => if v > 0 { Ok(v) } else { Err(name + " must be > 0") }
    Err(err) => Err("invalid " + name + ": " + err.to_string())
  }
}

///|
fn parse_nonnegative_int(name : String, value : String) -> Result[Int, String] {
  let parsed = try? @strconv.parse_int(value)
  match parsed {
    Ok(v) => if v >= 0 { Ok(v) } else { Err(name + " must be >= 0") }
    Err(err) => Err("invalid " + name + ": " + err.to_string())
  }
}

///|
fn parse_args(args : Array[String]) -> Result[BenchArgs, String] {
  let default = bench_args_default()
  let mut input_size = default.input_size
  let mut hidden_size = default.hidden_size
  let mut output_size = default.output_size
  let mut batch_size = default.batch_size
  let mut warmup = default.warmup
  let mut iters = default.iters
  let mut i = 1
  while i < args.length() {
    let arg = args[i]
    if arg == "--help" || arg == "-h" {
      return Err("help")
    }
    if arg == "--blas" {
      i = i + 1
      continue
    }
    if arg == "--fused" {
      i = i + 1
      continue
    }
    if arg == "--input" {
      if i + 1 >= args.length() {
        return Err("missing value for --input")
      }
      match parse_positive_int("input", args[i + 1]) {
        Ok(v) => input_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--input=") {
      let view = arg.sub(start=8) catch {
        _ => return Err("invalid --input value")
      }
      match parse_positive_int("input", view.to_string()) {
        Ok(v) => input_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--hidden" {
      if i + 1 >= args.length() {
        return Err("missing value for --hidden")
      }
      match parse_positive_int("hidden", args[i + 1]) {
        Ok(v) => hidden_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--hidden=") {
      let view = arg.sub(start=9) catch {
        _ => return Err("invalid --hidden value")
      }
      match parse_positive_int("hidden", view.to_string()) {
        Ok(v) => hidden_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--output" {
      if i + 1 >= args.length() {
        return Err("missing value for --output")
      }
      match parse_positive_int("output", args[i + 1]) {
        Ok(v) => output_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--output=") {
      let view = arg.sub(start=9) catch {
        _ => return Err("invalid --output value")
      }
      match parse_positive_int("output", view.to_string()) {
        Ok(v) => output_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--batch" {
      if i + 1 >= args.length() {
        return Err("missing value for --batch")
      }
      match parse_positive_int("batch", args[i + 1]) {
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
      match parse_positive_int("batch", view.to_string()) {
        Ok(v) => batch_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--warmup" {
      if i + 1 >= args.length() {
        return Err("missing value for --warmup")
      }
      match parse_nonnegative_int("warmup", args[i + 1]) {
        Ok(v) => warmup = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--warmup=") {
      let view = arg.sub(start=9) catch {
        _ => return Err("invalid --warmup value")
      }
      match parse_nonnegative_int("warmup", view.to_string()) {
        Ok(v) => warmup = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--iters" {
      if i + 1 >= args.length() {
        return Err("missing value for --iters")
      }
      match parse_positive_int("iters", args[i + 1]) {
        Ok(v) => iters = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--iters=") {
      let view = arg.sub(start=8) catch {
        _ => return Err("invalid --iters value")
      }
      match parse_positive_int("iters", view.to_string()) {
        Ok(v) => iters = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    return Err("unknown option: " + arg)
  }
  Ok({ input_size, hidden_size, output_size, batch_size, warmup, iters })
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
fn make_labels(batch : Int, output : Int) -> Array[Int] {
  let total = if batch <= 0 { 0 } else { batch }
  Array::makei(total, fn(i) { i % output })
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
fn run_bench(cfg : BenchArgs) -> Unit {
  let spec = match
    @nn.mlp_spec_new(
      cfg.input_size,
      cfg.hidden_size,
      cfg.output_size,
      cfg.batch_size,
    ) {
    Ok(s) => s
    Err(err) => {
      println("spec error: " + err.to_string())
      return
    }
  }
  let params = match
    @nn.mlp_init_params_with_policy(spec, 0, @nn.mlp_init_policy_deterministic) {
    Ok(p) => p
    Err(err) => {
      println("init error: " + err.to_string())
      return
    }
  }
  let inputs = seeded_values(0, spec.input_size * spec.batch_size, 0)
  let labels = make_labels(spec.batch_size, spec.output_size)
  let mut i = 0
  let mut last_loss = Float::from_int(0)
  while i < cfg.warmup {
    let forward = match @nn.mlp_forward(spec, params, inputs) {
      Ok(r) => r
      Err(err) => {
        println("forward error: " + err.to_string())
        return
      }
    }
    let probs = softmax_batch(forward.output, spec.batch_size, spec.output_size)
    last_loss = loss_from_probs(probs, labels, spec.output_size)
    i = i + 1
  }
  let samples = Array::make(cfg.iters, Float::from_int(0))
  i = 0
  while i < cfg.iters {
    let start = @env.now()
    let forward = match @nn.mlp_forward(spec, params, inputs) {
      Ok(r) => r
      Err(err) => {
        println("forward error: " + err.to_string())
        return
      }
    }
    let probs = softmax_batch(forward.output, spec.batch_size, spec.output_size)
    last_loss = loss_from_probs(probs, labels, spec.output_size)
    let end = @env.now()
    let elapsed = end - start
    samples[i] = Float::from_uint64(elapsed)
    i = i + 1
  }
  let stats = stats_from_samples(samples)
  let avg = stats[0]
  let p50 = stats[1]
  let p95 = stats[2]
  println(
    "{" +
    "\"avg_ms\":" +
    avg.to_string() +
    ",\"p50_ms\":" +
    p50.to_string() +
    ",\"p95_ms\":" +
    p95.to_string() +
    ",\"last_loss\":" +
    last_loss.to_string() +
    ",\"input\":" +
    cfg.input_size.to_string() +
    ",\"hidden\":" +
    cfg.hidden_size.to_string() +
    ",\"output\":" +
    cfg.output_size.to_string() +
    ",\"batch\":" +
    cfg.batch_size.to_string() +
    ",\"warmup\":" +
    cfg.warmup.to_string() +
    ",\"iters\":" +
    cfg.iters.to_string() +
    "}",
  )
}

///|
fn run_blas_bench(cfg : BenchArgs) -> Unit {
  // Setup matrices (same dimensions as MLP layer 1)
  let zero = Float::from_int(0)

  // Single-sample data
  let input_data = seeded_values(0, cfg.input_size, 0)
  let weight1_data = seeded_values(1, cfg.input_size * cfg.hidden_size, 100)
  let bias1_data = seeded_values(2, cfg.hidden_size, 200)
  let hidden_data = Array::make(cfg.hidden_size, zero)

  // Batch data
  let batch_input = seeded_values(0, cfg.batch_size * cfg.input_size, 0)
  let batch_hidden = Array::make(cfg.batch_size * cfg.hidden_size, zero)
  let batch_hidden_pure = Array::make(cfg.batch_size * cfg.hidden_size, zero)

  // Create numbt structures for single-sample
  let input = @numbt.vec_from_array(input_data)
  let mat = @numbt.mat_view(weight1_data, cfg.input_size, cfg.hidden_size)
  let bias = @numbt.vec_from_array(bias1_data)
  let hidden = @numbt.vec_from_array(hidden_data)

  // Clone for BLAS single version
  let hidden_blas_data = Array::make(cfg.hidden_size, zero)
  let hidden_blas = @numbt.vec_from_array(hidden_blas_data)

  // Warmup (sgemm only)
  let mut i = 0
  while i < cfg.warmup {
    @numbt.batch_matmul_bias_relu(
      batch_input,
      weight1_data,
      bias1_data,
      batch_hidden,
      cfg.batch_size,
      cfg.input_size,
      cfg.hidden_size,
    )
    i = i + 1
  }
  // Suppress unused variable warnings
  ignore(input)
  ignore(mat)
  ignore(bias)
  ignore(hidden)
  ignore(hidden_blas)
  ignore(hidden_data)
  ignore(input_data)
  ignore(batch_hidden_pure)

  // Prepare second layer
  let weight2_data = seeded_values(3, cfg.hidden_size * cfg.output_size, 300)
  let bias2_data = seeded_values(4, cfg.output_size, 400)
  let batch_output = Array::make(cfg.batch_size * cfg.output_size, zero)

  // Benchmark: Full 2-layer MLP forward with BLAS sgemm
  let sgemm_start = @env.now()
  i = 0
  while i < cfg.iters {
    // Layer 1: batch_input @ weight1 + bias1 -> batch_hidden (with ReLU)
    @numbt.batch_matmul_bias_relu(
      batch_input,
      weight1_data,
      bias1_data,
      batch_hidden,
      cfg.batch_size,
      cfg.input_size,
      cfg.hidden_size,
    )
    // Layer 2: batch_hidden @ weight2 + bias2 -> batch_output (no activation)
    @numbt.batch_matmul_bias(
      batch_hidden,
      weight2_data,
      bias2_data,
      batch_output,
      cfg.batch_size,
      cfg.hidden_size,
      cfg.output_size,
    )
    i = i + 1
  }
  let sgemm_end = @env.now()
  let sgemm_total_ms = Float::from_uint64(sgemm_end - sgemm_start)

  // Calculate per-batch time in microseconds
  let sgemm_per_batch_us = sgemm_total_ms * Float::from_int(1000) / Float::from_int(
      cfg.iters,
    )

  println(
    "{\"type\":\"blas_mlp_forward_bench\"" +
    ",\"input\":" +
    cfg.input_size.to_string() +
    ",\"hidden\":" +
    cfg.hidden_size.to_string() +
    ",\"output\":" +
    cfg.output_size.to_string() +
    ",\"batch\":" +
    cfg.batch_size.to_string() +
    ",\"iters\":" +
    cfg.iters.to_string() +
    ",\"sgemm_total_ms\":" +
    sgemm_total_ms.to_string() +
    ",\"sgemm_per_batch_us\":" +
    sgemm_per_batch_us.to_string() +
    "}",
  )
}

///|
fn run_blas_fused_bench(cfg : BenchArgs) -> Unit {
  // Setup data
  let zero = Float::from_int(0)
  let batch_input = seeded_values(0, cfg.batch_size * cfg.input_size, 0)
  let weight1_data = seeded_values(1, cfg.input_size * cfg.hidden_size, 100)
  let bias1_data = seeded_values(2, cfg.hidden_size, 200)
  let weight2_data = seeded_values(3, cfg.hidden_size * cfg.output_size, 300)
  let bias2_data = seeded_values(4, cfg.output_size, 400)

  // Create C-side buffers and initialize (single copy)
  let bufs = @blas.mlp_buffers_create(
    cfg.batch_size,
    cfg.input_size,
    cfg.hidden_size,
    cfg.output_size,
  )
  @blas.mlp_buffers_init(
    bufs,
    batch_input,
    weight1_data,
    bias1_data,
    weight2_data,
    bias2_data,
  )

  // Warmup
  let mut i = 0
  while i < cfg.warmup {
    @blas.mlp_forward_fused(bufs)
    i = i + 1
  }

  // Measure multiple batches to get meaningful timing
  // Run in groups of 1000 to accumulate timing
  let group_size = 1000
  let groups = if cfg.iters >= group_size {
    cfg.iters / group_size
  } else {
    1
  }
  let actual_iters = groups * group_size

  // Benchmark: Zero-copy forward pass
  let fused_start = @env.now()
  i = 0
  while i < actual_iters {
    @blas.mlp_forward_fused(bufs)
    i = i + 1
  }
  let fused_end = @env.now()
  let fused_total_ms = Float::from_uint64(fused_end - fused_start)

  // Get output for verification
  let fused_output = Array::make(cfg.batch_size * cfg.output_size, zero)
  @blas.mlp_buffers_get_output(bufs, fused_output)

  // Compare with old BLAS (with copy overhead)
  let batch_hidden = Array::make(cfg.batch_size * cfg.hidden_size, zero)
  let batch_output = Array::make(cfg.batch_size * cfg.output_size, zero)

  let copy_start = @env.now()
  i = 0
  while i < actual_iters {
    // Layer 1: batch_input @ weight1 + bias1 -> batch_hidden (with ReLU)
    @numbt.batch_matmul_bias_relu(
      batch_input,
      weight1_data,
      bias1_data,
      batch_hidden,
      cfg.batch_size,
      cfg.input_size,
      cfg.hidden_size,
    )
    // Layer 2: batch_hidden @ weight2 + bias2 -> batch_output (no activation)
    @numbt.batch_matmul_bias(
      batch_hidden,
      weight2_data,
      bias2_data,
      batch_output,
      cfg.batch_size,
      cfg.hidden_size,
      cfg.output_size,
    )
    i = i + 1
  }
  let copy_end = @env.now()
  let copy_total_ms = Float::from_uint64(copy_end - copy_start)

  // Verify outputs match (first few values)
  let mut output_match = true
  let check_count = if cfg.output_size < 3 { cfg.output_size } else { 3 }
  i = 0
  while i < check_count {
    let diff = (fused_output[i] - batch_output[i]).abs()
    if diff > Float::from_int(1) / Float::from_int(1000) {
      output_match = false
    }
    i = i + 1
  }

  // Calculate per-batch time in microseconds
  let fused_per_batch_us = fused_total_ms * Float::from_int(1000) / Float::from_int(
      actual_iters,
    )
  let copy_per_batch_us = copy_total_ms * Float::from_int(1000) / Float::from_int(
      actual_iters,
    )
  let speedup = if fused_total_ms > Float::from_int(0) {
    copy_total_ms / fused_total_ms
  } else {
    Float::from_int(-1) // indicates fused was too fast to measure
  }

  // Clean up
  @blas.mlp_buffers_free(bufs)

  println(
    "{\"type\":\"blas_fused_bench\"" +
    ",\"input\":" +
    cfg.input_size.to_string() +
    ",\"hidden\":" +
    cfg.hidden_size.to_string() +
    ",\"output\":" +
    cfg.output_size.to_string() +
    ",\"batch\":" +
    cfg.batch_size.to_string() +
    ",\"actual_iters\":" +
    actual_iters.to_string() +
    ",\"fused_total_ms\":" +
    fused_total_ms.to_string() +
    ",\"fused_per_batch_us\":" +
    fused_per_batch_us.to_string() +
    ",\"copy_total_ms\":" +
    copy_total_ms.to_string() +
    ",\"copy_per_batch_us\":" +
    copy_per_batch_us.to_string() +
    ",\"speedup\":" +
    speedup.to_string() +
    ",\"output_match\":" +
    output_match.to_string() +
    ",\"sample_fused_0\":" +
    fused_output[0].to_string() +
    ",\"sample_copy_0\":" +
    batch_output[0].to_string() +
    "}",
  )
}

///|
fn main {
  let args = @env.args()
  let program = if args.length() > 0 { args[0] } else { "bench" }

  // Check for --blas and --fused flags
  let mut use_blas = false
  let mut use_fused = false
  let mut i = 1
  while i < args.length() {
    if args[i] == "--blas" {
      use_blas = true
    }
    if args[i] == "--fused" {
      use_fused = true
    }
    i = i + 1
  }

  let cfg_result = parse_args(args)
  let cfg = match cfg_result {
    Ok(c) => c
    Err(msg) => {
      if msg != "help" {
        println("error: " + msg)
      }
      print_usage(program)
      return
    }
  }

  if use_fused {
    run_blas_fused_bench(cfg)
  } else if use_blas {
    run_blas_bench(cfg)
  } else {
    run_bench(cfg)
  }
}
