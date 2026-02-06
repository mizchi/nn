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
    " [--blas|--fused|--numbt] [--warmup N] [--iters N]",
  )
  println("  --blas       BLAS MLP forward benchmark")
  println("  --fused      Fused BLAS MLP benchmark")
  println("  --numbt      numbt operations benchmark (sort, cumsum, outer, etc.)")
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
    if arg == "--numbt" {
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
    let forward = match @nn_cpu.mlp_forward(spec, params, inputs) {
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
    let forward = match @nn_cpu.mlp_forward(spec, params, inputs) {
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
fn format_time_ms(total_ms : Float, iters : Int) -> String {
  // Convert to ms per operation with 3 decimal places
  let ms_per_op = total_ms / Float::from_int(iters)
  // Format with 3 decimal places
  let scaled = (ms_per_op * Float::from_int(1000)).floor().to_int()
  let whole = scaled / 1000
  let frac = scaled % 1000
  let frac_str = if frac < 10 {
    "00" + frac.to_string()
  } else if frac < 100 {
    "0" + frac.to_string()
  } else {
    frac.to_string()
  }
  whole.to_string() + "." + frac_str
}

///|
fn run_numbt_bench(cfg : BenchArgs) -> Unit {
  let n_small = 1024
  let n_mat = 100

  // Generate test data
  let vec_data = seeded_values(0, n_small, 0)
  let vec_a = @numbt.vec_from_array(vec_data)
  let vec_b = @numbt.vec_from_array(seeded_values(1, n_small, 100))

  let mat_data = seeded_values(2, n_mat * n_mat, 200)
  let mat = @numbt.mat_view(mat_data, n_mat, n_mat)

  let vec_100 = @numbt.vec_from_array(seeded_values(3, n_mat, 300))

  // Warmup
  let mut i = 0
  while i < cfg.warmup {
    ignore(@numbt.vec_sort(vec_a))
    i = i + 1
  }

  println("============================================================")
  println("numbt Benchmark (MoonBit + BLAS)")
  println("============================================================")
  println("")
  println("--- Vector operations (N=" + n_small.to_string() + ") ---")

  // --- Vector basic ops ---
  let add_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(vec_a + vec_b)
    i = i + 1
  }
  let add_ms = Float::from_uint64(@env.now() - add_start)
  println("vec_add: " + format_time_ms(add_ms, cfg.iters) + " ms")

  let dot_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_dot(vec_a, vec_b))
    i = i + 1
  }
  let dot_ms = Float::from_uint64(@env.now() - dot_start)
  println("vec_dot: " + format_time_ms(dot_ms, cfg.iters) + " ms")

  let sum_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_sum(vec_a))
    i = i + 1
  }
  let sum_ms = Float::from_uint64(@env.now() - sum_start)
  println("vec_sum: " + format_time_ms(sum_ms, cfg.iters) + " ms")

  let mean_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_mean(vec_a))
    i = i + 1
  }
  let mean_ms = Float::from_uint64(@env.now() - mean_start)
  println("vec_mean: " + format_time_ms(mean_ms, cfg.iters) + " ms")

  let std_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_std(vec_a))
    i = i + 1
  }
  let std_ms = Float::from_uint64(@env.now() - std_start)
  println("vec_std: " + format_time_ms(std_ms, cfg.iters) + " ms")

  let exp_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_exp(vec_a))
    i = i + 1
  }
  let exp_ms = Float::from_uint64(@env.now() - exp_start)
  println("vec_exp: " + format_time_ms(exp_ms, cfg.iters) + " ms")

  // --- Sort operations ---
  println("")
  println("--- Sort operations (N=" + n_small.to_string() + ") ---")

  let sort_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_sort(vec_a))
    i = i + 1
  }
  let sort_ms = Float::from_uint64(@env.now() - sort_start)
  println("vec_sort: " + format_time_ms(sort_ms, cfg.iters) + " ms")

  let argsort_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_argsort(vec_a))
    i = i + 1
  }
  let argsort_ms = Float::from_uint64(@env.now() - argsort_start)
  println("vec_argsort: " + format_time_ms(argsort_ms, cfg.iters) + " ms")

  // --- Cumulative operations ---
  println("")
  println("--- Cumulative operations (N=" + n_small.to_string() + ") ---")

  let cumsum_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_cumsum(vec_a))
    i = i + 1
  }
  let cumsum_ms = Float::from_uint64(@env.now() - cumsum_start)
  println("vec_cumsum: " + format_time_ms(cumsum_ms, cfg.iters) + " ms")

  let cumprod_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_cumprod(@numbt.vec_clip(vec_a, 0.1, 2.0)))
    i = i + 1
  }
  let cumprod_ms = Float::from_uint64(@env.now() - cumprod_start)
  println("vec_cumprod: " + format_time_ms(cumprod_ms, cfg.iters) + " ms")

  // --- Linear algebra extras ---
  println("")
  println("--- Linear algebra extras ---")

  let outer_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_outer(vec_100, vec_100))
    i = i + 1
  }
  let outer_ms = Float::from_uint64(@env.now() - outer_start)
  println("vec_outer(" + n_mat.to_string() + "): " + format_time_ms(outer_ms, cfg.iters) + " ms")

  let diag_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_diag(vec_100))
    i = i + 1
  }
  let diag_ms = Float::from_uint64(@env.now() - diag_start)
  println("vec_diag(" + n_mat.to_string() + "): " + format_time_ms(diag_ms, cfg.iters) + " ms")

  let trace_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.mat_trace(mat))
    i = i + 1
  }
  let trace_ms = Float::from_uint64(@env.now() - trace_start)
  println("mat_trace(" + n_mat.to_string() + "x" + n_mat.to_string() + "): " + format_time_ms(trace_ms, cfg.iters) + " ms")

  // --- Concatenate ---
  println("")
  println("--- Concatenate/Stack ---")

  let small_vecs : Array[@numbt.Vec] = [
    @numbt.vec_from_array(seeded_values(10, 256, 0)),
    @numbt.vec_from_array(seeded_values(11, 256, 0)),
    @numbt.vec_from_array(seeded_values(12, 256, 0)),
    @numbt.vec_from_array(seeded_values(13, 256, 0)),
  ]
  let concat_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_concatenate(small_vecs))
    i = i + 1
  }
  let concat_ms = Float::from_uint64(@env.now() - concat_start)
  println("vec_concatenate(4x256): " + format_time_ms(concat_ms, cfg.iters) + " ms")

  // --- Matrix ops ---
  println("")
  println("--- Matrix operations (" + n_mat.to_string() + "x" + n_mat.to_string() + ") ---")

  let mat_sum_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.mat_sum(mat))
    i = i + 1
  }
  let mat_sum_ms = Float::from_uint64(@env.now() - mat_sum_start)
  println("mat_sum: " + format_time_ms(mat_sum_ms, cfg.iters) + " ms")

  // --- New: Math functions ---
  println("")
  println("--- Math functions (N=" + n_small.to_string() + ") ---")

  let floor_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_floor(vec_a))
    i = i + 1
  }
  let floor_ms = Float::from_uint64(@env.now() - floor_start)
  println("vec_floor: " + format_time_ms(floor_ms, cfg.iters) + " ms")

  let ceil_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_ceil(vec_a))
    i = i + 1
  }
  let ceil_ms = Float::from_uint64(@env.now() - ceil_start)
  println("vec_ceil: " + format_time_ms(ceil_ms, cfg.iters) + " ms")

  let round_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_round(vec_a))
    i = i + 1
  }
  let round_ms = Float::from_uint64(@env.now() - round_start)
  println("vec_round: " + format_time_ms(round_ms, cfg.iters) + " ms")

  // --- New: Statistics ---
  println("")
  println("--- Statistics ---")

  let median_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_median(vec_a))
    i = i + 1
  }
  let median_ms = Float::from_uint64(@env.now() - median_start)
  println("vec_median: " + format_time_ms(median_ms, cfg.iters) + " ms")

  let percentile_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_percentile(vec_a, 50.0))
    i = i + 1
  }
  let percentile_ms = Float::from_uint64(@env.now() - percentile_start)
  println("vec_percentile(50): " + format_time_ms(percentile_ms, cfg.iters) + " ms")

  let unique_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_unique(vec_a))
    i = i + 1
  }
  let unique_ms = Float::from_uint64(@env.now() - unique_start)
  println("vec_unique: " + format_time_ms(unique_ms, cfg.iters) + " ms")

  // --- New: Search ---
  println("")
  println("--- Search ---")

  let sorted_vec = @numbt.vec_sort(vec_a)
  let searchsorted_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_searchsorted(sorted_vec, 0.5))
    i = i + 1
  }
  let searchsorted_ms = Float::from_uint64(@env.now() - searchsorted_start)
  println("vec_searchsorted: " + format_time_ms(searchsorted_ms, cfg.iters) + " ms")

  // --- New: Random ---
  println("")
  println("--- Random (N=" + n_small.to_string() + ") ---")

  let rand_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_rand(n_small))
    i = i + 1
  }
  let rand_ms = Float::from_uint64(@env.now() - rand_start)
  println("vec_rand: " + format_time_ms(rand_ms, cfg.iters) + " ms")

  let randn_start = @env.now()
  i = 0
  while i < cfg.iters {
    ignore(@numbt.vec_randn(n_small))
    i = i + 1
  }
  let randn_ms = Float::from_uint64(@env.now() - randn_start)
  println("vec_randn: " + format_time_ms(randn_ms, cfg.iters) + " ms")

  // --- New: Linear algebra inv/solve ---
  println("")
  println("--- Linear algebra: inv/solve (N=" + n_mat.to_string() + ") ---")

  // Create invertible matrix using FMat (A @ A^T + I)
  let fmat_a = @numbt.fmat_randn(n_mat, n_mat)
  let fmat_at = @numbt.fmat_transpose(fmat_a)
  let fmat_aat = @numbt.fmat_matmul(fmat_a, fmat_at)
  let fmat_eye = @numbt.fmat_eye(n_mat)
  let fmat_pd = @numbt.fmat_add(fmat_aat, fmat_eye)

  // FMat inv benchmark (fast path)
  let fmat_inv_start = @env.now()
  i = 0
  while i < cfg.iters / 10 {
    let tmp = @numbt.fmat_clone(fmat_pd)
    ignore(@numbt.fmat_inv(tmp))
    i = i + 1
  }
  let fmat_inv_ms = Float::from_uint64(@env.now() - fmat_inv_start)
  println("fmat_inv: " + format_time_ms(fmat_inv_ms, cfg.iters / 10) + " ms")

  // Mat inv benchmark (with conversion overhead) for comparison
  let mat_pd = @numbt.fmat_to_mat(fmat_pd)
  let mat_inv_start = @env.now()
  i = 0
  while i < cfg.iters / 10 {
    ignore(@numbt.mat_inv(@numbt.mat_clone(mat_pd)))
    i = i + 1
  }
  let mat_inv_ms = Float::from_uint64(@env.now() - mat_inv_start)
  println("mat_inv: " + format_time_ms(mat_inv_ms, cfg.iters / 10) + " ms (with conversion)")

  // --- New: LAPACK functions ---
  println("")
  println("--- LAPACK operations (N=" + n_mat.to_string() + ") ---")

  // SVD
  let svd_start = @env.now()
  i = 0
  while i < cfg.iters / 10 {
    let tmp = @numbt.fmat_clone(fmat_pd)
    ignore(@numbt.fmat_svd(tmp))
    i = i + 1
  }
  let svd_ms = Float::from_uint64(@env.now() - svd_start)
  println("fmat_svd: " + format_time_ms(svd_ms, cfg.iters / 10) + " ms")

  // Eigenvalues (symmetric)
  let eig_start = @env.now()
  i = 0
  while i < cfg.iters / 10 {
    let tmp = @numbt.fmat_clone(fmat_pd)
    ignore(@numbt.fmat_eig(tmp))
    i = i + 1
  }
  let eig_ms = Float::from_uint64(@env.now() - eig_start)
  println("fmat_eig: " + format_time_ms(eig_ms, cfg.iters / 10) + " ms")

  // Cholesky
  let chol_start = @env.now()
  i = 0
  while i < cfg.iters / 10 {
    let tmp = @numbt.fmat_clone(fmat_pd)
    ignore(@numbt.fmat_cholesky(tmp))
    i = i + 1
  }
  let chol_ms = Float::from_uint64(@env.now() - chol_start)
  println("fmat_cholesky: " + format_time_ms(chol_ms, cfg.iters / 10) + " ms")

  // QR
  let qr_start = @env.now()
  i = 0
  while i < cfg.iters / 10 {
    let tmp = @numbt.fmat_clone(fmat_pd)
    ignore(@numbt.fmat_qr(tmp))
    i = i + 1
  }
  let qr_ms = Float::from_uint64(@env.now() - qr_start)
  println("fmat_qr: " + format_time_ms(qr_ms, cfg.iters / 10) + " ms")

  // Determinant
  let det_start = @env.now()
  i = 0
  while i < cfg.iters / 10 {
    let tmp = @numbt.fmat_clone(fmat_pd)
    ignore(@numbt.fmat_det(tmp))
    i = i + 1
  }
  let det_ms = Float::from_uint64(@env.now() - det_start)
  println("fmat_det: " + format_time_ms(det_ms, cfg.iters / 10) + " ms")

  // Least squares
  let b_vec = @numbt.vec_from_array(seeded_values(100, n_mat, 0))
  let lstsq_start = @env.now()
  i = 0
  while i < cfg.iters / 10 {
    let tmp = @numbt.fmat_clone(fmat_pd)
    ignore(@numbt.fmat_lstsq(tmp, b_vec))
    i = i + 1
  }
  let lstsq_ms = Float::from_uint64(@env.now() - lstsq_start)
  println("fmat_lstsq: " + format_time_ms(lstsq_ms, cfg.iters / 10) + " ms")

  println("")
  println("============================================================")
  println("Benchmark complete")
  println("============================================================")
}

///|
fn main {
  let args = @env.args()
  let program = if args.length() > 0 { args[0] } else { "bench" }

  // Check for flags
  let mut use_blas = false
  let mut use_fused = false
  let mut use_numbt = false
  let mut i = 1
  while i < args.length() {
    if args[i] == "--blas" {
      use_blas = true
    }
    if args[i] == "--fused" {
      use_fused = true
    }
    if args[i] == "--numbt" {
      use_numbt = true
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

  if use_numbt {
    run_numbt_bench(cfg)
  } else if use_fused {
    run_blas_fused_bench(cfg)
  } else if use_blas {
    run_blas_bench(cfg)
  } else {
    run_bench(cfg)
  }
}
