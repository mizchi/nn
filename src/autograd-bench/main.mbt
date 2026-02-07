///|
fn rand_tensor(shape : @tensor.Shape, seed : Int) -> @tensor.Tensor {
  let n = shape.numel()
  let data = FixedArray::make(n, Float::from_int(0))
  let mut rng = seed
  let scale = Float::from_double((2.0 / n.to_double()).sqrt())
  for i = 0; i < n; i = i + 1 {
    rng = rng * 1103515245 + 12345
    let val = ((rng >> 16) & 0x7FFF).to_double() / 32768.0 - 0.5
    data[i] = Float::from_double(val * 2.0 * scale.to_double())
  }
  @tensor.tensor_new_fixed(shape, data)
}

///|
fn rand_labels(batch : Int, num_classes : Int, seed : Int) -> Array[Int] {
  let mut rng = seed
  Array::makei(batch, fn(_i) {
    rng = rng * 1103515245 + 12345
    ((rng >> 16) & 0x7FFF) % num_classes
  })
}

///|
fn ns_to_ms(ns : UInt64) -> String {
  let us = ns / 1000UL
  let ms_whole = us / 1000UL
  let ms_frac = us % 1000UL
  let frac_str = if ms_frac < 10UL {
    "00" + ms_frac.to_string()
  } else if ms_frac < 100UL {
    "0" + ms_frac.to_string()
  } else {
    ms_frac.to_string()
  }
  ms_whole.to_string() + "." + frac_str
}

///|
fn median_u64(arr : Array[UInt64]) -> UInt64 {
  let sorted = arr.copy()
  sorted.sort()
  sorted[sorted.length() / 2]
}

///|
fn avg_u64(arr : Array[UInt64]) -> UInt64 {
  let mut sum = 0UL
  for v in arr {
    sum = sum + v
  }
  sum / arr.length().to_uint64()
}

///|
/// Run one training step using Linear layers with workspace.
/// Returns (forward_ns, backward_ns, step_ns).
fn run_step(
  w1_data : @tensor.Tensor,
  b1_data : @tensor.Tensor,
  w2_data : @tensor.Tensor,
  b2_data : @tensor.Tensor,
  input_data : @tensor.Tensor,
  labels : Array[Int],
  lr : Float,
  ws1 : @autograd.LinearWorkspace,
  ws2 : @autograd.LinearWorkspace,
) -> (UInt64, UInt64, UInt64, Float) {
  let t0 = @tensor.clock_ns()
  // --- Forward ---
  let tape = @autograd.Tape::new()
  let w1 = tape.create_var(w1_data, true)
  let b1 = tape.create_var(b1_data, true)
  let linear1 = @autograd.Linear::from_vars(w1, b1)
  let w2 = tape.create_var(w2_data, true)
  let b2 = tape.create_var(b2_data, true)
  let linear2 = @autograd.Linear::from_vars(w2, b2)
  let x = tape.create_var(input_data, false)
  let h = linear1.forward_ws(x, ws1)
  let ha = @autograd.var_relu(h)
  let logits = linear2.forward_ws(ha, ws2)
  let loss = @autograd.var_cross_entropy(logits, labels)
  let t1 = @tensor.clock_ns()
  // --- Backward ---
  tape.backward(loss)
  let t2 = @tensor.clock_ns()
  // --- SGD step ---
  let params = linear1.parameters()
  params.push_iter(linear2.parameters().iter())
  @autograd.sgd_step(params, lr)
  let t3 = @tensor.clock_ns()
  let loss_val = loss.data().data[0]
  (t1 - t0, t2 - t1, t3 - t2, loss_val)
}

///|
fn bench_mlp(
  batch : Int,
  input_dim : Int,
  hidden_dim : Int,
  output_dim : Int,
  warmup : Int,
  iters : Int,
) -> Unit {
  let input_data = rand_tensor(
    @tensor.shape_new([batch, input_dim]).unwrap(),
    42,
  )
  let labels = rand_labels(batch, output_dim, 99)
  let lr = Float::from_double(0.01)
  let w1_data = rand_tensor(
    @tensor.shape_new([hidden_dim, input_dim]).unwrap(),
    100,
  )
  let b1_data = @tensor.tensor_zeros(@tensor.shape_new([hidden_dim]).unwrap())
  let w2_data = rand_tensor(
    @tensor.shape_new([output_dim, hidden_dim]).unwrap(),
    200,
  )
  let b2_data = @tensor.tensor_zeros(@tensor.shape_new([output_dim]).unwrap())
  // Pre-allocate workspaces (reused across iterations)
  let ws1 = @autograd.LinearWorkspace::new(batch, input_dim, hidden_dim)
  let ws2 = @autograd.LinearWorkspace::new(batch, hidden_dim, output_dim)
  // Warmup
  for _i = 0; _i < warmup; _i = _i + 1 {
    ignore(
      run_step(
        w1_data, b1_data, w2_data, b2_data, input_data, labels, lr, ws1, ws2,
      ),
    )
  }
  // Benchmark
  let fwd_times : Array[UInt64] = Array::make(iters, 0UL)
  let bwd_times : Array[UInt64] = Array::make(iters, 0UL)
  let step_times : Array[UInt64] = Array::make(iters, 0UL)
  let total_times : Array[UInt64] = Array::make(iters, 0UL)
  let mut last_loss = Float::from_int(0)
  for i = 0; i < iters; i = i + 1 {
    let (fwd, bwd, step, loss_val) = run_step(
      w1_data, b1_data, w2_data, b2_data, input_data, labels, lr, ws1, ws2,
    )
    fwd_times[i] = fwd
    bwd_times[i] = bwd
    step_times[i] = step
    total_times[i] = fwd + bwd + step
    last_loss = loss_val
  }
  let fwd_med = median_u64(fwd_times)
  let bwd_med = median_u64(bwd_times)
  let step_med = median_u64(step_times)
  let total_med = median_u64(total_times)
  let fwd_avg = avg_u64(fwd_times)
  let bwd_avg = avg_u64(bwd_times)
  let step_avg = avg_u64(step_times)
  let total_avg = avg_u64(total_times)
  println(
    "MLP [" +
    batch.to_string() +
    ", " +
    input_dim.to_string() +
    "] -> " +
    hidden_dim.to_string() +
    " -> " +
    output_dim.to_string(),
  )
  println(
    "  iters: " +
    iters.to_string() +
    " (warmup: " +
    warmup.to_string() +
    ")",
  )
  println("  loss:  " + last_loss.to_double().to_string())
  println(
    "  forward:  avg=" +
    ns_to_ms(fwd_avg) +
    "ms  p50=" +
    ns_to_ms(fwd_med) +
    "ms",
  )
  println(
    "  backward: avg=" +
    ns_to_ms(bwd_avg) +
    "ms  p50=" +
    ns_to_ms(bwd_med) +
    "ms",
  )
  println(
    "  sgd_step: avg=" +
    ns_to_ms(step_avg) +
    "ms  p50=" +
    ns_to_ms(step_med) +
    "ms",
  )
  println(
    "  total:    avg=" +
    ns_to_ms(total_avg) +
    "ms  p50=" +
    ns_to_ms(total_med) +
    "ms",
  )
  println("")
}

///|
fn main {
  println("=== MoonBit Autograd Benchmark ===")
  println("")
  // Small
  bench_mlp(32, 784, 128, 10, 10, 100)
  // Medium
  bench_mlp(64, 784, 256, 10, 10, 100)
  // Large
  bench_mlp(128, 784, 512, 10, 10, 50)
  // XL
  bench_mlp(256, 784, 1024, 10, 10, 20)
}
