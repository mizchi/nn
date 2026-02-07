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
/// Run one training step using fused Linear+ReLU with workspace.
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
  let ha = linear1.forward_relu_managed(x, ws1)
  let logits = linear2.forward_ws_managed(ha, ws2)
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
    "  iters: " + iters.to_string() + " (warmup: " + warmup.to_string() + ")",
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
fn profile_xl() -> Unit {
  let batch = 256
  let input_dim = 784
  let hidden_dim = 1024
  let output_dim = 10
  let one = Float::from_int(1)
  let zero = Float::from_int(0)
  let n = batch * hidden_dim
  // Allocate buffers
  let a = FixedArray::make(batch * input_dim, one)
  let b = FixedArray::make(hidden_dim * input_dim, one)
  let c = FixedArray::make(batch * hidden_dim, zero)
  let d = FixedArray::make(hidden_dim * input_dim, zero)
  let e = FixedArray::make(batch * input_dim, zero)
  let relu_in = FixedArray::make(n, one)
  let relu_out = FixedArray::make(n, zero)
  let warmup = 5
  let iters = 20
  println("=== XL Backward Profile ===")
  println("")
  // 1. sgemm forward: (256,1024,784) NoTrans,Trans
  for _i = 0; _i < warmup; _i = _i + 1 {
    @tensor.sgemm(
      0, 1, batch, hidden_dim, input_dim, one, a, input_dim, b, input_dim, zero,
      c, hidden_dim,
    )
  }
  let mut total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.sgemm(
      0, 1, batch, hidden_dim, input_dim, one, a, input_dim, b, input_dim, zero,
      c, hidden_dim,
    )
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  sgemm fwd (256,1024,784) NoT,T: " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // 2. sgemm backward dW: (1024,784,256) Trans,NoTrans
  for _i = 0; _i < warmup; _i = _i + 1 {
    @tensor.sgemm(
      1, 0, hidden_dim, input_dim, batch, one, c, hidden_dim, a, input_dim, zero,
      d, input_dim,
    )
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.sgemm(
      1, 0, hidden_dim, input_dim, batch, one, c, hidden_dim, a, input_dim, zero,
      d, input_dim,
    )
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  sgemm bwd dW (1024,784,256) T,N: " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // 3. sgemm backward dx linear2: (256,1024,10) NoTrans,NoTrans
  let small_w = FixedArray::make(output_dim * hidden_dim, one)
  let small_dy = FixedArray::make(batch * output_dim, one)
  let dx_l2 = FixedArray::make(batch * hidden_dim, zero)
  for _i = 0; _i < warmup; _i = _i + 1 {
    @tensor.sgemm(
      0, 0, batch, hidden_dim, output_dim, one, small_dy, output_dim, small_w, hidden_dim,
      zero, dx_l2, hidden_dim,
    )
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.sgemm(
      0, 0, batch, hidden_dim, output_dim, one, small_dy, output_dim, small_w, hidden_dim,
      zero, dx_l2, hidden_dim,
    )
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  sgemm l2 dx (256,1024,10) N,N:   " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // 4a. relu_backward (same buffer for dy and x - warm cache)
  for _i = 0; _i < warmup; _i = _i + 1 {
    @tensor.relu_backward_inplace(relu_in, relu_in, relu_out, n)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.relu_backward_inplace(relu_in, relu_in, relu_out, n)
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  relu_bwd (same buf):              " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // 4b. relu_backward (separate buffers - more realistic)
  let relu_dy = FixedArray::make(n, one)
  let relu_x2 = FixedArray::make(n, one)
  let relu_dx = FixedArray::make(n, zero)
  for _i = 0; _i < warmup; _i = _i + 1 {
    @tensor.relu_backward_inplace(relu_dy, relu_x2, relu_dx, n)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.relu_backward_inplace(relu_dy, relu_x2, relu_dx, n)
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  relu_bwd (sep bufs):              " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // 4c. relu backward after sgemm (cold cache simulation)
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    // sgemm writes to c, evicting relu buffers from cache
    @tensor.sgemm(
      0, 0, batch, hidden_dim, output_dim, one, small_dy, output_dim, small_w, hidden_dim,
      zero, dx_l2, hidden_dim,
    )
    let t0 = @tensor.clock_ns()
    @tensor.relu_backward_inplace(relu_dy, relu_x2, relu_dx, n)
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  relu_bwd (after sgemm):           " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // 5. FixedArray::make allocation
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    let _x = FixedArray::make(n, zero)
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  FixedArray::make(262144):          " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // 6. accumulate_inplace
  let acc_a = FixedArray::make(hidden_dim * input_dim, one)
  let acc_b = FixedArray::make(hidden_dim * input_dim, one)
  for _i = 0; _i < warmup; _i = _i + 1 {
    @tensor.accumulate_inplace(acc_a, acc_b, hidden_dim * input_dim)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.accumulate_inplace(acc_a, acc_b, hidden_dim * input_dim)
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  accumulate (802816):               " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // 7. bias_add_backward
  let bias_dy = FixedArray::make(batch * hidden_dim, one)
  let bias_db = FixedArray::make(hidden_dim, zero)
  for _i = 0; _i < warmup; _i = _i + 1 {
    @tensor.bias_add_backward(bias_dy, bias_db, batch * hidden_dim, hidden_dim)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.bias_add_backward(bias_dy, bias_db, batch * hidden_dim, hidden_dim)
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  bias_add_backward (256x1024):      " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Profile actual backward
  println("")
  println("  --- Actual backward per-node ---")
  let input_data = rand_tensor(
    @tensor.shape_new([batch, input_dim]).unwrap(),
    42,
  )
  let labels = rand_labels(batch, output_dim, 99)
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
  let ws1 = @autograd.LinearWorkspace::new(batch, input_dim, hidden_dim)
  let ws2 = @autograd.LinearWorkspace::new(batch, hidden_dim, output_dim)

  // Warmup
  for _i = 0; _i < 3; _i = _i + 1 {
    let tape = @autograd.Tape::new()
    let w1 = tape.create_var(w1_data, true)
    let b1_var = tape.create_var(b1_data, true)
    let linear1 = @autograd.Linear::from_vars(w1, b1_var)
    let w2 = tape.create_var(w2_data, true)
    let b2_var = tape.create_var(b2_data, true)
    let linear2 = @autograd.Linear::from_vars(w2, b2_var)
    let x_var = tape.create_var(input_data, false)
    let ha_var = linear1.forward_relu_managed(x_var, ws1)
    let logits_var = linear2.forward_ws_managed(ha_var, ws2)
    let loss_var = @autograd.var_cross_entropy(logits_var, labels)
    tape.backward(loss_var)
  }
  // Profile run (5 iterations to see variance)
  let node_names = [
    "w1", "b1", "w2", "b2", "x", "fused_l1_relu", "linear2_out", "cross_entropy",
  ]
  for run = 0; run < 5; run = run + 1 {
    let tape = @autograd.Tape::new()
    let w1p = tape.create_var(w1_data, true)
    let b1p = tape.create_var(b1_data, true)
    let linear1p = @autograd.Linear::from_vars(w1p, b1p)
    let w2p = tape.create_var(w2_data, true)
    let b2p = tape.create_var(b2_data, true)
    let linear2p = @autograd.Linear::from_vars(w2p, b2p)
    let xp = tape.create_var(input_data, false)
    let hap = linear1p.forward_relu_managed(xp, ws1)
    let logitsp = linear2p.forward_ws_managed(hap, ws2)
    let lossp = @autograd.var_cross_entropy(logitsp, labels)
    let timings = tape.backward_profiled(lossp)
    let mut line = "  run" + run.to_string() + ": "
    for t in timings {
      let (idx, ns) = t
      let name = if idx < node_names.length() {
        node_names[idx]
      } else {
        "node_" + idx.to_string()
      }
      line = line + name + "=" + ns_to_ms(ns) + "ms "
    }
    println(line)
  }
  println("")
  // Manual backward (no tape/closures)
  println("  --- Manual backward (no tape) ---")
  let w1_manual = FixedArray::make(hidden_dim * input_dim, one)
  let w2_manual = FixedArray::make(output_dim * hidden_dim, one)
  let input_manual = FixedArray::make(batch * input_dim, one)
  let h_manual = FixedArray::make(batch * hidden_dim, one) // linear1 output
  let ha_manual = FixedArray::make(batch * hidden_dim, one) // relu output
  let logits_manual = FixedArray::make(batch * output_dim, one)
  // Pre-allocate all backward buffers
  let d_logits = FixedArray::make(batch * output_dim, zero)
  let d_ha = FixedArray::make(batch * hidden_dim, zero) // dx from linear2
  let d_h = FixedArray::make(batch * hidden_dim, zero) // dx from relu
  let d_w1 = FixedArray::make(hidden_dim * input_dim, zero) // dW from linear1
  let d_w2 = FixedArray::make(output_dim * hidden_dim, zero) // dW from linear2
  let d_b1 = FixedArray::make(hidden_dim, zero)
  let d_b2 = FixedArray::make(output_dim, zero)
  // Warmup manual backward
  for _i = 0; _i < 5; _i = _i + 1 {
    // cross_entropy backward (just compute softmax - one_hot, tiny)
    @tensor.softmax_inplace(d_logits, batch, output_dim)
    // linear2 backward dx: d_ha = d_logits @ w2
    @tensor.sgemm(
      0, 0, batch, hidden_dim, output_dim, one, d_logits, output_dim, w2_manual,
      hidden_dim, zero, d_ha, hidden_dim,
    )
    // relu backward
    @tensor.relu_backward_inplace(d_ha, h_manual, d_h, batch * hidden_dim)
    // linear1 backward dW: d_w1 = d_h^T @ input
    @tensor.sgemm(
      1, 0, hidden_dim, input_dim, batch, one, d_h, hidden_dim, input_manual, input_dim,
      zero, d_w1, input_dim,
    )
    // linear1 backward dbias
    @tensor.bias_add_backward(d_h, d_b1, batch * hidden_dim, hidden_dim)
  }
  // Measure manual backward
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    // cross_entropy backward
    @tensor.softmax_inplace(d_logits, batch, output_dim)
    // linear2 backward dx
    @tensor.sgemm(
      0, 0, batch, hidden_dim, output_dim, one, d_logits, output_dim, w2_manual,
      hidden_dim, zero, d_ha, hidden_dim,
    )
    // linear2 backward dW
    @tensor.sgemm(
      1, 0, output_dim, hidden_dim, batch, one, d_logits, output_dim, ha_manual,
      hidden_dim, zero, d_w2, hidden_dim,
    )
    // relu backward
    @tensor.relu_backward_inplace(d_ha, h_manual, d_h, batch * hidden_dim)
    // linear1 backward dW
    @tensor.sgemm(
      1, 0, hidden_dim, input_dim, batch, one, d_h, hidden_dim, input_manual, input_dim,
      zero, d_w1, input_dim,
    )
    // linear1 backward dbias
    @tensor.bias_add_backward(d_h, d_b1, batch * hidden_dim, hidden_dim)
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  manual backward total:            " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Individual timing
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    @tensor.sgemm(
      0, 0, batch, hidden_dim, output_dim, one, d_logits, output_dim, w2_manual,
      hidden_dim, zero, d_ha, hidden_dim,
    )
    let t0 = @tensor.clock_ns()
    @tensor.relu_backward_inplace(d_ha, h_manual, d_h, batch * hidden_dim)
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  relu after l2_dx sgemm:           " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  println("")
  // 7b. Tape backward with inline timing: measure INSIDE the closure
  println("  --- Inline closure timing ---")
  // Build the graph once and time the backward with internal profiling
  let tape_diag = @autograd.Tape::new()
  let w1d = tape_diag.create_var(w1_data, true)
  let b1d = tape_diag.create_var(b1_data, true)
  let l1d = @autograd.Linear::from_vars(w1d, b1d)
  let w2d = tape_diag.create_var(w2_data, true)
  let b2d = tape_diag.create_var(b2_data, true)
  let l2d = @autograd.Linear::from_vars(w2d, b2d)
  let xd = tape_diag.create_var(input_data, false)
  // Use separate non-fused path to isolate relu node
  let hd = l1d.forward_ws(xd, ws1)
  let had = @autograd.var_relu(hd)
  let logd = l2d.forward_ws(had, ws2)
  let lossd = @autograd.var_cross_entropy(logd, labels)
  // Warmup
  for _i = 0; _i < 3; _i = _i + 1 {
    let t2 = @autograd.Tape::new()
    let w1t = t2.create_var(w1_data, true)
    let b1t = t2.create_var(b1_data, true)
    let l1t = @autograd.Linear::from_vars(w1t, b1t)
    let w2t = t2.create_var(w2_data, true)
    let b2t = t2.create_var(b2_data, true)
    let l2t = @autograd.Linear::from_vars(w2t, b2t)
    let xt = t2.create_var(input_data, false)
    let ht = l1t.forward_ws(xt, ws1)
    let hat = @autograd.var_relu(ht)
    let logt = l2t.forward_ws(hat, ws2)
    let losst = @autograd.var_cross_entropy(logt, labels)
    t2.backward(losst)
  }
  // Now run backward but call closures manually with timing between them
  // Set up grad for output
  let ones_diag = @tensor.tensor_ones(lossd.data().shape)
  tape_diag.set_grad(lossd.id, ones_diag)
  // Cross entropy backward (node 8)
  let t_ce0 = @tensor.clock_ns()
  tape_diag.run_backward_node(8)
  let t_ce1 = @tensor.clock_ns()
  // Linear2 backward (node 7)
  tape_diag.run_backward_node(7)
  let t_l2 = @tensor.clock_ns()
  // relu backward (node 6)
  tape_diag.run_backward_node(6)
  let t_relu = @tensor.clock_ns()
  // Linear1 backward (node 5)
  tape_diag.run_backward_node(5)
  let t_l1 = @tensor.clock_ns()
  println("  cross_entropy: " + ns_to_ms(t_ce1 - t_ce0) + "ms")
  println("  linear2:       " + ns_to_ms(t_l2 - t_ce1) + "ms")
  println("  relu:          " + ns_to_ms(t_relu - t_l2) + "ms")
  println("  linear1:       " + ns_to_ms(t_l1 - t_relu) + "ms")
  println("")
  // 7c. Same operations outside closure, using same buffers
  println("  --- Same ops, no closure ---")
  // Simulate: after linear2 backward wrote to ws2.dx_data,
  // call relu_backward with same buffers the closure would use
  let diag_dy = ws2.dx_data // same buffer linear2 backward writes dx to
  let diag_x = ws1.y_data // same buffer linear1 forward output goes to
  let diag_out = FixedArray::make(batch * hidden_dim, zero)
  let diag_n = batch * hidden_dim
  // Warmup
  for _i = 0; _i < 5; _i = _i + 1 {
    @tensor.relu_backward_inplace(diag_dy, diag_x, diag_out, diag_n)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    // Run linear2's dx sgemm first (to match cache state)
    @tensor.sgemm(
      0, 0, batch, hidden_dim, output_dim, one, small_dy, output_dim, small_w, hidden_dim,
      zero, diag_dy, hidden_dim,
    )
    let t0_d = @tensor.clock_ns()
    @tensor.relu_backward_inplace(diag_dy, diag_x, diag_out, diag_n)
    total = total + @tensor.clock_ns() - t0_d
  }
  println(
    "  relu_bwd (same ws bufs, after sgemm): " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Test A: sgemm writes to a DIFFERENT buffer, relu reads ws bufs
  let sgemm_out_alt = FixedArray::make(batch * hidden_dim, zero)
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    @tensor.sgemm(
      0, 0, batch, hidden_dim, output_dim, one, small_dy, output_dim, small_w, hidden_dim,
      zero, sgemm_out_alt, hidden_dim,
    )
    let t0_d = @tensor.clock_ns()
    @tensor.relu_backward_inplace(diag_dy, diag_x, diag_out, diag_n)
    total = total + @tensor.clock_ns() - t0_d
  }
  println(
    "  relu_bwd (sgemm to diff buf):         " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Test B: no sgemm at all, just relu with ws buffers
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0_d = @tensor.clock_ns()
    @tensor.relu_backward_inplace(diag_dy, diag_x, diag_out, diag_n)
    total = total + @tensor.clock_ns() - t0_d
  }
  println(
    "  relu_bwd (ws bufs, no sgemm):         " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Test C: big sgemm (l1 dW) then relu with ws bufs
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    @tensor.sgemm(
      1, 0, hidden_dim, input_dim, batch, one, c, hidden_dim, a, input_dim, zero,
      d, input_dim,
    )
    let t0_d = @tensor.clock_ns()
    @tensor.relu_backward_inplace(diag_dy, diag_x, diag_out, diag_n)
    total = total + @tensor.clock_ns() - t0_d
  }
  println(
    "  relu_bwd (ws bufs, after big sgemm):  " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Test D: fresh alloc buffers of same size - do they also have the problem?
  let fresh_dy = FixedArray::make(diag_n, one)
  let fresh_x = FixedArray::make(diag_n, one)
  let fresh_out = FixedArray::make(diag_n, zero)
  for _i = 0; _i < 5; _i = _i + 1 {
    @tensor.relu_backward_inplace(fresh_dy, fresh_x, fresh_out, diag_n)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0_d = @tensor.clock_ns()
    @tensor.relu_backward_inplace(fresh_dy, fresh_x, fresh_out, diag_n)
    total = total + @tensor.clock_ns() - t0_d
  }
  println(
    "  relu_bwd (fresh alloc, no sgemm):     " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Test E: ws bufs but with padding to break aliasing
  let pad = FixedArray::make(64, zero) // 256-byte pad to shift alignment
  ignore(pad)
  let padded_dy = FixedArray::make(diag_n, one)
  let _pad2 = FixedArray::make(64, zero)
  let padded_x = FixedArray::make(diag_n, one)
  let _pad3 = FixedArray::make(64, zero)
  let padded_out = FixedArray::make(diag_n, zero)
  for _i = 0; _i < 5; _i = _i + 1 {
    @tensor.relu_backward_inplace(padded_dy, padded_x, padded_out, diag_n)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0_d = @tensor.clock_ns()
    @tensor.relu_backward_inplace(padded_dy, padded_x, padded_out, diag_n)
    total = total + @tensor.clock_ns() - t0_d
  }
  println(
    "  relu_bwd (padded alloc, no sgemm):    " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Test F: what makes ws bufs special? Try many fresh buffers allocated together
  let batch_bufs : Array[FixedArray[Float]] = []
  for _i = 0; _i < 10; _i = _i + 1 {
    batch_bufs.push(FixedArray::make(diag_n, one))
  }
  // Use 3 consecutive buffers from the batch
  for _i = 0; _i < 5; _i = _i + 1 {
    @tensor.relu_backward_inplace(
      batch_bufs[0],
      batch_bufs[1],
      batch_bufs[2],
      diag_n,
    )
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0_d = @tensor.clock_ns()
    @tensor.relu_backward_inplace(
      batch_bufs[0],
      batch_bufs[1],
      batch_bufs[2],
      diag_n,
    )
    total = total + @tensor.clock_ns() - t0_d
  }
  println(
    "  relu_bwd (batch[0,1,2] consecutive):  " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Use non-consecutive buffers
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0_d = @tensor.clock_ns()
    @tensor.relu_backward_inplace(
      batch_bufs[0],
      batch_bufs[3],
      batch_bufs[7],
      diag_n,
    )
    total = total + @tensor.clock_ns() - t0_d
  }
  println(
    "  relu_bwd (batch[0,3,7] spaced):       " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Test G: Re-allocate workspace-like buffers right here
  let ws_fresh1 = @autograd.LinearWorkspace::new(batch, input_dim, hidden_dim)
  let ws_fresh2 = @autograd.LinearWorkspace::new(batch, hidden_dim, output_dim)
  let relu_pre_fresh = FixedArray::make(
    batch * hidden_dim + 4096,
    Float::from_int(0),
  )
  let relu_bwd_fresh = FixedArray::make(
    batch * hidden_dim + 4096,
    Float::from_int(0),
  )
  // Use ws_fresh2.dx_data, ws_fresh1.y_data, relu_bwd_fresh for relu_backward
  for _i = 0; _i < 5; _i = _i + 1 {
    @tensor.relu_backward_inplace(
      ws_fresh2.dx_data,
      ws_fresh1.y_data,
      relu_bwd_fresh,
      diag_n,
    )
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0_d = @tensor.clock_ns()
    @tensor.relu_backward_inplace(
      ws_fresh2.dx_data,
      ws_fresh1.y_data,
      relu_bwd_fresh,
      diag_n,
    )
    total = total + @tensor.clock_ns() - t0_d
  }
  println(
    "  relu_bwd (re-alloc ws bufs):          " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  ignore(relu_pre_fresh)
  println("")
  // 7h. Test C-managed relu_backward_managed
  println("  --- C-managed relu_backward tests ---")
  // Populate g_relu_pre with sgemm_to_relu_pre
  @tensor.sgemm_to_relu_pre(
    0, 1, batch, hidden_dim, input_dim, one, a, input_dim, b, input_dim, n,
  )
  // Populate g_grad_buf with sgemm_to_grad_buf
  @tensor.sgemm_to_grad_buf(
    0, 0, batch, hidden_dim, output_dim, one, small_dy, output_dim, small_w, hidden_dim,
    n,
  )
  let managed_dx = FixedArray::make(n, zero)
  for _i = 0; _i < 5; _i = _i + 1 {
    @tensor.relu_backward_managed(managed_dx, n)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.relu_backward_managed(managed_dx, n)
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  relu_bwd_managed (C-managed dy+x):    " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Test after sgemm evicts cache
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    @tensor.sgemm(
      0, 0, batch, hidden_dim, output_dim, one, small_dy, output_dim, small_w, hidden_dim,
      zero, dx_l2, hidden_dim,
    )
    let t0 = @tensor.clock_ns()
    @tensor.relu_backward_managed(managed_dx, n)
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  relu_bwd_managed (after sgemm):       " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // 7i. sgemm dW with old-gen input (xd.data simulation)
  // ws1.dx_data is old-gen, use it as "old-gen input" to sgemm dW
  println("")
  println("  --- sgemm dW with old-gen vs fresh ---")
  let fresh_rdx = FixedArray::make(n, one)
  let fresh_input = FixedArray::make(batch * input_dim, one)
  let fresh_dw = FixedArray::make(hidden_dim * input_dim, zero)
  // Fresh inputs
  for _i = 0; _i < 5; _i = _i + 1 {
    @tensor.sgemm(
      1, 0, hidden_dim, input_dim, batch, one, fresh_rdx, hidden_dim, fresh_input,
      input_dim, zero, fresh_dw, input_dim,
    )
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.sgemm(
      1, 0, hidden_dim, input_dim, batch, one, fresh_rdx, hidden_dim, fresh_input,
      input_dim, zero, fresh_dw, input_dim,
    )
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  sgemm dW (all fresh):                 " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Old-gen C = ws1.dw_data (workspace)
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.sgemm(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      fresh_rdx,
      hidden_dim,
      fresh_input,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  sgemm dW (C=ws1.dw_data):             " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // sgemm dW with B=input_data (the actual captured xd.data from closure)
  let input_data_arr = input_data.contiguous().data
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.sgemm(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      fresh_rdx,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  sgemm dW (B=input_data,C=ws1.dw):     " +
    ns_to_ms(total / iters.to_uint64()) +
    "ms",
  )
  // Full fused backward simulation outside closure, using actual ws buffers
  println("")
  println("  --- Simulated fused backward ---")
  // Simulate what forward_relu_managed backward does, using actual ws1 buffers
  let sim_rdx = FixedArray::make(n, zero)
  // Populate C-managed buffers
  @tensor.sgemm_to_relu_pre(
    0, 1, batch, hidden_dim, input_dim, one, a, input_dim, b, input_dim, n,
  )
  @tensor.sgemm_to_grad_buf(
    0, 0, batch, hidden_dim, output_dim, one, small_dy, output_dim, small_w, hidden_dim,
    n,
  )
  // Warmup
  for _i = 0; _i < 3; _i = _i + 1 {
    @tensor.relu_backward_managed(sim_rdx, n)
    @tensor.sgemm(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      sim_rdx,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    @tensor.bias_add_backward(sim_rdx, ws1.db_data, n, hidden_dim)
  }
  // Timed run
  let mut t_relu_m = 0UL
  let mut t_sgemm_dw = 0UL
  let mut t_bias = 0UL
  let mut t_alloc = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let ta0 = @tensor.clock_ns()
    let rdx = FixedArray::make(n, zero)
    let ta1 = @tensor.clock_ns()
    @tensor.relu_backward_managed(rdx, n)
    let ta2 = @tensor.clock_ns()
    @tensor.sgemm(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      rdx,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    let ta3 = @tensor.clock_ns()
    @tensor.bias_add_backward(rdx, ws1.db_data, n, hidden_dim)
    let ta4 = @tensor.clock_ns()
    t_alloc = t_alloc + ta1 - ta0
    t_relu_m = t_relu_m + ta2 - ta1
    t_sgemm_dw = t_sgemm_dw + ta3 - ta2
    t_bias = t_bias + ta4 - ta3
  }
  let iters_u = iters.to_uint64()
  println("  alloc:         " + ns_to_ms(t_alloc / iters_u) + "ms")
  println("  relu_managed:  " + ns_to_ms(t_relu_m / iters_u) + "ms")
  println("  sgemm dW:      " + ns_to_ms(t_sgemm_dw / iters_u) + "ms")
  println("  bias_bwd:      " + ns_to_ms(t_bias / iters_u) + "ms")
  let sim_total = t_alloc + t_relu_m + t_sgemm_dw + t_bias
  println("  total:         " + ns_to_ms(sim_total / iters_u) + "ms")
  // Same ops wrapped in a closure to test closure overhead
  println("")
  println("  --- Closure overhead test ---")
  let closure_fn : () -> Unit = fn() {
    let rdx = FixedArray::make(n, zero)
    @tensor.relu_backward_managed(rdx, n)
    @tensor.sgemm(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      rdx,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    @tensor.bias_add_backward(rdx, ws1.db_data, n, hidden_dim)
  }
  for _i = 0; _i < 3; _i = _i + 1 {
    closure_fn()
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    closure_fn()
    total = total + @tensor.clock_ns() - t0
  }
  println("  closure call:  " + ns_to_ms(total / iters_u) + "ms")
  // Same but with FixedArray::make OUTSIDE the closure
  let pre_rdx = FixedArray::make(n, zero)
  let closure_fn2 : () -> Unit = fn() {
    @tensor.relu_backward_managed(pre_rdx, n)
    @tensor.sgemm(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      pre_rdx,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    @tensor.bias_add_backward(pre_rdx, ws1.db_data, n, hidden_dim)
  }
  for _i = 0; _i < 3; _i = _i + 1 {
    closure_fn2()
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    closure_fn2()
    total = total + @tensor.clock_ns() - t0
  }
  println("  closure no-alloc: " + ns_to_ms(total / iters_u) + "ms")
  // Test: tape backward vs direct closure call
  println("")
  println("  --- Tape backward isolation ---")
  for run = 0; run < 3; run = run + 1 {
    // Build a minimal graph: just input -> fused_l1_relu -> output
    let tape3 = @autograd.Tape::new()
    let w1v = tape3.create_var(w1_data, true) // node 0
    let b1v = tape3.create_var(b1_data, true) // node 1
    let l1v = @autograd.Linear::from_vars(w1v, b1v)
    let xv = tape3.create_var(input_data, false) // node 2
    // Set up C-managed buffers by running fused forward
    let hv = l1v.forward_relu_managed(xv, ws1) // node 3
    // Manually set gradient for node 3 (skip linear2/cross_entropy entirely)
    let ones_grad = @tensor.tensor_ones(hv.data().shape)
    tape3.set_grad(hv.id, ones_grad)
    let t0 = @tensor.clock_ns()
    tape3.run_backward_node(hv.id)
    let t1 = @tensor.clock_ns()
    println(
      "  run" +
      run.to_string() +
      " fused_only backward: " +
      ns_to_ms(t1 - t0) +
      "ms",
    )
  }
  // Test: closure with Tensor argument (matching bwd signature)
  println("")
  println("  --- Closure with Tensor arg ---")
  let dummy_grad = @tensor.tensor_ones(
    @tensor.shape_new([batch, hidden_dim]).unwrap(),
  )
  let bwd_like : (@tensor.Tensor) -> Unit = fn(dy) {
    ignore(dy)
    @tensor.relu_backward_fully_managed(n)
    @tensor.sgemm_relu_dx_a(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    for i = 0; i < ws1.db_data.length(); i = i + 1 {
      ws1.db_data[i] = zero
    }
    @tensor.bias_add_bwd_relu_dx(ws1.db_data, n, hidden_dim)
  }
  for _i = 0; _i < 3; _i = _i + 1 {
    bwd_like(dummy_grad)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    bwd_like(dummy_grad)
    total = total + @tensor.clock_ns() - t0
  }
  println("  bwd_like(tensor):  " + ns_to_ms(total / iters_u) + "ms")
  // Test: same but stored in an Array (simulating tape.nodes[].backward_fn)
  let stored_fns : Array[(@tensor.Tensor) -> Unit] = [bwd_like]
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    stored_fns[0](dummy_grad)
    total = total + @tensor.clock_ns() - t0
  }
  println("  stored_fn[0](tensor): " + ns_to_ms(total / iters_u) + "ms")
  // Detailed timing of fully managed ops
  println("")
  println("  --- Fully managed op timing ---")
  // Populate C-managed buffers
  @tensor.sgemm_to_relu_pre(
    0, 1, batch, hidden_dim, input_dim, one, a, input_dim, b, input_dim, n,
  )
  @tensor.sgemm_to_grad_buf(
    0, 0, batch, hidden_dim, output_dim, one, small_dy, output_dim, small_w, hidden_dim,
    n,
  )
  for _i = 0; _i < 5; _i = _i + 1 {
    @tensor.relu_backward_fully_managed(n)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.relu_backward_fully_managed(n)
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  relu_bwd_fully_managed:               " +
    ns_to_ms(total / iters_u) +
    "ms",
  )
  // sgemm_relu_dx_a after relu_bwd_fully_managed
  for _i = 0; _i < 5; _i = _i + 1 {
    @tensor.relu_backward_fully_managed(n)
    @tensor.sgemm_relu_dx_a(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    @tensor.relu_backward_fully_managed(n)
    let t0 = @tensor.clock_ns()
    @tensor.sgemm_relu_dx_a(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  sgemm_relu_dx_a (after relu_fully):   " +
    ns_to_ms(total / iters_u) +
    "ms",
  )
  // Compare: sgemm with MoonBit buffer (not g_relu_dx) after relu_backward_managed
  let test_rdx = FixedArray::make(n, one)
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    @tensor.relu_backward_managed(test_rdx, n)
    let t0 = @tensor.clock_ns()
    @tensor.sgemm(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      test_rdx,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  sgemm (MoonBit rdx after relu_mgd):   " +
    ns_to_ms(total / iters_u) +
    "ms",
  )
  // Both relu+sgemm timed together
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.relu_backward_fully_managed(n)
    @tensor.sgemm_relu_dx_a(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  relu_fully+sgemm_rdx_a together:      " +
    ns_to_ms(total / iters_u) +
    "ms",
  )
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    @tensor.relu_backward_managed(test_rdx, n)
    @tensor.sgemm(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      test_rdx,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  relu_mgd+sgemm (MoonBit rdx) together:" +
    ns_to_ms(total / iters_u) +
    "ms",
  )
  // A/B test: same ops as closure_fn, but with Tensor arg
  println("")
  println("  --- A/B closure arg test ---")
  // Version A: () -> Unit, same as closure_fn
  let test_a : () -> Unit = fn() {
    let rdx = FixedArray::make(n, zero)
    @tensor.relu_backward_managed(rdx, n)
    @tensor.sgemm(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      rdx,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    @tensor.bias_add_backward(rdx, ws1.db_data, n, hidden_dim)
  }
  for _i = 0; _i < 5; _i = _i + 1 {
    test_a()
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    test_a()
    total = total + @tensor.clock_ns() - t0
  }
  println("  () -> Unit:            " + ns_to_ms(total / iters_u) + "ms")
  // Version B: (Tensor) -> Unit, ignore arg, SAME ops
  let test_b : (@tensor.Tensor) -> Unit = fn(dy) {
    ignore(dy)
    let rdx = FixedArray::make(n, zero)
    @tensor.relu_backward_managed(rdx, n)
    @tensor.sgemm(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      rdx,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    @tensor.bias_add_backward(rdx, ws1.db_data, n, hidden_dim)
  }
  for _i = 0; _i < 5; _i = _i + 1 {
    test_b(dummy_grad)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    test_b(dummy_grad)
    total = total + @tensor.clock_ns() - t0
  }
  println("  (Tensor) -> Unit:      " + ns_to_ms(total / iters_u) + "ms")
  // Version C: (Tensor) -> Unit, ignore arg, fully managed (same as bwd_like)
  let test_c : (@tensor.Tensor) -> Unit = fn(dy) {
    ignore(dy)
    @tensor.relu_backward_fully_managed(n)
    @tensor.sgemm_relu_dx_a(
      1,
      0,
      hidden_dim,
      input_dim,
      batch,
      one,
      hidden_dim,
      input_data_arr,
      input_dim,
      zero,
      ws1.dw_data,
      input_dim,
    )
    for i = 0; i < ws1.db_data.length(); i = i + 1 {
      ws1.db_data[i] = zero
    }
    @tensor.bias_add_bwd_relu_dx(ws1.db_data, n, hidden_dim)
  }
  for _i = 0; _i < 5; _i = _i + 1 {
    test_c(dummy_grad)
  }
  total = 0UL
  for _i = 0; _i < iters; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    test_c(dummy_grad)
    total = total + @tensor.clock_ns() - t0
  }
  println("  (Tensor) fully managed: " + ns_to_ms(total / iters_u) + "ms")
  println("")
  // 8. contiguous check overhead
  let shape = @tensor.shape_new([batch, hidden_dim]).unwrap()
  let t_data = FixedArray::make(n, one)
  let tensor = @tensor.tensor_new_fixed(shape, t_data)
  total = 0UL
  for _i = 0; _i < iters * 10; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    ignore(tensor.contiguous())
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  contiguous() check (x200):         " +
    ns_to_ms(total / (iters * 10).to_uint64()) +
    "ms",
  )
  // 9. shape.strides()
  total = 0UL
  for _i = 0; _i < iters * 10; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    ignore(shape.strides())
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  shape.strides() (x200):            " +
    ns_to_ms(total / (iters * 10).to_uint64()) +
    "ms",
  )
  // 10. tensor_new_fixed
  total = 0UL
  for _i = 0; _i < iters * 10; _i = _i + 1 {
    let t0 = @tensor.clock_ns()
    ignore(@tensor.tensor_new_fixed(shape, t_data))
    total = total + @tensor.clock_ns() - t0
  }
  println(
    "  tensor_new_fixed (x200):           " +
    ns_to_ms(total / (iters * 10).to_uint64()) +
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
  // Profile
  profile_xl()
}
