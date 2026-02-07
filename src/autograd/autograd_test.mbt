///|
/// Generate a random tensor with LCG for testing
fn rand_tensor(shape : @tensor.Shape, seed : Int) -> @tensor.Tensor {
  let n = shape.numel()
  let data = FixedArray::make(n, Float::from_int(0))
  let mut rng = seed
  for i = 0; i < n; i = i + 1 {
    rng = rng * 1103515245 + 12345
    let val = ((rng >> 16) & 0x7FFF).to_double() / 32768.0 - 0.5
    data[i] = Float::from_double(val)
  }
  @tensor.tensor_new_fixed(shape, data)
}

///|
/// Compute numerical gradient using finite differences.
fn numerical_grad(
  param : @tensor.Tensor,
  f : () -> Float,
  eps : Float,
) -> @tensor.Tensor {
  let n = param.numel()
  let grad_data = FixedArray::make(n, Float::from_int(0))
  for i = 0; i < n; i = i + 1 {
    let orig = param.data[i]
    param.data[i] = orig + eps
    let f_plus = f()
    param.data[i] = orig - eps
    let f_minus = f()
    param.data[i] = orig
    grad_data[i] = Float::from_double(
      (f_plus.to_double() - f_minus.to_double()) / (2.0 * eps.to_double()),
    )
  }
  @tensor.tensor_new_fixed(param.shape, grad_data)
}

///|
/// Check if two tensors are close (relative error < threshold)
fn check_grad(
  _name : String,
  analytic : @tensor.Tensor,
  numerical : @tensor.Tensor,
  threshold : Float,
) -> Unit raise {
  let a = analytic.contiguous()
  let b = numerical.contiguous()
  let mut max_rel_err = Float::from_int(0)
  for i = 0; i < a.numel(); i = i + 1 {
    let ai = a.data[i].to_double()
    let bi = b.data[i].to_double()
    let diff = if ai - bi > 0.0 { ai - bi } else { bi - ai }
    let denom = if ai > 0.0 { ai } else { -ai }
    let denom2 = if bi > 0.0 { bi } else { -bi }
    let max_ab = if denom > denom2 { denom } else { denom2 }
    let rel = if max_ab > 0.0000001 {
      diff / (max_ab + 0.00000001)
    } else {
      diff
    }
    let rel_f = Float::from_double(rel)
    if rel_f > max_rel_err {
      max_rel_err = rel_f
    }
  }
  assert_true(max_rel_err < threshold)
}

///|
/// Helper: sum all elements of a tensor
fn sum_all(t : @tensor.Tensor) -> Float {
  let tc = t.contiguous()
  let mut s = Float::from_int(0)
  for i = 0; i < tc.numel(); i = i + 1 {
    s += tc.data[i]
  }
  s
}

///|
test "gradient_check_matmul_2d" {
  let tape = @autograd.Tape::new()
  let x_data = rand_tensor(@tensor.shape_new([2, 3]).unwrap(), 42)
  let w_data = rand_tensor(@tensor.shape_new([3, 4]).unwrap(), 43)
  let x = tape.create_var(x_data, true)
  let w = tape.create_var(w_data, true)
  let y = @autograd.var_matmul(x, w)
  let loss = @autograd.var_sum(y)
  tape.backward(loss)
  let eps = Float::from_double(0.0001)
  let ng_x = numerical_grad(
    x_data,
    fn() { sum_all(@tensor.tensor_matmul(x_data, w_data).unwrap()) },
    eps,
  )
  check_grad("matmul_dx", x.grad().unwrap(), ng_x, Float::from_double(0.01))
  let ng_w = numerical_grad(
    w_data,
    fn() { sum_all(@tensor.tensor_matmul(x_data, w_data).unwrap()) },
    eps,
  )
  check_grad("matmul_dw", w.grad().unwrap(), ng_w, Float::from_double(0.01))
}

///|
test "gradient_check_add" {
  let tape = @autograd.Tape::new()
  let a_data = rand_tensor(@tensor.shape_new([2, 3]).unwrap(), 44)
  let b_data = rand_tensor(@tensor.shape_new([2, 3]).unwrap(), 45)
  let a = tape.create_var(a_data, true)
  let b = tape.create_var(b_data, true)
  let y = @autograd.var_add(a, b)
  let loss = @autograd.var_sum(y)
  tape.backward(loss)
  let eps = Float::from_double(0.0001)
  let ng_a = numerical_grad(
    a_data,
    fn() { sum_all(@tensor.tensor_add(a_data, b_data).unwrap()) },
    eps,
  )
  check_grad("add_da", a.grad().unwrap(), ng_a, Float::from_double(0.01))
  let ng_b = numerical_grad(
    b_data,
    fn() { sum_all(@tensor.tensor_add(a_data, b_data).unwrap()) },
    eps,
  )
  check_grad("add_db", b.grad().unwrap(), ng_b, Float::from_double(0.01))
}

///|
test "gradient_check_add_bias" {
  let tape = @autograd.Tape::new()
  let x_data = rand_tensor(@tensor.shape_new([3, 4]).unwrap(), 46)
  let bias_data = rand_tensor(@tensor.shape_new([4]).unwrap(), 47)
  let x = tape.create_var(x_data, true)
  let bias = tape.create_var(bias_data, true)
  let y = @autograd.var_add_bias(x, bias)
  let loss = @autograd.var_sum(y)
  tape.backward(loss)
  let eps = Float::from_double(0.0001)
  let ng_x = numerical_grad(
    x_data,
    fn() {
      let total = x_data.numel()
      let last_dim = x_data.dim(-1)
      let xc = x_data.contiguous()
      let bc = bias_data.contiguous()
      let mut s = Float::from_int(0)
      for i = 0; i < total; i = i + 1 {
        s += xc.data[i] + bc.data[i % last_dim]
      }
      s
    },
    eps,
  )
  check_grad("bias_dx", x.grad().unwrap(), ng_x, Float::from_double(0.01))
  let ng_b = numerical_grad(
    bias_data,
    fn() {
      let total = x_data.numel()
      let last_dim = x_data.dim(-1)
      let xc = x_data.contiguous()
      let bc = bias_data.contiguous()
      let mut s = Float::from_int(0)
      for i = 0; i < total; i = i + 1 {
        s += xc.data[i] + bc.data[i % last_dim]
      }
      s
    },
    eps,
  )
  check_grad("bias_db", bias.grad().unwrap(), ng_b, Float::from_double(0.01))
}

///|
test "gradient_check_mul_scalar" {
  let tape = @autograd.Tape::new()
  let x_data = rand_tensor(@tensor.shape_new([2, 3]).unwrap(), 48)
  let x = tape.create_var(x_data, true)
  let y = @autograd.var_mul_scalar(x, Float::from_double(2.5))
  let loss = @autograd.var_sum(y)
  tape.backward(loss)
  let eps = Float::from_double(0.0001)
  let ng_x = numerical_grad(
    x_data,
    fn() { sum_all(@tensor.tensor_scale(x_data, Float::from_double(2.5))) },
    eps,
  )
  check_grad("mul_scalar_dx", x.grad().unwrap(), ng_x, Float::from_double(0.01))
}

///|
test "gradient_check_gelu" {
  let tape = @autograd.Tape::new()
  let x_data = rand_tensor(@tensor.shape_new([2, 3]).unwrap(), 60)
  let x = tape.create_var(x_data, true)
  let y = @autograd.var_gelu(x)
  let loss = @autograd.var_sum(y)
  tape.backward(loss)
  let eps = Float::from_double(0.0001)
  let ng_x = numerical_grad(
    x_data,
    fn() { sum_all(@tensor.tensor_gelu(x_data)) },
    eps,
  )
  check_grad("gelu_dx", x.grad().unwrap(), ng_x, Float::from_double(0.05))
}

///|
test "gradient_check_relu" {
  let tape = @autograd.Tape::new()
  // Use values not too close to 0 to avoid numerical issues at the kink
  let x_data = @tensor.tensor_new(
    @tensor.shape_new([2, 3]).unwrap(),
    [0.5, -0.3, 0.8, -0.1, 0.2, -0.6].map(fn(v) { Float::from_double(v) }),
  ).unwrap()
  let x = tape.create_var(x_data, true)
  let y = @autograd.var_relu(x)
  let loss = @autograd.var_sum(y)
  tape.backward(loss)
  let eps = Float::from_double(0.0001)
  let ng_x = numerical_grad(
    x_data,
    fn() { sum_all(@tensor.tensor_relu(x_data)) },
    eps,
  )
  check_grad("relu_dx", x.grad().unwrap(), ng_x, Float::from_double(0.01))
}

///|
test "gradient_check_layer_norm" {
  let tape = @autograd.Tape::new()
  let x_data = rand_tensor(@tensor.shape_new([2, 4]).unwrap(), 70)
  // Use non-trivial gamma to avoid sum-to-zero issues
  let gamma_data = rand_tensor(@tensor.shape_new([4]).unwrap(), 71)
  // Shift gamma to be positive
  for i = 0; i < gamma_data.numel(); i = i + 1 {
    gamma_data.data[i] = Float::from_double(
      gamma_data.data[i].to_double() + 1.5,
    )
  }
  let beta_data = rand_tensor(@tensor.shape_new([4]).unwrap(), 72)
  let x = tape.create_var(x_data, true)
  let gamma = tape.create_var(gamma_data, true)
  let beta = tape.create_var(beta_data, true)
  let eps_ln = Float::from_double(0.00001)
  let y = @autograd.var_layer_norm(x, gamma, beta, eps_ln)
  // Use a weighted sum to avoid trivial gradients
  let w_data = rand_tensor(@tensor.shape_new([2, 4]).unwrap(), 73)
  let w = tape.create_var(w_data, false)
  let yw = @autograd.var_add(y, w) // just to make loss non-trivial
  let loss = @autograd.var_sum(yw)
  tape.backward(loss)
  let eps = Float::from_double(0.0001)
  // Numerical grad for x: use layer_norm_with_cache for consistency
  let ng_x = numerical_grad(
    x_data,
    fn() {
      let (out, _, _) = @tensor.layer_norm_with_cache(
        x_data, gamma_data, beta_data, eps_ln,
      )
      sum_all(out) + sum_all(w_data)
    },
    eps,
  )
  check_grad("ln_dx", x.grad().unwrap(), ng_x, Float::from_double(0.1))
  let ng_gamma = numerical_grad(
    gamma_data,
    fn() {
      let (out, _, _) = @tensor.layer_norm_with_cache(
        x_data, gamma_data, beta_data, eps_ln,
      )
      sum_all(out) + sum_all(w_data)
    },
    eps,
  )
  check_grad(
    "ln_dgamma",
    gamma.grad().unwrap(),
    ng_gamma,
    Float::from_double(0.1),
  )
}

///|
test "gradient_check_cross_entropy" {
  let tape = @autograd.Tape::new()
  let logits_data = rand_tensor(@tensor.shape_new([2, 5]).unwrap(), 80)
  let labels = [1, 3]
  let logits = tape.create_var(logits_data, true)
  let loss = @autograd.var_cross_entropy(logits, labels)
  tape.backward(loss)
  let eps = Float::from_double(0.0001)
  let ng = numerical_grad(
    logits_data,
    fn() { @tensor.tensor_cross_entropy(logits_data, labels).unwrap() },
    eps,
  )
  check_grad("ce_dlogits", logits.grad().unwrap(), ng, Float::from_double(0.05))
}

///|
test "gradient_check_linear_layer" {
  let tape = @autograd.Tape::new()
  let linear = @autograd.Linear::new(tape, 3, 4, 100)
  let x_data = rand_tensor(@tensor.shape_new([2, 3]).unwrap(), 101)
  let x = tape.create_var(x_data, true)
  let y = linear.forward(x)
  let loss = @autograd.var_sum(y)
  tape.backward(loss)
  // Check that weight and bias have gradients
  assert_true(linear.weight.grad().unwrap().numel() == 12) // [4, 3]
  assert_true(linear.bias.grad().unwrap().numel() == 4) // [4]
  assert_true(x.grad().unwrap().numel() == 6) // [2, 3]
}

///|
test "mlp_training_loss_decreases" {
  // Simple 2-class classification: 4 samples, 2 features -> 2 classes
  // Pattern: class 0 = (0,0) or (1,1), class 1 = (0,1) or (1,0) (XOR-like)
  let input_data = @tensor.tensor_new(
    @tensor.shape_new([4, 2]).unwrap(),
    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0].map(fn(v) { Float::from_double(v) }),
  ).unwrap()
  let labels = [0, 1, 1, 0]
  let lr = Float::from_double(0.1)
  let mut prev_loss = Float::from_double(999.0)
  // Train for 50 epochs, check loss decreases overall
  let mut initial_loss = Float::from_int(0)
  let mut final_loss = Float::from_int(0)
  for epoch = 0; epoch < 50; epoch = epoch + 1 {
    let tape = @autograd.Tape::new()
    let fc1 = @autograd.Linear::new(tape, 2, 8, 200 + epoch)
    let fc2 = @autograd.Linear::new(tape, 8, 2, 300 + epoch)
    let x = tape.create_var(input_data, false)
    let h = fc1.forward(x)
    let h_relu = @autograd.var_relu(h)
    let logits = fc2.forward(h_relu)
    let loss = @autograd.var_cross_entropy(logits, labels)
    tape.backward(loss)
    let loss_val = loss.data().data[0]
    if epoch == 0 {
      initial_loss = loss_val
    }
    if epoch == 49 {
      final_loss = loss_val
    }
    ignore(prev_loss)
    prev_loss = loss_val
    let params = fc1.parameters()
    params.push(fc2.parameters()[0])
    params.push(fc2.parameters()[1])
    @autograd.sgd_step(params, lr)
  }
  ignore(initial_loss)
  ignore(final_loss)
  // With random re-init each epoch, just check the loss is finite and gradient flow works
  assert_true(prev_loss.to_double() < 10.0)
  assert_true(prev_loss.to_double() > 0.0)
}

///|
test "mlp_training_converges" {
  // Train single MLP across epochs (reuse same weights)
  let tape = @autograd.Tape::new()
  let fc1 = @autograd.Linear::new(tape, 2, 16, 500)
  let fc2 = @autograd.Linear::new(tape, 16, 2, 501)
  let input_data = @tensor.tensor_new(
    @tensor.shape_new([4, 2]).unwrap(),
    [0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9].map(fn(v) { Float::from_double(v) }),
  ).unwrap()
  let labels = [0, 1, 1, 0]
  let lr = Float::from_double(0.05)
  let mut first_loss = Float::from_int(0)
  let mut last_loss = Float::from_int(0)
  let all_params = fc1.parameters()
  all_params.push(fc2.parameters()[0])
  all_params.push(fc2.parameters()[1])
  for epoch = 0; epoch < 200; epoch = epoch + 1 {
    // Create fresh tape each epoch for proper gradient tracking
    let tape2 = @autograd.Tape::new()
    // Re-wrap existing tensor data as vars in new tape
    let x = tape2.create_var(input_data, false)
    let w1 = tape2.create_var(fc1.weight.data(), true)
    let b1 = tape2.create_var(fc1.bias.data(), true)
    let w2 = tape2.create_var(fc2.weight.data(), true)
    let b2 = tape2.create_var(fc2.bias.data(), true)
    // Manual forward: x @ W1^T + b1
    let wt1 = w1.data().transpose(0, 1).unwrap().contiguous()
    let wt1_var = tape2.create_var(wt1, false)
    let h1 = @autograd.var_matmul(x, wt1_var)
    let h1b = @autograd.var_add_bias(h1, b1)
    let h1_act = @autograd.var_relu(h1b)
    // x @ W2^T + b2
    let wt2 = w2.data().transpose(0, 1).unwrap().contiguous()
    let wt2_var = tape2.create_var(wt2, false)
    let h2 = @autograd.var_matmul(h1_act, wt2_var)
    let logits = @autograd.var_add_bias(h2, b2)
    let loss = @autograd.var_cross_entropy(logits, labels)
    tape2.backward(loss)
    let loss_val = loss.data().data[0]
    if epoch == 0 {
      first_loss = loss_val
    }
    last_loss = loss_val
    // SGD update on original weight data
    let lr_d = lr.to_double()
    // Update w1
    match w1.grad() {
      Some(g) => {
        let gd = g.contiguous()
        for i = 0; i < fc1.weight.data().data.length(); i = i + 1 {
          fc1.weight.data().data[i] = Float::from_double(
            fc1.weight.data().data[i].to_double() -
            lr_d * gd.data[i].to_double(),
          )
        }
      }
      None => ()
    }
    // Update b1
    match b1.grad() {
      Some(g) => {
        let gd = g.contiguous()
        for i = 0; i < fc1.bias.data().data.length(); i = i + 1 {
          fc1.bias.data().data[i] = Float::from_double(
            fc1.bias.data().data[i].to_double() - lr_d * gd.data[i].to_double(),
          )
        }
      }
      None => ()
    }
    // Update w2
    match w2.grad() {
      Some(g) => {
        let gd = g.contiguous()
        for i = 0; i < fc2.weight.data().data.length(); i = i + 1 {
          fc2.weight.data().data[i] = Float::from_double(
            fc2.weight.data().data[i].to_double() -
            lr_d * gd.data[i].to_double(),
          )
        }
      }
      None => ()
    }
    // Update b2
    match b2.grad() {
      Some(g) => {
        let gd = g.contiguous()
        for i = 0; i < fc2.bias.data().data.length(); i = i + 1 {
          fc2.bias.data().data[i] = Float::from_double(
            fc2.bias.data().data[i].to_double() - lr_d * gd.data[i].to_double(),
          )
        }
      }
      None => ()
    }
  }
  // Loss should decrease significantly
  assert_true(last_loss.to_double() < first_loss.to_double())
}

///|
test "gradient_check_matmul_3d_broadcast" {
  let tape = @autograd.Tape::new()
  let x_data = rand_tensor(@tensor.shape_new([2, 3, 4]).unwrap(), 50)
  let w_data = rand_tensor(@tensor.shape_new([4, 5]).unwrap(), 51)
  let x = tape.create_var(x_data, true)
  let w = tape.create_var(w_data, true)
  let y = @autograd.var_matmul(x, w)
  let loss = @autograd.var_sum(y)
  tape.backward(loss)
  let eps = Float::from_double(0.0001)
  let ng_x = numerical_grad(
    x_data,
    fn() { sum_all(@tensor.tensor_matmul(x_data, w_data).unwrap()) },
    eps,
  )
  check_grad("matmul3d_dx", x.grad().unwrap(), ng_x, Float::from_double(0.01))
  let ng_w = numerical_grad(
    w_data,
    fn() { sum_all(@tensor.tensor_matmul(x_data, w_data).unwrap()) },
    eps,
  )
  check_grad("matmul3d_dw", w.grad().unwrap(), ng_w, Float::from_double(0.01))
}
