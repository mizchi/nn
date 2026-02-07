///|
/// Linear layer: y = x @ W^T + bias
/// weight: [out_features, in_features], bias: [out_features]
/// Forward: x [*, in_features] @ weight^T [in_features, out_features] + bias
pub struct Linear {
  weight : Var // [out_features, in_features]
  bias : Var // [out_features]
}

///|
/// Pre-allocated workspace buffers for Linear forward/backward.
/// Reuse across iterations to avoid repeated allocation.
pub struct LinearWorkspace {
  y_data : FixedArray[Float] // [n_rows * out_features]
  dx_data : FixedArray[Float] // [n_rows * in_features]
  dw_data : FixedArray[Float] // [out_features * in_features]
  db_data : FixedArray[Float] // [out_features]
}

///|
pub fn LinearWorkspace::new(
  n_rows : Int,
  in_features : Int,
  out_features : Int,
) -> LinearWorkspace {
  let zero = Float::from_int(0)
  LinearWorkspace::{
    y_data: FixedArray::make(n_rows * out_features, zero),
    dx_data: FixedArray::make(n_rows * in_features, zero),
    dw_data: FixedArray::make(out_features * in_features, zero),
    db_data: FixedArray::make(out_features, zero),
  }
}

///|
/// Create a Linear layer from existing Var tensors.
pub fn Linear::from_vars(weight : Var, bias : Var) -> Linear {
  Linear::{ weight, bias }
}

///|
/// Create a new Linear layer with random initialization.
pub fn Linear::new(
  tape : Tape,
  in_features : Int,
  out_features : Int,
  seed : Int,
) -> Linear {
  let mut rng = seed
  let scale = Float::from_double(
    (2.0 / (in_features + out_features).to_double()).sqrt(), // Xavier init
  )
  // Initialize weight [out_features, in_features]
  let w_shape = @tensor.shape_new([out_features, in_features]).unwrap()
  let w_data = FixedArray::make(out_features * in_features, Float::from_int(0))
  for i = 0; i < w_data.length(); i = i + 1 {
    rng = rng * 1103515245 + 12345
    let val = ((rng >> 16) & 0x7FFF).to_double() / 32768.0 - 0.5
    w_data[i] = Float::from_double(val * 2.0 * scale.to_double())
  }
  let weight = tape.create_var(@tensor.tensor_new_fixed(w_shape, w_data), true)
  // Initialize bias [out_features] to zeros
  let b_shape = @tensor.shape_new([out_features]).unwrap()
  let bias = tape.create_var(@tensor.tensor_zeros(b_shape), true)
  Linear::{ weight, bias }
}

///|
/// Forward pass: x @ W^T + bias (allocates fresh buffers)
/// x: [*, in_features] -> [*, out_features]
pub fn Linear::forward(self : Linear, x : Var) -> Var {
  let tape = x.tape
  let xd = x.data().contiguous()
  let wd = self.weight.data().contiguous()
  let bd = self.bias.data().contiguous()
  let in_features = wd.dim(1)
  let out_features = wd.dim(0)
  let n_rows = xd.numel() / in_features
  let one = Float::from_int(1)
  let zero = Float::from_int(0)
  let total = n_rows * out_features
  let y_shape = if xd.ndim() == 2 {
    @tensor.shape_new([n_rows, out_features]).unwrap()
  } else {
    @tensor.shape_new([xd.dim(0), xd.dim(1), out_features]).unwrap()
  }
  // y = x @ W^T + bias (sgemm then in-place bias add)
  let y_data = FixedArray::make(total, zero)
  @tensor.sgemm(
    0,
    1,
    n_rows,
    out_features,
    in_features,
    one,
    xd.data,
    in_features,
    wd.data,
    in_features,
    zero,
    y_data,
    out_features,
  )
  @tensor.bias_add_inplace(y_data, bd.data, total, out_features)
  let result = @tensor.tensor_new_fixed(y_shape, y_data)
  let w_id = self.weight.id
  let x_id = x.id
  let bias_id = self.bias.id
  let x_needs_grad = needs_grad(tape, x_id)
  tape.push_node(result, fn(dy) {
    let dyc = dy.contiguous()
    // dx = dy @ W (skip if x doesn't need gradient)
    if x_needs_grad {
      let dx_data = FixedArray::make(n_rows * in_features, zero)
      @tensor.sgemm(
        0,
        0,
        n_rows,
        in_features,
        out_features,
        one,
        dyc.data,
        out_features,
        wd.data,
        in_features,
        zero,
        dx_data,
        in_features,
      )
      accumulate(tape, x_id, @tensor.tensor_new_fixed(xd.shape, dx_data))
    }
    // dW = dy^T @ x
    let dw_data = FixedArray::make(out_features * in_features, zero)
    @tensor.sgemm(
      1,
      0,
      out_features,
      in_features,
      n_rows,
      one,
      dyc.data,
      out_features,
      xd.data,
      in_features,
      zero,
      dw_data,
      in_features,
    )
    accumulate(tape, w_id, @tensor.tensor_new_fixed(wd.shape, dw_data))
    // dbias = sum(dy, axis=0)
    let db_data = FixedArray::make(out_features, zero)
    @tensor.bias_add_backward(dyc.data, db_data, dyc.numel(), out_features)
    accumulate(tape, bias_id, @tensor.tensor_new_fixed(bd.shape, db_data))
  })
}

///|
/// Forward pass with pre-allocated workspace (avoids allocation).
/// x: [*, in_features] -> [*, out_features]
pub fn Linear::forward_ws(self : Linear, x : Var, ws : LinearWorkspace) -> Var {
  let tape = x.tape
  let xd = x.data().contiguous()
  let wd = self.weight.data().contiguous()
  let bd = self.bias.data().contiguous()
  let in_features = wd.dim(1)
  let out_features = wd.dim(0)
  let n_rows = xd.numel() / in_features
  let one = Float::from_int(1)
  let zero = Float::from_int(0)
  let total = n_rows * out_features
  let y_shape = if xd.ndim() == 2 {
    @tensor.shape_new([n_rows, out_features]).unwrap()
  } else {
    @tensor.shape_new([xd.dim(0), xd.dim(1), out_features]).unwrap()
  }
  // y = x @ W^T + bias (reuse workspace, sgemm beta=0 overwrites fully)
  @tensor.sgemm(
    0,
    1,
    n_rows,
    out_features,
    in_features,
    one,
    xd.data,
    in_features,
    wd.data,
    in_features,
    zero,
    ws.y_data,
    out_features,
  )
  @tensor.bias_add_inplace(ws.y_data, bd.data, total, out_features)
  let result = @tensor.tensor_new_fixed(y_shape, ws.y_data)
  let w_id = self.weight.id
  let x_id = x.id
  let bias_id = self.bias.id
  let x_needs_grad = needs_grad(tape, x_id)
  let x_shape = xd.shape
  let w_shape = wd.shape
  let b_shape = bd.shape
  tape.push_node(result, fn(dy) {
    let dy_data = dy.contiguous().data
    // dW = dy^T @ x (reuse workspace)
    @tensor.sgemm(
      1,
      0,
      out_features,
      in_features,
      n_rows,
      one,
      dy_data,
      out_features,
      xd.data,
      in_features,
      zero,
      ws.dw_data,
      in_features,
    )
    accumulate_raw(tape, w_id, ws.dw_data, w_shape)
    // dbias = sum(dy, axis=0) (must zero first)
    for i = 0; i < ws.db_data.length(); i = i + 1 {
      ws.db_data[i] = zero
    }
    @tensor.bias_add_backward(dy_data, ws.db_data, dy.numel(), out_features)
    accumulate_raw(tape, bias_id, ws.db_data, b_shape)
    // dx = dy @ W (skip if x doesn't need gradient)
    if x_needs_grad {
      @tensor.sgemm(
        0,
        0,
        n_rows,
        in_features,
        out_features,
        one,
        dy_data,
        out_features,
        wd.data,
        in_features,
        zero,
        ws.dx_data,
        in_features,
      )
      accumulate_raw(tape, x_id, ws.dx_data, x_shape)
    }
  })
}

///|
/// Fused Linear+ReLU forward with workspace: y = relu(x @ W^T + bias)
/// Saves one tape node and eliminates closure overhead between linear and relu.
/// Allocates fresh relu cache buffers each call to avoid GC old-gen cache aliasing.
pub fn Linear::forward_relu_ws(
  self : Linear,
  x : Var,
  ws : LinearWorkspace,
) -> Var {
  let tape = x.tape
  let xd = x.data().contiguous()
  let wd = self.weight.data().contiguous()
  let bd = self.bias.data().contiguous()
  let in_features = wd.dim(1)
  let out_features = wd.dim(0)
  let n_rows = xd.numel() / in_features
  let one = Float::from_int(1)
  let zero = Float::from_int(0)
  let total = n_rows * out_features
  let y_shape = if xd.ndim() == 2 {
    @tensor.shape_new([n_rows, out_features]).unwrap()
  } else {
    @tensor.shape_new([xd.dim(0), xd.dim(1), out_features]).unwrap()
  }
  // Fresh alloc for pre-relu cache (short-lived, stays in GC young gen)
  let relu_pre = FixedArray::make(total, zero)
  // Forward step 1: linear output into relu_pre (cache for relu backward)
  @tensor.sgemm(
    0,
    1,
    n_rows,
    out_features,
    in_features,
    one,
    xd.data,
    in_features,
    wd.data,
    in_features,
    zero,
    relu_pre,
    out_features,
  )
  @tensor.bias_add_inplace(relu_pre, bd.data, total, out_features)
  // Forward step 2: relu(relu_pre) -> ws.y_data (the actual output)
  @tensor.relu_forward(relu_pre, ws.y_data, total)
  let result = @tensor.tensor_new_fixed(y_shape, ws.y_data)
  let w_id = self.weight.id
  let x_id = x.id
  let bias_id = self.bias.id
  let x_needs_grad = needs_grad(tape, x_id)
  let x_shape = xd.shape
  let w_shape = wd.shape
  let b_shape = bd.shape
  tape.push_node(result, fn(dy) {
    let dy_data = dy.contiguous().data
    // Fresh alloc for relu backward output (avoids old-gen cache aliasing)
    let relu_dx = FixedArray::make(total, zero)
    // Backward step 1: relu backward
    @tensor.relu_backward_inplace(dy_data, relu_pre, relu_dx, total)
    // Backward step 2: linear backward using relu_dx as effective dy
    // dW = relu_dx^T @ x
    @tensor.sgemm(
      1,
      0,
      out_features,
      in_features,
      n_rows,
      one,
      relu_dx,
      out_features,
      xd.data,
      in_features,
      zero,
      ws.dw_data,
      in_features,
    )
    accumulate_raw(tape, w_id, ws.dw_data, w_shape)
    // dbias = sum(relu_dx, axis=0)
    for i = 0; i < ws.db_data.length(); i = i + 1 {
      ws.db_data[i] = zero
    }
    @tensor.bias_add_backward(relu_dx, ws.db_data, total, out_features)
    accumulate_raw(tape, bias_id, ws.db_data, b_shape)
    // dx = relu_dx @ W (skip if x doesn't need gradient)
    if x_needs_grad {
      @tensor.sgemm(
        0,
        0,
        n_rows,
        in_features,
        out_features,
        one,
        relu_dx,
        out_features,
        wd.data,
        in_features,
        zero,
        ws.dx_data,
        in_features,
      )
      accumulate_raw(tape, x_id, ws.dx_data, x_shape)
    }
  })
}

///|
/// Fused Linear+ReLU forward using C-managed relu workspace: y = relu(x @ W^T + bias)
/// Pre-relu values are stored in C-managed (malloc) memory, completely outside GC.
/// This avoids GC old-generation cache aliasing that causes 10x relu_backward slowdown.
/// Pair with forward_ws_managed on the downstream linear to get full C-managed path.
pub fn Linear::forward_relu_managed(
  self : Linear,
  x : Var,
  ws : LinearWorkspace,
) -> Var {
  let tape = x.tape
  let xd = x.data().contiguous()
  let wd = self.weight.data().contiguous()
  let bd = self.bias.data().contiguous()
  let in_features = wd.dim(1)
  let out_features = wd.dim(0)
  let n_rows = xd.numel() / in_features
  let one = Float::from_int(1)
  let zero = Float::from_int(0)
  let total = n_rows * out_features
  let y_shape = if xd.ndim() == 2 {
    @tensor.shape_new([n_rows, out_features]).unwrap()
  } else {
    @tensor.shape_new([xd.dim(0), xd.dim(1), out_features]).unwrap()
  }
  // Forward step 1: sgemm writes directly to C-managed relu_pre buffer
  @tensor.sgemm_to_relu_pre(
    0,
    1,
    n_rows,
    out_features,
    in_features,
    one,
    xd.data,
    in_features,
    wd.data,
    in_features,
    total,
  )
  @tensor.bias_add_relu_pre(bd.data, total, out_features)
  // Forward step 2: relu(C-managed relu_pre) -> ws.y_data
  @tensor.relu_from_pre(ws.y_data, total)
  let result = @tensor.tensor_new_fixed(y_shape, ws.y_data)
  let w_id = self.weight.id
  let x_id = x.id
  let bias_id = self.bias.id
  let x_needs_grad = needs_grad(tape, x_id)
  let x_shape = xd.shape
  let w_shape = wd.shape
  let b_shape = bd.shape
  tape.push_node(result, fn(dy) {
    ignore(dy)
    // Backward step 1: relu backward — dy and x from C-managed buffers
    // Reuse ws.y_data as relu_dx (forward output no longer needed during backward)
    // This avoids FixedArray::make inside the closure which triggers GC + cache pollution
    @tensor.relu_backward_managed(ws.y_data, total)
    // Backward step 2: dW = relu_dx^T @ x
    @tensor.sgemm(
      1,
      0,
      out_features,
      in_features,
      n_rows,
      one,
      ws.y_data,
      out_features,
      xd.data,
      in_features,
      zero,
      ws.dw_data,
      in_features,
    )
    accumulate_raw(tape, w_id, ws.dw_data, w_shape)
    // dbias = sum(relu_dx, axis=0)
    for i = 0; i < ws.db_data.length(); i = i + 1 {
      ws.db_data[i] = zero
    }
    @tensor.bias_add_backward(ws.y_data, ws.db_data, total, out_features)
    accumulate_raw(tape, bias_id, ws.db_data, b_shape)
    // dx = relu_dx @ W (skip if x doesn't need gradient)
    if x_needs_grad {
      @tensor.sgemm(
        0,
        0,
        n_rows,
        in_features,
        out_features,
        one,
        ws.y_data,
        out_features,
        wd.data,
        in_features,
        zero,
        ws.dx_data,
        in_features,
      )
      accumulate_raw(tape, x_id, ws.dx_data, x_shape)
    }
  })
}

///|
/// Forward pass with workspace, writing backward dx to C-managed grad_buf.
/// Pair with forward_relu_managed upstream for full C-managed relu backward path.
pub fn Linear::forward_ws_managed(
  self : Linear,
  x : Var,
  ws : LinearWorkspace,
) -> Var {
  let tape = x.tape
  let xd = x.data().contiguous()
  let wd = self.weight.data().contiguous()
  let bd = self.bias.data().contiguous()
  let in_features = wd.dim(1)
  let out_features = wd.dim(0)
  let n_rows = xd.numel() / in_features
  let one = Float::from_int(1)
  let zero = Float::from_int(0)
  let total = n_rows * out_features
  let y_shape = if xd.ndim() == 2 {
    @tensor.shape_new([n_rows, out_features]).unwrap()
  } else {
    @tensor.shape_new([xd.dim(0), xd.dim(1), out_features]).unwrap()
  }
  // y = x @ W^T + bias (reuse workspace, sgemm beta=0 overwrites fully)
  @tensor.sgemm(
    0,
    1,
    n_rows,
    out_features,
    in_features,
    one,
    xd.data,
    in_features,
    wd.data,
    in_features,
    zero,
    ws.y_data,
    out_features,
  )
  @tensor.bias_add_inplace(ws.y_data, bd.data, total, out_features)
  let result = @tensor.tensor_new_fixed(y_shape, ws.y_data)
  let w_id = self.weight.id
  let x_id = x.id
  let bias_id = self.bias.id
  let x_needs_grad = needs_grad(tape, x_id)
  let x_shape = xd.shape
  let w_shape = wd.shape
  let b_shape = bd.shape
  let dx_total = n_rows * in_features
  tape.push_node(result, fn(dy) {
    let dy_data = dy.contiguous().data
    // dW = dy^T @ x (reuse workspace)
    @tensor.sgemm(
      1,
      0,
      out_features,
      in_features,
      n_rows,
      one,
      dy_data,
      out_features,
      xd.data,
      in_features,
      zero,
      ws.dw_data,
      in_features,
    )
    accumulate_raw(tape, w_id, ws.dw_data, w_shape)
    // dbias = sum(dy, axis=0) (must zero first)
    for i = 0; i < ws.db_data.length(); i = i + 1 {
      ws.db_data[i] = zero
    }
    @tensor.bias_add_backward(dy_data, ws.db_data, dy.numel(), out_features)
    accumulate_raw(tape, bias_id, ws.db_data, b_shape)
    // dx = dy @ W — write to C-managed grad_buf, then copy to ws.dx_data
    if x_needs_grad {
      @tensor.sgemm_to_grad_buf(
        0,
        0,
        n_rows,
        in_features,
        out_features,
        one,
        dy_data,
        out_features,
        wd.data,
        in_features,
        dx_total,
      )
      @tensor.grad_buf_to_fixed(ws.dx_data, dx_total)
      accumulate_raw(tape, x_id, ws.dx_data, x_shape)
    }
  })
}

///|
/// Get all trainable parameters.
pub fn Linear::parameters(self : Linear) -> Array[Var] {
  [self.weight, self.bias]
}

///|
/// Layer normalization layer.
pub struct LayerNormLayer {
  gamma : Var // [features]
  beta : Var // [features]
  eps : Float
}

///|
pub fn LayerNormLayer::new(
  tape : Tape,
  features : Int,
  eps : Float,
) -> LayerNormLayer {
  let shape = @tensor.shape_new([features]).unwrap()
  let gamma = tape.create_var(@tensor.tensor_ones(shape), true)
  let beta = tape.create_var(@tensor.tensor_zeros(shape), true)
  LayerNormLayer::{ gamma, beta, eps }
}

///|
pub fn LayerNormLayer::forward(self : LayerNormLayer, x : Var) -> Var {
  var_layer_norm(x, self.gamma, self.beta, self.eps)
}

///|
pub fn LayerNormLayer::parameters(self : LayerNormLayer) -> Array[Var] {
  [self.gamma, self.beta]
}
