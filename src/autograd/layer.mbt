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
    0, 1, n_rows, out_features, in_features, one, xd.data, in_features,
    wd.data, in_features, zero, y_data, out_features,
  )
  @tensor.bias_add_inplace(y_data, bd.data, total, out_features)
  let result = @tensor.tensor_new_fixed(y_shape, y_data)
  let w_id = self.weight.id
  let x_id = x.id
  let bias_id = self.bias.id
  tape.push_node(result, fn(dy) {
    let dyc = dy.contiguous()
    // dx = dy @ W
    let dx_data = FixedArray::make(n_rows * in_features, zero)
    @tensor.sgemm(
      0, 0, n_rows, in_features, out_features, one, dyc.data, out_features,
      wd.data, in_features, zero, dx_data, in_features,
    )
    accumulate(tape, x_id, @tensor.tensor_new_fixed(xd.shape, dx_data))
    // dW = dy^T @ x
    let dw_data = FixedArray::make(out_features * in_features, zero)
    @tensor.sgemm(
      1, 0, out_features, in_features, n_rows, one, dyc.data, out_features,
      xd.data, in_features, zero, dw_data, in_features,
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
pub fn Linear::forward_ws(
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
    0, 1, n_rows, out_features, in_features, one, xd.data, in_features,
    wd.data, in_features, zero, ws.y_data, out_features,
  )
  @tensor.bias_add_inplace(ws.y_data, bd.data, total, out_features)
  let result = @tensor.tensor_new_fixed(y_shape, ws.y_data)
  let w_id = self.weight.id
  let x_id = x.id
  let bias_id = self.bias.id
  tape.push_node(result, fn(dy) {
    let dyc = dy.contiguous()
    // dx = dy @ W (reuse ws.dx_data, sgemm beta=0 overwrites)
    @tensor.sgemm(
      0, 0, n_rows, in_features, out_features, one, dyc.data, out_features,
      wd.data, in_features, zero, ws.dx_data, in_features,
    )
    accumulate(tape, x_id, @tensor.tensor_new_fixed(xd.shape, ws.dx_data))
    // dW = dy^T @ x (reuse ws.dw_data)
    @tensor.sgemm(
      1, 0, out_features, in_features, n_rows, one, dyc.data, out_features,
      xd.data, in_features, zero, ws.dw_data, in_features,
    )
    accumulate(tape, w_id, @tensor.tensor_new_fixed(wd.shape, ws.dw_data))
    // dbias = sum(dy, axis=0) (must zero db_data first since backward accumulates)
    for i = 0; i < ws.db_data.length(); i = i + 1 {
      ws.db_data[i] = zero
    }
    @tensor.bias_add_backward(dyc.data, ws.db_data, dyc.numel(), out_features)
    accumulate(
      tape, bias_id, @tensor.tensor_new_fixed(bd.shape, ws.db_data),
    )
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
