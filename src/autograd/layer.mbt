///|
/// Linear layer: y = x @ W^T + bias
/// weight: [out_features, in_features], bias: [out_features]
/// Forward: x [*, in_features] @ weight^T [in_features, out_features] + bias
pub struct Linear {
  weight : Var // [out_features, in_features]
  bias : Var // [out_features]
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
/// Forward pass: x @ W^T + bias
/// x: [*, in_features] -> [*, out_features]
/// Uses BLAS sgemm with trans flags to avoid transpose copies.
pub fn Linear::forward(self : Linear, x : Var) -> Var {
  let tape = x.tape
  let xd = x.data().contiguous()
  let wd = self.weight.data().contiguous()
  let in_features = wd.dim(1)
  let out_features = wd.dim(0)
  // n_rows = total rows (batch*seq for 3D, batch for 2D)
  let n_rows = xd.numel() / in_features
  // y = x @ W^T: sgemm(NoTrans, Trans, n_rows, out, in, 1, x, in, w, in, 0, y, out)
  let y_shape = if xd.ndim() == 2 {
    @tensor.shape_new([n_rows, out_features]).unwrap()
  } else {
    @tensor.shape_new([xd.dim(0), xd.dim(1), out_features]).unwrap()
  }
  let y_data = FixedArray::make(n_rows * out_features, Float::from_int(0))
  @tensor.sgemm(
    0, 1, n_rows, out_features, in_features, Float::from_int(1), xd.data,
    in_features, wd.data, in_features, Float::from_int(0), y_data, out_features,
  )
  let result = @tensor.tensor_new_fixed(y_shape, y_data)
  let w_id = self.weight.id
  let x_id = x.id
  let y = tape.push_node(result, fn(dy) {
    let dyc = dy.contiguous()
    // dx = dy @ W: sgemm(NoTrans, NoTrans, n_rows, in, out, 1, dy, out, w, in, 0, dx, in)
    let dx_data = FixedArray::make(n_rows * in_features, Float::from_int(0))
    @tensor.sgemm(
      0, 0, n_rows, in_features, out_features, Float::from_int(1), dyc.data,
      out_features, wd.data, in_features, Float::from_int(0), dx_data,
      in_features,
    )
    let dx = @tensor.tensor_new_fixed(xd.shape, dx_data)
    accumulate(tape, x_id, dx)
    // dW = dy^T @ x: sgemm(Trans, NoTrans, out, in, n_rows, 1, dy, out, x, in, 0, dw, in)
    let dw_data = FixedArray::make(out_features * in_features, Float::from_int(0))
    @tensor.sgemm(
      1, 0, out_features, in_features, n_rows, Float::from_int(1), dyc.data,
      out_features, xd.data, in_features, Float::from_int(0), dw_data,
      in_features,
    )
    let dw = @tensor.tensor_new_fixed(wd.shape, dw_data)
    accumulate(tape, w_id, dw)
  })
  var_add_bias(y, self.bias)
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
