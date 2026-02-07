///|
/// GELU activation: y = gelu(x)
/// Uses existing tensor_gelu / gelu_backward from @tensor.
pub fn var_gelu(x : Var) -> Var {
  let tape = x.tape
  let xd = x.data().contiguous()
  let result = @tensor.tensor_gelu(xd)
  tape.push_node(result, fn(dy) {
    let dx = @tensor.gelu_backward(dy, xd)
    accumulate(tape, x.id, dx)
  })
}

///|
/// Layer normalization over the last dimension.
/// gamma [features], beta [features].
/// Uses layer_norm_with_cache for forward (caches mean/rstd for backward).
pub fn var_layer_norm(x : Var, gamma : Var, beta : Var, eps : Float) -> Var {
  let tape = x.tape
  let xd = x.data().contiguous()
  let gd = gamma.data().contiguous()
  let bd = beta.data().contiguous()
  let (out, mean, rstd) = @tensor.layer_norm_with_cache(xd, gd, bd, eps)
  tape.push_node(out, fn(dy) {
    let d_gamma = @tensor.tensor_zeros(gd.shape)
    let d_beta = @tensor.tensor_zeros(bd.shape)
    let dx = @tensor.layer_norm_backward(
      dy, xd, mean, rstd, gd, d_gamma, d_beta,
    )
    accumulate(tape, x.id, dx)
    accumulate(tape, gamma.id, d_gamma)
    accumulate(tape, beta.id, d_beta)
  })
}

///|
/// Cross-entropy loss: logits [batch, vocab], labels [batch] -> scalar loss.
/// Uses existing cross_entropy_backward from @tensor.
pub fn var_cross_entropy(logits : Var, labels : Array[Int]) -> Var {
  let tape = logits.tape
  let ld = logits.data().contiguous()
  let loss_val = @tensor.tensor_cross_entropy(ld, labels).unwrap()
  let result_shape = @tensor.shape_new([1]).unwrap()
  let result = @tensor.tensor_new_fixed(
    result_shape,
    FixedArray::make(1, loss_val),
  )
  tape.push_node(result, fn(dy) {
    let dy_scalar = dy.contiguous().data[0]
    let d_logits = @tensor.cross_entropy_backward(ld, labels)
    // Scale by upstream gradient
    if dy_scalar != Float::from_int(1) {
      let d_data = d_logits.contiguous().data
      @tensor.scale_inplace(d_data, dy_scalar, d_data.length())
    }
    accumulate(tape, logits.id, d_logits)
  })
}

///|
/// ReLU activation: y = max(0, x)
pub fn var_relu(x : Var) -> Var {
  let tape = x.tape
  let xd = x.data().contiguous()
  let result = @tensor.tensor_relu(xd)
  tape.push_node(result, fn(dy) {
    let dyc = dy.contiguous()
    let n = xd.numel()
    let dx_data = FixedArray::make(n, Float::from_int(0))
    @tensor.relu_backward_inplace(dyc.data, xd.data, dx_data, n)
    let dx = @tensor.tensor_new_fixed(xd.shape, dx_data)
    accumulate(tape, x.id, dx)
  })
}
