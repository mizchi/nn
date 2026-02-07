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
/// Computes softmax once and caches for backward (avoids double computation).
pub fn var_cross_entropy(logits : Var, labels : Array[Int]) -> Var {
  let tape = logits.tape
  let ld = logits.data().contiguous()
  let batch = ld.dim(0)
  let vocab = ld.dim(1)
  let total = batch * vocab
  // Compute softmax once and cache for backward
  let probs = FixedArray::make(total, Float::from_int(0))
  for i = 0; i < total; i = i + 1 {
    probs[i] = ld.data[i]
  }
  @tensor.softmax_inplace(probs, batch, vocab)
  // Compute loss = -sum(log(probs[label])) / batch
  let mut loss = Float::from_int(0)
  for i = 0; i < batch; i = i + 1 {
    loss = loss - @math.lnf(probs[i * vocab + labels[i]])
  }
  let inv_batch = Float::from_int(1) / Float::from_int(batch)
  loss = loss * inv_batch
  let result_shape = @tensor.shape_new([1]).unwrap()
  let result = @tensor.tensor_new_fixed(
    result_shape,
    FixedArray::make(1, loss),
  )
  tape.push_node(result, fn(dy) {
    let dy_scalar = dy.contiguous().data[0]
    // d_logits = (probs - one_hot) / batch, scaled by dy
    let scale = inv_batch * dy_scalar
    let d_data = FixedArray::make(total, Float::from_int(0))
    // d_data = scale * probs (saxpy: d_data += scale * probs, d_data starts at 0)
    @tensor.saxpy(total, scale, probs, d_data)
    for i = 0; i < batch; i = i + 1 {
      d_data[i * vocab + labels[i]] = d_data[i * vocab + labels[i]] - scale
    }
    let d_logits = @tensor.tensor_new_fixed(ld.shape, d_data)
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
