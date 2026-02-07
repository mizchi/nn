///|
/// Gradient struct for a single transformer block (mirrors TransformerBlockParams)
pub struct TransformerBlockGrads {
  d_ln1_gamma : Tensor
  d_ln1_beta : Tensor
  d_w_q : Tensor
  d_w_k : Tensor
  d_w_v : Tensor
  d_w_o : Tensor
  d_ln2_gamma : Tensor
  d_ln2_beta : Tensor
  d_ff_w1 : Tensor
  d_ff_b1 : Tensor
  d_ff_w2 : Tensor
  d_ff_b2 : Tensor
}

///|
/// Gradient struct for full transformer (mirrors TransformerParams)
pub struct TransformerGrads {
  d_token_embedding : Tensor
  d_pos_embedding : Tensor
  d_blocks : Array[TransformerBlockGrads]
  d_ln_final_gamma : Tensor
  d_ln_final_beta : Tensor
  d_lm_head : Tensor
}

///|
/// Cache for a single transformer block (intermediate activations for backward)
pub struct TransformerBlockCache {
  x_input : Tensor // input to this block [batch, seq, d_model]
  ln1_out : Tensor // output of LN1
  ln1_mean : FixedArray[Float] // per-position mean for LN1
  ln1_rstd : FixedArray[Float] // per-position reciprocal std for LN1
  q_proj : Tensor // [batch, seq, d_model]
  k_proj : Tensor // [batch, seq, d_model]
  v_proj : Tensor // [batch, seq, d_model]
  attn_weights : FixedArray[Float] // [batch*num_heads*seq_q*seq_k] softmax output
  attn_out_pre_wo : Tensor // [batch, seq, d_model] before W_o
  x_after_res1 : Tensor // after first residual
  ln2_out : Tensor // output of LN2
  ln2_mean : FixedArray[Float]
  ln2_rstd : FixedArray[Float]
  ff_pre_gelu : Tensor // W1*x + b1, before GELU [batch, seq, d_ff]
  ff_post_gelu : Tensor // after GELU [batch, seq, d_ff]
}

///|
/// Cache for full transformer forward pass
pub struct TransformerCache {
  tokens : Array[Array[Int]]
  x_with_pos : Tensor // after token + positional embedding
  block_caches : Array[TransformerBlockCache]
  hidden_before_final_ln : Tensor
  ln_final_mean : FixedArray[Float]
  ln_final_rstd : FixedArray[Float]
  ln_final_out : Tensor
  logits : Tensor // [batch, seq, vocab_size]
}

///|
pub struct TransformerLmStepMetric {
  step_ns : UInt64
  loss : Float
  perplexity : Float
}

///|
priv struct TransformerBlockForwardWorkspace {
  ln1_out : FixedArray[Float]
  q_proj : FixedArray[Float]
  k_proj : FixedArray[Float]
  v_proj : FixedArray[Float]
  attn_weights : FixedArray[Float]
  attn_concat : FixedArray[Float]
  attn_out : FixedArray[Float]
  x_after_res1 : FixedArray[Float]
  ln2_out : FixedArray[Float]
  hidden_out : FixedArray[Float]
}

///|
priv struct TransformerForwardWorkspace {
  blocks : Array[TransformerBlockForwardWorkspace]
  ln_final_out : FixedArray[Float]
}

///|
fn transformer_forward_workspace(
  config : TransformerConfig,
  batch : Int,
  seq : Int,
) -> TransformerForwardWorkspace {
  let d_model = config.d_model
  let proj_size = batch * seq * d_model
  let attn_w_size = batch * config.num_heads * seq * seq
  let blocks = Array::makei(config.num_layers, fn(_i) {
    TransformerBlockForwardWorkspace::{
      ln1_out: FixedArray::make(proj_size, Float::from_int(0)),
      q_proj: FixedArray::make(proj_size, Float::from_int(0)),
      k_proj: FixedArray::make(proj_size, Float::from_int(0)),
      v_proj: FixedArray::make(proj_size, Float::from_int(0)),
      attn_weights: FixedArray::make(attn_w_size, Float::from_int(0)),
      attn_concat: FixedArray::make(proj_size, Float::from_int(0)),
      attn_out: FixedArray::make(proj_size, Float::from_int(0)),
      x_after_res1: FixedArray::make(proj_size, Float::from_int(0)),
      ln2_out: FixedArray::make(proj_size, Float::from_int(0)),
      hidden_out: FixedArray::make(proj_size, Float::from_int(0)),
    }
  })
  TransformerForwardWorkspace::{
    blocks,
    ln_final_out: FixedArray::make(proj_size, Float::from_int(0)),
  }
}

///|
pub struct AdamwConfig {
  lr : Float
  beta1 : Float
  beta2 : Float
  eps : Float
  weight_decay : Float
} derive(Show, Eq)

///|
pub struct TransformerAdamwState {
  mut step : Int
  mut beta1_pow : Float
  mut beta2_pow : Float
  m : TransformerGrads
  v : TransformerGrads
}

///|
pub fn adamw_config_default(lr : Float) -> AdamwConfig {
  AdamwConfig::{
    lr,
    beta1: Float::from_double(0.9),
    beta2: Float::from_double(0.999),
    eps: Float::from_double(0.00000001),
    weight_decay: Float::from_double(0.01),
  }
}

///|
pub fn transformer_zero_adamw_state(
  config : TransformerConfig,
) -> TransformerAdamwState {
  TransformerAdamwState::{
    step: 0,
    beta1_pow: Float::from_int(1),
    beta2_pow: Float::from_int(1),
    m: transformer_zero_grads(config),
    v: transformer_zero_grads(config),
  }
}

///|
/// Initialize zero gradients matching the parameter shapes
pub fn transformer_zero_grads(config : TransformerConfig) -> TransformerGrads {
  let d = config.d_model
  let d_ff = config.d_ff
  let blocks = Array::makei(config.num_layers, fn(_i) {
    TransformerBlockGrads::{
      d_ln1_gamma: tensor_zeros(shape_new_unchecked([d])),
      d_ln1_beta: tensor_zeros(shape_new_unchecked([d])),
      d_w_q: tensor_zeros(shape_new_unchecked([d, d])),
      d_w_k: tensor_zeros(shape_new_unchecked([d, d])),
      d_w_v: tensor_zeros(shape_new_unchecked([d, d])),
      d_w_o: tensor_zeros(shape_new_unchecked([d, d])),
      d_ln2_gamma: tensor_zeros(shape_new_unchecked([d])),
      d_ln2_beta: tensor_zeros(shape_new_unchecked([d])),
      d_ff_w1: tensor_zeros(shape_new_unchecked([d, d_ff])),
      d_ff_b1: tensor_zeros(shape_new_unchecked([d_ff])),
      d_ff_w2: tensor_zeros(shape_new_unchecked([d_ff, d])),
      d_ff_b2: tensor_zeros(shape_new_unchecked([d])),
    }
  })
  TransformerGrads::{
    d_token_embedding: tensor_zeros(shape_new_unchecked([config.vocab_size, d])),
    d_pos_embedding: tensor_zeros(shape_new_unchecked([config.max_seq_len, d])),
    d_blocks: blocks,
    d_ln_final_gamma: tensor_zeros(shape_new_unchecked([d])),
    d_ln_final_beta: tensor_zeros(shape_new_unchecked([d])),
    d_lm_head: tensor_zeros(shape_new_unchecked([d, config.vocab_size])),
  }
}

///|
/// Layer normalization with cache: returns (output, mean, rstd)
/// mean and rstd have length = outer (number of vectors normalized)
pub fn layer_norm_with_cache(
  t : Tensor,
  gamma : Tensor,
  beta : Tensor,
  eps : Float,
) -> (Tensor, FixedArray[Float], FixedArray[Float]) {
  let result = FixedArray::make(t.numel(), Float::from_int(0))
  layer_norm_with_cache_to(t, gamma, beta, eps, result)
}

///|
fn layer_norm_with_cache_to(
  t : Tensor,
  gamma : Tensor,
  beta : Tensor,
  eps : Float,
  result : FixedArray[Float],
) -> (Tensor, FixedArray[Float], FixedArray[Float]) {
  let last_dim = t.dim(-1)
  let tc = t.contiguous()
  let gc = gamma.contiguous()
  let bc = beta.contiguous()
  let outer = t.numel() / last_dim
  let means = FixedArray::make(outer, Float::from_int(0))
  let rstds = FixedArray::make(outer, Float::from_int(0))
  native_layer_norm_fwd(
    tc.data,
    gc.data,
    bc.data,
    result,
    means,
    rstds,
    outer,
    last_dim,
    eps,
  )
  let out = Tensor::{
    data: result,
    shape: t.shape,
    offset: 0,
    strides: t.shape.strides(),
  }
  (out, means, rstds)
}

///|
fn add_residual_into(
  a : Tensor,
  b : Tensor,
  out_data : FixedArray[Float],
) -> Tensor {
  let ac = a.contiguous()
  let bc = b.contiguous()
  for i = 0; i < out_data.length(); i = i + 1 {
    out_data[i] = ac.data[i] + bc.data[i]
  }
  Tensor::{
    data: out_data,
    shape: a.shape,
    offset: 0,
    strides: a.shape.strides(),
  }
}

///|
/// Cross-entropy backward: returns d_logits [batch, vocab_size]
/// d_logits = softmax(logits) - one_hot(labels)
pub fn cross_entropy_backward(logits : Tensor, labels : Array[Int]) -> Tensor {
  let batch = logits.dim(0)
  let vocab = logits.dim(1)
  let lc = logits.contiguous()
  let result = FixedArray::make(batch * vocab, Float::from_int(0))
  // Copy logits, compute softmax inplace
  for i = 0; i < batch * vocab; i = i + 1 {
    result[i] = lc.data[i]
  }
  native_softmax_inplace(result, batch, vocab)
  for b = 0; b < batch; b = b + 1 {
    let base = b * vocab
    // subtract one-hot
    result[base + labels[b]] = result[base + labels[b]] - Float::from_int(1)
    // average over batch
    let inv_batch = Float::from_double(1.0 / batch.to_double())
    for v = 0; v < vocab; v = v + 1 {
      result[base + v] = result[base + v] * inv_batch
    }
  }
  Tensor::{
    data: result,
    shape: logits.shape,
    offset: 0,
    strides: logits.shape.strides(),
  }
}

///|
fn cross_entropy_forward_backward_fixed(
  logits : Tensor,
  labels : FixedArray[Int],
) -> (Float, Tensor) {
  let batch = logits.dim(0)
  let vocab = logits.dim(1)
  let lc = logits.contiguous()
  let d_logits_data = FixedArray::make(batch * vocab, Float::from_int(0))
  let loss = cross_entropy_fwd_bwd(lc.data, labels, d_logits_data, batch, vocab)
  (
    loss,
    Tensor::{
      data: d_logits_data,
      shape: logits.shape,
      offset: 0,
      strides: logits.shape.strides(),
    },
  )
}

///|
/// Cross-entropy forward + backward.
/// Returns `(loss, d_logits)` where `d_logits = (softmax - one_hot) / batch`.
pub fn cross_entropy_forward_backward(
  logits : Tensor,
  labels : Array[Int],
) -> (Float, Tensor) {
  let batch = logits.dim(0)
  let labels_fixed = FixedArray::make(batch, 0)
  for i = 0; i < batch; i = i + 1 {
    labels_fixed[i] = labels[i]
  }
  cross_entropy_forward_backward_fixed(logits, labels_fixed)
}

///|
/// Batched linear backward
/// dy: [batch, seq, out_dim], x_cached: [batch, seq, in_dim], w: [in_dim, out_dim]
/// Returns dx: [batch, seq, in_dim]
/// Accumulates into d_w: [in_dim, out_dim]
fn batched_linear_backward(
  dy : Tensor,
  x_cached : Tensor,
  w : Tensor,
  d_w : Tensor,
) -> Tensor {
  let batch = dy.dim(0)
  let seq = dy.dim(1)
  let out_dim = dy.dim(2)
  let in_dim = x_cached.dim(2)
  let n = batch * seq
  let dyc = dy.contiguous()
  let xc = x_cached.contiguous()
  let wc = w.contiguous()
  let dx_data = FixedArray::make(n * in_dim, Float::from_int(0))
  // dx = dy @ W^T : [n, out_dim] @ [out_dim, in_dim] -> [n, in_dim]
  native_sgemm(
    0,
    1,
    n,
    in_dim,
    out_dim,
    Float::from_int(1),
    dyc.data,
    out_dim,
    wc.data,
    out_dim,
    Float::from_int(0),
    dx_data,
    in_dim,
  )
  // dW += x^T @ dy : [in_dim, n] @ [n, out_dim] -> [in_dim, out_dim], beta=1.0
  native_sgemm(
    1,
    0,
    in_dim,
    out_dim,
    n,
    Float::from_int(1),
    xc.data,
    in_dim,
    dyc.data,
    out_dim,
    Float::from_int(1),
    d_w.data,
    out_dim,
  )
  Tensor::{
    data: dx_data,
    shape: x_cached.shape,
    offset: 0,
    strides: x_cached.shape.strides(),
  }
}

///|
/// Fused batched linear backward (single FFI call).
/// dy: [batch, seq, out_dim], x_cached: [batch, seq, in_dim], w: [in_dim, out_dim]
/// Returns dx: [batch, seq, in_dim]
/// Accumulates into d_w: [in_dim, out_dim]
fn batched_linear_backward_fused(
  dy : Tensor,
  x_cached : Tensor,
  w : Tensor,
  d_w : Tensor,
) -> Tensor {
  let batch = dy.dim(0)
  let seq = dy.dim(1)
  let out_dim = dy.dim(2)
  let in_dim = x_cached.dim(2)
  let n = batch * seq
  let dyc = dy.contiguous()
  let xc = x_cached.contiguous()
  let wc = w.contiguous()
  let dx_data = FixedArray::make(n * in_dim, Float::from_int(0))
  batched_linear_backward_raw(
    dyc.data,
    xc.data,
    wc.data,
    d_w.data,
    dx_data,
    n,
    in_dim,
    out_dim,
  )
  Tensor::{
    data: dx_data,
    shape: x_cached.shape,
    offset: 0,
    strides: x_cached.shape.strides(),
  }
}

///|
/// Add bias backward: accumulates into d_bias, returns dy unchanged
fn add_bias_backward(dy : Tensor, d_bias : Tensor) -> Unit {
  let dyc = dy.contiguous()
  let last_dim = dy.dim(-1)
  native_bias_add_bwd(dyc.data, d_bias.data, dyc.numel(), last_dim)
}

///|
/// Layer norm backward
/// dy: gradient from above, x: original input to layer norm
/// mean, rstd: cached from forward, gamma: LN parameter
/// Accumulates into d_gamma, d_beta. Returns dx.
pub fn layer_norm_backward(
  dy : Tensor,
  x : Tensor,
  mean : FixedArray[Float],
  rstd : FixedArray[Float],
  gamma : Tensor,
  d_gamma : Tensor,
  d_beta : Tensor,
) -> Tensor {
  let last_dim = x.dim(-1)
  let outer = x.numel() / last_dim
  let dyc = dy.contiguous()
  let xc = x.contiguous()
  let gc = gamma.contiguous()
  let dx_data = FixedArray::make(xc.numel(), Float::from_int(0))
  native_layer_norm_bwd(
    dyc.data,
    xc.data,
    mean,
    rstd,
    gc.data,
    dx_data,
    d_gamma.data,
    d_beta.data,
    outer,
    last_dim,
  )
  Tensor::{
    data: dx_data,
    shape: x.shape,
    offset: 0,
    strides: x.shape.strides(),
  }
}

///|
/// In-place variant of layer_norm_backward_add:
/// residual is overwritten with residual + dx_layer_norm and returned.
fn layer_norm_backward_add_inplace(
  dy : Tensor,
  residual : Tensor,
  x : Tensor,
  mean : FixedArray[Float],
  rstd : FixedArray[Float],
  gamma : Tensor,
  d_gamma : Tensor,
  d_beta : Tensor,
) -> Tensor {
  let last_dim = x.dim(-1)
  let outer = x.numel() / last_dim
  let dyc = dy.contiguous()
  let residual_c = residual.contiguous()
  let xc = x.contiguous()
  let gc = gamma.contiguous()
  native_layer_norm_bwd_add_inplace(
    dyc.data,
    residual_c.data,
    xc.data,
    mean,
    rstd,
    gc.data,
    d_gamma.data,
    d_beta.data,
    outer,
    last_dim,
  )
  Tensor::{
    data: residual_c.data,
    shape: x.shape,
    offset: 0,
    strides: x.shape.strides(),
  }
}

///|
/// GELU backward: dx = dy * gelu'(x)
/// gelu'(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * inner'
/// where inner = sqrt(2/pi) * (x + 0.044715 * x^3), inner' = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
pub fn gelu_backward(dy : Tensor, x_pre_gelu : Tensor) -> Tensor {
  let dyc = dy.contiguous()
  let xc = x_pre_gelu.contiguous()
  let result = FixedArray::make(xc.numel(), Float::from_int(0))
  native_gelu_backward(dyc.data, xc.data, result, xc.numel())
  Tensor::{
    data: result,
    shape: x_pre_gelu.shape,
    offset: 0,
    strides: x_pre_gelu.shape.strides(),
  }
}

///|
/// Embedding backward: scatter-add d_out into d_weight
/// d_out: [batch, seq, d_model], indices: [batch][seq]
fn embedding_backward(
  d_out : Tensor,
  indices : Array[Array[Int]],
  d_weight : Tensor,
) -> Unit {
  let batch = indices.length()
  let seq = indices[0].length()
  let d_model = d_out.dim(2)
  let dc = d_out.contiguous()
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < seq; s = s + 1 {
      let token_idx = indices[b][s]
      let src_base = b * seq * d_model + s * d_model
      let dst_base = token_idx * d_model
      for d = 0; d < d_model; d = d + 1 {
        d_weight.data[dst_base + d] = d_weight.data[dst_base + d] +
          dc.data[src_base + d]
      }
    }
  }
}

///|
/// Positional embedding backward: scatter-add into d_pos
/// Also returns dx = dy (since positional add is just addition)
fn positional_embedding_backward(
  dy : Tensor,
  d_pos : Tensor,
  seq : Int,
) -> Unit {
  let batch = dy.dim(0)
  let d_model = dy.dim(2)
  let dyc = dy.contiguous()
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < seq; s = s + 1 {
      let src_base = b * seq * d_model + s * d_model
      let dst_base = s * d_model
      for d = 0; d < d_model; d = d + 1 {
        d_pos.data[dst_base + d] = d_pos.data[dst_base + d] +
          dyc.data[src_base + d]
      }
    }
  }
}

///|
/// Feed-forward backward reference implementation (non-fused).
/// Kept for numerical regression checks in whitebox tests.
pub fn feed_forward_backward_reference(
  dy : Tensor,
  cache : TransformerBlockCache,
  params : TransformerBlockParams,
  grads : TransformerBlockGrads,
) -> Tensor {
  // Forward was: h = x @ W1 + b1 -> GELU -> h_act @ W2 + b2
  // Backward: d_b2, d(h_act @ W2), d_gelu, d_b1, d(x @ W1)

  // bias2 backward
  add_bias_backward(dy, grads.d_ff_b2)

  // W2 backward: dy is [batch, seq, d_model], ff_post_gelu is [batch, seq, d_ff]
  let d_gelu_out = batched_linear_backward(
    dy,
    cache.ff_post_gelu,
    params.ff_w2,
    grads.d_ff_w2,
  )

  // GELU backward
  let d_pre_gelu = gelu_backward(d_gelu_out, cache.ff_pre_gelu)

  // bias1 backward
  add_bias_backward(d_pre_gelu, grads.d_ff_b1)

  // W1 backward: d_pre_gelu is [batch, seq, d_ff], ln2_out is [batch, seq, d_model]
  let dx = batched_linear_backward(
    d_pre_gelu,
    cache.ln2_out,
    params.ff_w1,
    grads.d_ff_w1,
  )
  dx
}

///|
/// Feed-forward backward fused path:
///   dy -> (dW2, db2) -> GELU backward -> (dW1, db1, dx)
fn feed_forward_backward_fused(
  dy : Tensor,
  cache : TransformerBlockCache,
  params : TransformerBlockParams,
  grads : TransformerBlockGrads,
) -> Tensor {
  let batch = dy.dim(0)
  let seq = dy.dim(1)
  let d_model = dy.dim(2)
  let d_ff = cache.ff_pre_gelu.dim(2)
  let n = batch * seq
  let dyc = dy.contiguous()
  let post_c = cache.ff_post_gelu.contiguous()
  let pre_c = cache.ff_pre_gelu.contiguous()
  let ln2_c = cache.ln2_out.contiguous()
  let w2_c = params.ff_w2.contiguous()
  let w1_c = params.ff_w1.contiguous()
  let dx_data = FixedArray::make(n * d_model, Float::from_int(0))
  fused_ffn_backward(
    dyc.data,
    post_c.data,
    pre_c.data,
    ln2_c.data,
    w2_c.data,
    w1_c.data,
    grads.d_ff_w2.data,
    grads.d_ff_b2.data,
    grads.d_ff_w1.data,
    grads.d_ff_b1.data,
    dx_data,
    n,
    d_model,
    d_ff,
  )
  Tensor::{
    data: dx_data,
    shape: dy.shape,
    offset: 0,
    strides: dy.shape.strides(),
  }
}

///|
/// Feed-forward backward
/// dy: [batch, seq, d_model]
/// Returns dx: [batch, seq, d_model]
fn feed_forward_backward(
  dy : Tensor,
  cache : TransformerBlockCache,
  params : TransformerBlockParams,
  grads : TransformerBlockGrads,
) -> Tensor {
  feed_forward_backward_fused(dy, cache, params, grads)
}

///|
/// Multi-head attention backward
/// dy: [batch, seq, d_model]
/// Returns dx: [batch, seq, d_model]
fn multi_head_attention_backward(
  dy : Tensor,
  cache : TransformerBlockCache,
  params : TransformerBlockParams,
  config : TransformerConfig,
  grads : TransformerBlockGrads,
) -> Tensor {
  let batch = dy.dim(0)
  let seq = dy.dim(1)
  let d_model = config.d_model
  let num_heads = config.num_heads
  let d_k = d_model / num_heads

  // W_o backward: dy @ W_o^T -> d_concat
  let d_concat = batched_linear_backward_fused(
    dy,
    cache.attn_out_pre_wo,
    params.w_o,
    grads.d_w_o,
  )

  // d_concat is [batch, seq, d_model]
  // Run attention backward directly on [batch, seq, d_model] interleaved layout.
  // This removes reshape_for_heads/reshape_from_heads copies.
  let scale = Float::from_double(1.0 / d_k.to_double().sqrt())
  let proj_size = batch * seq * d_model
  let dq_proj_data = FixedArray::make(proj_size, Float::from_int(0))
  let dk_proj_data = FixedArray::make(proj_size, Float::from_int(0))
  let dv_proj_data = FixedArray::make(proj_size, Float::from_int(0))
  let dc = d_concat.contiguous()
  let qc = cache.q_proj.contiguous()
  let kc = cache.k_proj.contiguous()
  let vc = cache.v_proj.contiguous()
  attention_backward_batch_interleaved(
    dc.data,
    qc.data,
    kc.data,
    vc.data,
    cache.attn_weights,
    dq_proj_data,
    dk_proj_data,
    dv_proj_data,
    batch,
    num_heads,
    seq,
    d_k,
    scale,
  )
  let proj_shape = shape_new_unchecked([batch, seq, d_model])
  let dq_proj = Tensor::{
    data: dq_proj_data,
    shape: proj_shape,
    offset: 0,
    strides: proj_shape.strides(),
  }
  let dk_proj = Tensor::{
    data: dk_proj_data,
    shape: proj_shape,
    offset: 0,
    strides: proj_shape.strides(),
  }
  let dv_proj = Tensor::{
    data: dv_proj_data,
    shape: proj_shape,
    offset: 0,
    strides: proj_shape.strides(),
  }

  // W_q, W_k, W_v backward (self-attention: input is the same for Q, K, V)
  let dx_q = batched_linear_backward_fused(
    dq_proj,
    cache.ln1_out,
    params.w_q,
    grads.d_w_q,
  )
  let dx_k = batched_linear_backward_fused(
    dk_proj,
    cache.ln1_out,
    params.w_k,
    grads.d_w_k,
  )
  let dx_v = batched_linear_backward_fused(
    dv_proj,
    cache.ln1_out,
    params.w_v,
    grads.d_w_v,
  )

  // Reuse dx_q buffer and accumulate the remaining two contributions.
  let n = batch * seq * d_model
  accumulate_inplace(dx_q.data, dx_k.data, n)
  accumulate_inplace(dx_q.data, dx_v.data, n)
  dx_q
}

///|
/// Single transformer block backward
/// dy: [batch, seq, d_model]
/// Returns dx: [batch, seq, d_model]
fn transformer_block_backward(
  dy : Tensor,
  cache : TransformerBlockCache,
  params : TransformerBlockParams,
  config : TransformerConfig,
  grads : TransformerBlockGrads,
) -> Tensor {
  // Forward: x -> LN1 -> MHA -> +residual1 -> LN2 -> FFN -> +residual2
  // Backward:
  // dy comes from after residual2
  // residual2: dy goes to both FFN path and x_after_res1

  // FFN backward
  let d_ln2 = feed_forward_backward(dy, cache, params, grads)

  // LN2 backward
  let d_res1 = layer_norm_backward_add_inplace(
    d_ln2,
    dy,
    cache.x_after_res1,
    cache.ln2_mean,
    cache.ln2_rstd,
    params.ln2_gamma,
    grads.d_ln2_gamma,
    grads.d_ln2_beta,
  )

  // MHA backward
  let d_ln1 = multi_head_attention_backward(
    d_res1, cache, params, config, grads,
  )

  // LN1 backward + residual1 add
  layer_norm_backward_add_inplace(
    d_ln1,
    d_res1,
    cache.x_input,
    cache.ln1_mean,
    cache.ln1_rstd,
    params.ln1_gamma,
    grads.d_ln1_gamma,
    grads.d_ln1_beta,
  )
}

///|
/// Forward pass with cache for training
fn transformer_forward_with_cache_impl(
  tokens : Array[Array[Int]],
  params : TransformerParams,
  config : TransformerConfig,
  mask : Tensor?,
  workspace : TransformerForwardWorkspace?,
) -> (Tensor, TransformerCache) {
  let batch = tokens.length()
  let seq = tokens[0].length()
  let d_model = config.d_model
  let num_heads = config.num_heads
  let d_k = d_model / num_heads
  let scale = Float::from_double(1.0 / d_k.to_double().sqrt())
  let proj_shape = shape_new_unchecked([batch, seq, d_model])
  let mc = match mask {
    None => None
    Some(m) => Some(m.contiguous())
  }
  let proj_size = batch * seq * d_model
  let attn_w_size = batch * num_heads * seq * seq

  // Token embedding
  let tok_emb = embedding(tokens, params.token_embedding)
  // Add positional embedding
  let x_with_pos = add_positional_embedding(tok_emb, params.pos_embedding)
  let mut hidden = x_with_pos
  let block_caches : Array[TransformerBlockCache] = []
  for layer = 0; layer < config.num_layers; layer = layer + 1 {
    let ws_block = match workspace {
      None => None
      Some(ws) => Some(ws.blocks[layer])
    }
    let bp = params.blocks[layer]
    let x_input = hidden
    let (
      ln1_out_data,
      q_proj_data,
      k_proj_data,
      v_proj_data,
      attn_weights_all,
      attn_concat_data,
      attn_out_data,
      x_after_res1_data,
      ln2_out_data,
      hidden_data,
    ) = match ws_block {
      Some(ws) =>
        (
          ws.ln1_out,
          ws.q_proj,
          ws.k_proj,
          ws.v_proj,
          ws.attn_weights,
          ws.attn_concat,
          ws.attn_out,
          ws.x_after_res1,
          ws.ln2_out,
          ws.hidden_out,
        )
      None =>
        (
          FixedArray::make(proj_size, Float::from_int(0)),
          FixedArray::make(proj_size, Float::from_int(0)),
          FixedArray::make(proj_size, Float::from_int(0)),
          FixedArray::make(proj_size, Float::from_int(0)),
          FixedArray::make(attn_w_size, Float::from_int(0)),
          FixedArray::make(proj_size, Float::from_int(0)),
          FixedArray::make(proj_size, Float::from_int(0)),
          FixedArray::make(proj_size, Float::from_int(0)),
          FixedArray::make(proj_size, Float::from_int(0)),
          FixedArray::make(proj_size, Float::from_int(0)),
        )
    }

    // LN1 with cache
    let (ln1_out, ln1_mean, ln1_rstd) = layer_norm_with_cache_to(
      x_input,
      bp.ln1_gamma,
      bp.ln1_beta,
      config.eps,
      ln1_out_data,
    )

    // MHA block fused forward (QKV projection + attention + W_o)
    let ln1c = ln1_out.contiguous()
    let wq = bp.w_q.contiguous()
    let wk = bp.w_k.contiguous()
    let wv = bp.w_v.contiguous()
    let wo = bp.w_o.contiguous()
    match mc {
      Some(m) =>
        mha_forward_batch_interleaved_masked(
          ln1c.data,
          wq.data,
          wk.data,
          wv.data,
          wo.data,
          m.data,
          q_proj_data,
          k_proj_data,
          v_proj_data,
          attn_weights_all,
          attn_concat_data,
          attn_out_data,
          batch,
          seq,
          d_model,
          num_heads,
          scale,
        )
      None =>
        mha_forward_batch_interleaved(
          ln1c.data,
          wq.data,
          wk.data,
          wv.data,
          wo.data,
          q_proj_data,
          k_proj_data,
          v_proj_data,
          attn_weights_all,
          attn_concat_data,
          attn_out_data,
          batch,
          seq,
          d_model,
          num_heads,
          scale,
        )
    }
    let q_proj = Tensor::{
      data: q_proj_data,
      shape: proj_shape,
      offset: 0,
      strides: proj_shape.strides(),
    }
    let k_proj = Tensor::{
      data: k_proj_data,
      shape: proj_shape,
      offset: 0,
      strides: proj_shape.strides(),
    }
    let v_proj = Tensor::{
      data: v_proj_data,
      shape: proj_shape,
      offset: 0,
      strides: proj_shape.strides(),
    }
    let attn_concat = Tensor::{
      data: attn_concat_data,
      shape: proj_shape,
      offset: 0,
      strides: proj_shape.strides(),
    }
    let attn_out = Tensor::{
      data: attn_out_data,
      shape: proj_shape,
      offset: 0,
      strides: proj_shape.strides(),
    }

    // Residual 1
    let x_after_res1 = add_residual_into(x_input, attn_out, x_after_res1_data)

    // LN2 with cache
    let (ln2_out, ln2_mean, ln2_rstd) = layer_norm_with_cache_to(
      x_after_res1,
      bp.ln2_gamma,
      bp.ln2_beta,
      config.eps,
      ln2_out_data,
    )

    // FFN with cache
    let ff_h = batched_linear(ln2_out, bp.ff_w1).unwrap()
    let ff_pre_gelu = add_bias(ff_h, bp.ff_b1)
    let ff_post_gelu = tensor_gelu(ff_pre_gelu)
    let ff_out_h = batched_linear(ff_post_gelu, bp.ff_w2).unwrap()
    let ff_out = add_bias(ff_out_h, bp.ff_b2)

    // Residual 2
    hidden = add_residual_into(x_after_res1, ff_out, hidden_data)
    block_caches.push(TransformerBlockCache::{
      x_input,
      ln1_out,
      ln1_mean,
      ln1_rstd,
      q_proj,
      k_proj,
      v_proj,
      attn_weights: attn_weights_all,
      attn_out_pre_wo: attn_concat,
      x_after_res1,
      ln2_out,
      ln2_mean,
      ln2_rstd,
      ff_pre_gelu,
      ff_post_gelu,
    })
  }

  // Final layer norm
  let hidden_before_final_ln = hidden
  let (ln_final_out, ln_final_mean, ln_final_rstd) = match workspace {
    Some(ws) =>
      layer_norm_with_cache_to(
        hidden_before_final_ln,
        params.ln_final_gamma,
        params.ln_final_beta,
        config.eps,
        ws.ln_final_out,
      )
    None =>
      layer_norm_with_cache(
        hidden_before_final_ln,
        params.ln_final_gamma,
        params.ln_final_beta,
        config.eps,
      )
  }

  // LM head
  let logits = batched_linear(ln_final_out, params.lm_head).unwrap()
  let cache = TransformerCache::{
    tokens,
    x_with_pos,
    block_caches,
    hidden_before_final_ln,
    ln_final_mean,
    ln_final_rstd,
    ln_final_out,
    logits,
  }
  (logits, cache)
}

///|
fn transformer_forward_with_cache_workspace(
  tokens : Array[Array[Int]],
  params : TransformerParams,
  config : TransformerConfig,
  mask : Tensor?,
  workspace : TransformerForwardWorkspace,
) -> (Tensor, TransformerCache) {
  transformer_forward_with_cache_impl(
    tokens,
    params,
    config,
    mask,
    Some(workspace),
  )
}

///|
/// Forward pass with cache for training
pub fn transformer_forward_with_cache(
  tokens : Array[Array[Int]],
  params : TransformerParams,
  config : TransformerConfig,
  mask : Tensor?,
) -> (Tensor, TransformerCache) {
  transformer_forward_with_cache_impl(tokens, params, config, mask, None)
}

///|
fn transformer_backward_from_d_logits(
  cache : TransformerCache,
  params : TransformerParams,
  config : TransformerConfig,
  grads : TransformerGrads,
  d_logits : Tensor,
) -> Unit {
  let seq = cache.tokens[0].length()
  // LM head backward: logits = ln_final_out @ lm_head
  let d_ln_final_out = batched_linear_backward_fused(
    d_logits,
    cache.ln_final_out,
    params.lm_head,
    grads.d_lm_head,
  )

  // Final LN backward
  let d_hidden = layer_norm_backward(
    d_ln_final_out,
    cache.hidden_before_final_ln,
    cache.ln_final_mean,
    cache.ln_final_rstd,
    params.ln_final_gamma,
    grads.d_ln_final_gamma,
    grads.d_ln_final_beta,
  )

  // Blocks backward (reverse order)
  let mut dy = d_hidden
  for i = config.num_layers - 1; i >= 0; i = i - 1 {
    dy = transformer_block_backward(
      dy,
      cache.block_caches[i],
      params.blocks[i],
      config,
      grads.d_blocks[i],
    )
  }

  // Positional embedding backward
  positional_embedding_backward(dy, grads.d_pos_embedding, seq)

  // Token embedding backward
  embedding_backward(dy, cache.tokens, grads.d_token_embedding)
}

///|
/// Full transformer backward pass
/// Returns the loss value
pub fn transformer_backward(
  cache : TransformerCache,
  params : TransformerParams,
  config : TransformerConfig,
  grads : TransformerGrads,
  labels : Array[Int],
) -> Float {
  let batch = cache.tokens.length()
  let seq = cache.tokens[0].length()

  // Compute loss
  // Extract last-position logits for loss: [batch, vocab_size]
  let vocab = config.vocab_size
  let last_logits_data = FixedArray::make(batch * vocab, Float::from_int(0))
  let logits_c = cache.logits.contiguous()
  for b = 0; b < batch; b = b + 1 {
    for v = 0; v < vocab; v = v + 1 {
      last_logits_data[b * vocab + v] = logits_c.data[b * seq * vocab +
        (seq - 1) * vocab +
        v]
    }
  }
  let last_logits = Tensor::{
    data: last_logits_data,
    shape: shape_new_unchecked([batch, vocab]),
    offset: 0,
    strides: shape_new_unchecked([batch, vocab]).strides(),
  }

  // Cross-entropy forward + backward on last position only
  let (loss, d_last_logits) = cross_entropy_forward_backward(
    last_logits, labels,
  )

  // Expand to full sequence gradient: [batch, seq, vocab]
  let d_logits_data = FixedArray::make(batch * seq * vocab, Float::from_int(0))
  for b = 0; b < batch; b = b + 1 {
    for v = 0; v < vocab; v = v + 1 {
      d_logits_data[b * seq * vocab + (seq - 1) * vocab + v] = d_last_logits.data[b *
        vocab +
        v]
    }
  }
  let d_logits = Tensor::{
    data: d_logits_data,
    shape: cache.logits.shape,
    offset: 0,
    strides: cache.logits.shape.strides(),
  }
  transformer_backward_from_d_logits(cache, params, config, grads, d_logits)
  loss
}

///|
/// Full transformer backward for causal LM objective over all positions.
/// labels: [batch, seq], where labels[b][s] is the target token for logits[b, s, :]
pub fn transformer_backward_lm(
  cache : TransformerCache,
  params : TransformerParams,
  config : TransformerConfig,
  grads : TransformerGrads,
  labels : Array[Array[Int]],
) -> Float {
  let batch = cache.tokens.length()
  let seq = cache.tokens[0].length()
  let vocab = config.vocab_size
  let flat_labels = FixedArray::make(batch * seq, 0)
  let mut idx = 0
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < seq; s = s + 1 {
      flat_labels[idx] = labels[b][s]
      idx = idx + 1
    }
  }
  let logits_c = cache.logits.contiguous()
  let flat_shape = shape_new_unchecked([batch * seq, vocab])
  let flat_logits = Tensor::{
    data: logits_c.data,
    shape: flat_shape,
    offset: 0,
    strides: flat_shape.strides(),
  }
  let (loss, d_flat_logits) = cross_entropy_forward_backward_fixed(
    flat_logits, flat_labels,
  )
  let d_logits = Tensor::{
    data: d_flat_logits.data,
    shape: cache.logits.shape,
    offset: 0,
    strides: cache.logits.shape.strides(),
  }
  transformer_backward_from_d_logits(cache, params, config, grads, d_logits)
  loss
}

///|
/// SGD update: param -= lr * grad
fn sgd_update_tensor(param : Tensor, grad : Tensor, lr : Float) -> Unit {
  for i = 0; i < param.data.length(); i = i + 1 {
    param.data[i] = param.data[i] - lr * grad.data[i]
  }
}

///|
/// Apply SGD to all parameters
fn sgd_update(
  params : TransformerParams,
  grads : TransformerGrads,
  lr : Float,
) -> Unit {
  sgd_update_tensor(params.token_embedding, grads.d_token_embedding, lr)
  sgd_update_tensor(params.pos_embedding, grads.d_pos_embedding, lr)
  for i = 0; i < params.blocks.length(); i = i + 1 {
    let p = params.blocks[i]
    let g = grads.d_blocks[i]
    sgd_update_tensor(p.ln1_gamma, g.d_ln1_gamma, lr)
    sgd_update_tensor(p.ln1_beta, g.d_ln1_beta, lr)
    sgd_update_tensor(p.w_q, g.d_w_q, lr)
    sgd_update_tensor(p.w_k, g.d_w_k, lr)
    sgd_update_tensor(p.w_v, g.d_w_v, lr)
    sgd_update_tensor(p.w_o, g.d_w_o, lr)
    sgd_update_tensor(p.ln2_gamma, g.d_ln2_gamma, lr)
    sgd_update_tensor(p.ln2_beta, g.d_ln2_beta, lr)
    sgd_update_tensor(p.ff_w1, g.d_ff_w1, lr)
    sgd_update_tensor(p.ff_b1, g.d_ff_b1, lr)
    sgd_update_tensor(p.ff_w2, g.d_ff_w2, lr)
    sgd_update_tensor(p.ff_b2, g.d_ff_b2, lr)
  }
  sgd_update_tensor(params.ln_final_gamma, grads.d_ln_final_gamma, lr)
  sgd_update_tensor(params.ln_final_beta, grads.d_ln_final_beta, lr)
  sgd_update_tensor(params.lm_head, grads.d_lm_head, lr)
}

///|
fn adamw_update_tensor(
  param : Tensor,
  grad : Tensor,
  m : Tensor,
  v : Tensor,
  cfg : AdamwConfig,
  bias_c1 : Float,
  bias_c2 : Float,
) -> Unit {
  adamw_step_inplace(
    param.data,
    grad.data,
    m.data,
    v.data,
    param.data.length(),
    cfg.lr,
    cfg.beta1,
    cfg.beta2,
    cfg.eps,
    cfg.weight_decay,
    bias_c1,
    bias_c2,
  )
}

///|
fn adamw_update(
  params : TransformerParams,
  grads : TransformerGrads,
  state : TransformerAdamwState,
  cfg : AdamwConfig,
) -> Unit {
  let one = Float::from_int(1)
  state.step = state.step + 1
  state.beta1_pow = state.beta1_pow * cfg.beta1
  state.beta2_pow = state.beta2_pow * cfg.beta2
  let bias_c1 = one - state.beta1_pow
  let bias_c2 = one - state.beta2_pow
  let m = state.m
  let v = state.v
  adamw_update_tensor(
    params.token_embedding,
    grads.d_token_embedding,
    m.d_token_embedding,
    v.d_token_embedding,
    cfg,
    bias_c1,
    bias_c2,
  )
  adamw_update_tensor(
    params.pos_embedding,
    grads.d_pos_embedding,
    m.d_pos_embedding,
    v.d_pos_embedding,
    cfg,
    bias_c1,
    bias_c2,
  )
  for i = 0; i < params.blocks.length(); i = i + 1 {
    let p = params.blocks[i]
    let g = grads.d_blocks[i]
    let m_b = m.d_blocks[i]
    let v_b = v.d_blocks[i]
    adamw_update_tensor(
      p.ln1_gamma,
      g.d_ln1_gamma,
      m_b.d_ln1_gamma,
      v_b.d_ln1_gamma,
      cfg,
      bias_c1,
      bias_c2,
    )
    adamw_update_tensor(
      p.ln1_beta,
      g.d_ln1_beta,
      m_b.d_ln1_beta,
      v_b.d_ln1_beta,
      cfg,
      bias_c1,
      bias_c2,
    )
    adamw_update_tensor(
      p.w_q,
      g.d_w_q,
      m_b.d_w_q,
      v_b.d_w_q,
      cfg,
      bias_c1,
      bias_c2,
    )
    adamw_update_tensor(
      p.w_k,
      g.d_w_k,
      m_b.d_w_k,
      v_b.d_w_k,
      cfg,
      bias_c1,
      bias_c2,
    )
    adamw_update_tensor(
      p.w_v,
      g.d_w_v,
      m_b.d_w_v,
      v_b.d_w_v,
      cfg,
      bias_c1,
      bias_c2,
    )
    adamw_update_tensor(
      p.w_o,
      g.d_w_o,
      m_b.d_w_o,
      v_b.d_w_o,
      cfg,
      bias_c1,
      bias_c2,
    )
    adamw_update_tensor(
      p.ln2_gamma,
      g.d_ln2_gamma,
      m_b.d_ln2_gamma,
      v_b.d_ln2_gamma,
      cfg,
      bias_c1,
      bias_c2,
    )
    adamw_update_tensor(
      p.ln2_beta,
      g.d_ln2_beta,
      m_b.d_ln2_beta,
      v_b.d_ln2_beta,
      cfg,
      bias_c1,
      bias_c2,
    )
    adamw_update_tensor(
      p.ff_w1,
      g.d_ff_w1,
      m_b.d_ff_w1,
      v_b.d_ff_w1,
      cfg,
      bias_c1,
      bias_c2,
    )
    adamw_update_tensor(
      p.ff_b1,
      g.d_ff_b1,
      m_b.d_ff_b1,
      v_b.d_ff_b1,
      cfg,
      bias_c1,
      bias_c2,
    )
    adamw_update_tensor(
      p.ff_w2,
      g.d_ff_w2,
      m_b.d_ff_w2,
      v_b.d_ff_w2,
      cfg,
      bias_c1,
      bias_c2,
    )
    adamw_update_tensor(
      p.ff_b2,
      g.d_ff_b2,
      m_b.d_ff_b2,
      v_b.d_ff_b2,
      cfg,
      bias_c1,
      bias_c2,
    )
  }
  adamw_update_tensor(
    params.ln_final_gamma,
    grads.d_ln_final_gamma,
    m.d_ln_final_gamma,
    v.d_ln_final_gamma,
    cfg,
    bias_c1,
    bias_c2,
  )
  adamw_update_tensor(
    params.ln_final_beta,
    grads.d_ln_final_beta,
    m.d_ln_final_beta,
    v.d_ln_final_beta,
    cfg,
    bias_c1,
    bias_c2,
  )
  adamw_update_tensor(
    params.lm_head,
    grads.d_lm_head,
    m.d_lm_head,
    v.d_lm_head,
    cfg,
    bias_c1,
    bias_c2,
  )
}

///|
/// Zero all gradients
fn zero_grads(grads : TransformerGrads) -> Unit {
  zero_tensor(grads.d_token_embedding)
  zero_tensor(grads.d_pos_embedding)
  for i = 0; i < grads.d_blocks.length(); i = i + 1 {
    let g = grads.d_blocks[i]
    zero_tensor(g.d_ln1_gamma)
    zero_tensor(g.d_ln1_beta)
    zero_tensor(g.d_w_q)
    zero_tensor(g.d_w_k)
    zero_tensor(g.d_w_v)
    zero_tensor(g.d_w_o)
    zero_tensor(g.d_ln2_gamma)
    zero_tensor(g.d_ln2_beta)
    zero_tensor(g.d_ff_w1)
    zero_tensor(g.d_ff_b1)
    zero_tensor(g.d_ff_w2)
    zero_tensor(g.d_ff_b2)
  }
  zero_tensor(grads.d_ln_final_gamma)
  zero_tensor(grads.d_ln_final_beta)
  zero_tensor(grads.d_lm_head)
}

///|
fn zero_tensor(t : Tensor) -> Unit {
  zero_inplace(t.data, t.data.length())
}

///|
pub struct TransformerLmCheckpoint {
  params : TransformerParams
  adamw_state : TransformerAdamwState?
  global_step : Int
}

///|
fn build_lm_windows(
  all_ids : Array[Int],
  seq_len : Int,
) -> (Array[Array[Int]], Array[Array[Int]]) {
  let inputs : Array[Array[Int]] = []
  let targets : Array[Array[Int]] = []
  for i = 0; i + seq_len < all_ids.length(); i = i + 1 {
    let input_seq : Array[Int] = []
    let target_seq : Array[Int] = []
    for j = 0; j < seq_len; j = j + 1 {
      input_seq.push(all_ids[i + j])
      target_seq.push(all_ids[i + j + 1])
    }
    inputs.push(input_seq)
    targets.push(target_seq)
  }
  (inputs, targets)
}

///|
fn effective_batch_size(batch_size : Int, windows : Int) -> Int {
  if windows <= 0 {
    0
  } else if batch_size <= 0 {
    1
  } else if batch_size > windows {
    windows
  } else {
    batch_size
  }
}

///|
fn train_lm_profile_steps_from_windows_inplace(
  inputs : Array[Array[Int]],
  targets : Array[Array[Int]],
  params : TransformerParams,
  grads : TransformerGrads,
  adamw_state : TransformerAdamwState,
  config : TransformerConfig,
  warmup_steps : Int,
  steps : Int,
  batch_size : Int,
  lr : Float,
  step_offset : Int,
  use_adamw : Bool,
) -> Array[TransformerLmStepMetric] {
  let metrics : Array[TransformerLmStepMetric] = []
  if inputs.length() == 0 || steps <= 0 {
    return metrics
  }
  let effective_warmup = if warmup_steps < 0 { 0 } else { warmup_steps }
  let effective_batch = effective_batch_size(batch_size, inputs.length())
  let seq_len = config.max_seq_len
  let mask = causal_mask(seq_len).unwrap()
  let forward_workspace = transformer_forward_workspace(
    config, effective_batch, seq_len,
  )
  let total_steps = effective_warmup + steps
  let offset = if step_offset < 0 { 0 } else { step_offset }
  let adamw_cfg = adamw_config_default(lr)
  for step = 0; step < total_steps; step = step + 1 {
    zero_grads(grads)
    let batch_inputs : Array[Array[Int]] = []
    let batch_targets : Array[Array[Int]] = []
    let logical_step = offset + step
    let base = logical_step * effective_batch % inputs.length()
    for i = 0; i < effective_batch; i = i + 1 {
      let idx = (base + i) % inputs.length()
      batch_inputs.push(inputs[idx])
      batch_targets.push(targets[idx])
    }
    let t0 = clock_ns()
    let (_, cache) = transformer_forward_with_cache_workspace(
      batch_inputs,
      params,
      config,
      Some(mask),
      forward_workspace,
    )
    let loss = transformer_backward_lm(
      cache, params, config, grads, batch_targets,
    )
    if use_adamw {
      adamw_update(params, grads, adamw_state, adamw_cfg)
    } else {
      sgd_update(params, grads, lr)
    }
    let t1 = clock_ns()
    if step >= effective_warmup {
      metrics.push(TransformerLmStepMetric::{
        step_ns: t1 - t0,
        loss,
        perplexity: @math.expf(loss),
      })
    }
  }
  metrics
}

///|
fn flatten_lm_labels(
  labels : Array[Array[Int]],
  seq_len : Int,
) -> FixedArray[Int] {
  let batch = labels.length()
  let out = FixedArray::make(batch * seq_len, 0)
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < seq_len; s = s + 1 {
      out[b * seq_len + s] = labels[b][s]
    }
  }
  out
}

///|
fn append_ascii_bytes(out : Array[Byte], text : String) -> Unit {
  for c in text {
    out.push(c.to_int().to_byte())
  }
}

///|
fn append_u32_le(out : Array[Byte], value : Int) -> Unit {
  let v = if value < 0 { 0 } else { value }
  out.push((v % 256).to_byte())
  out.push((v / 256 % 256).to_byte())
  out.push((v / 65536 % 256).to_byte())
  out.push((v / 16777216 % 256).to_byte())
}

///|
fn read_u32_le(bytes : Bytes, offset : Int) -> Result[(Int, Int), String] {
  if offset < 0 || offset + 4 > bytes.length() {
    return Err("checkpoint truncated while reading u32")
  }
  let v = bytes[offset].to_int() +
    bytes[offset + 1].to_int() * 256 +
    bytes[offset + 2].to_int() * 65536 +
    bytes[offset + 3].to_int() * 16777216
  Ok((v, offset + 4))
}

///|
fn append_f32_le(out : Array[Byte], value : Float) -> Unit {
  let b = value.to_le_bytes()
  for i = 0; i < b.length(); i = i + 1 {
    out.push(b[i])
  }
}

///|
fn read_f32_le(bytes : Bytes, offset : Int) -> Result[(Float, Int), String] {
  if offset < 0 || offset + 4 > bytes.length() {
    return Err("checkpoint truncated while reading f32")
  }
  let bits = bytes[offset].to_int() +
    bytes[offset + 1].to_int() * 256 +
    bytes[offset + 2].to_int() * 65536 +
    bytes[offset + 3].to_int() * 16777216
  Ok((Float::reinterpret_from_int(bits), offset + 4))
}

///|
fn append_tensor_f32(out : Array[Byte], t : Tensor) -> Unit {
  for i = 0; i < t.data.length(); i = i + 1 {
    append_f32_le(out, t.data[i])
  }
}

///|
fn read_tensor_f32(
  bytes : Bytes,
  offset : Int,
  t : Tensor,
) -> Result[Int, String] {
  let mut off = offset
  for i = 0; i < t.data.length(); i = i + 1 {
    let r = read_f32_le(bytes, off)
    let (v, next_off) = match r {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    t.data[i] = v
    off = next_off
  }
  Ok(off)
}

///|
fn append_transformer_params_f32(
  out : Array[Byte],
  params : TransformerParams,
) -> Unit {
  append_tensor_f32(out, params.token_embedding)
  append_tensor_f32(out, params.pos_embedding)
  for bi = 0; bi < params.blocks.length(); bi = bi + 1 {
    let b = params.blocks[bi]
    append_tensor_f32(out, b.ln1_gamma)
    append_tensor_f32(out, b.ln1_beta)
    append_tensor_f32(out, b.w_q)
    append_tensor_f32(out, b.w_k)
    append_tensor_f32(out, b.w_v)
    append_tensor_f32(out, b.w_o)
    append_tensor_f32(out, b.ln2_gamma)
    append_tensor_f32(out, b.ln2_beta)
    append_tensor_f32(out, b.ff_w1)
    append_tensor_f32(out, b.ff_b1)
    append_tensor_f32(out, b.ff_w2)
    append_tensor_f32(out, b.ff_b2)
  }
  append_tensor_f32(out, params.ln_final_gamma)
  append_tensor_f32(out, params.ln_final_beta)
  append_tensor_f32(out, params.lm_head)
}

///|
fn read_transformer_params_f32(
  bytes : Bytes,
  offset : Int,
  params : TransformerParams,
) -> Result[Int, String] {
  let mut off = offset
  off = match read_tensor_f32(bytes, off, params.token_embedding) {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  off = match read_tensor_f32(bytes, off, params.pos_embedding) {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  for bi = 0; bi < params.blocks.length(); bi = bi + 1 {
    let b = params.blocks[bi]
    off = match read_tensor_f32(bytes, off, b.ln1_gamma) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.ln1_beta) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.w_q) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.w_k) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.w_v) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.w_o) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.ln2_gamma) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.ln2_beta) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.ff_w1) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.ff_b1) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.ff_w2) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.ff_b2) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
  }
  off = match read_tensor_f32(bytes, off, params.ln_final_gamma) {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  off = match read_tensor_f32(bytes, off, params.ln_final_beta) {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  read_tensor_f32(bytes, off, params.lm_head)
}

///|
fn append_transformer_grads_f32(
  out : Array[Byte],
  grads : TransformerGrads,
) -> Unit {
  append_tensor_f32(out, grads.d_token_embedding)
  append_tensor_f32(out, grads.d_pos_embedding)
  for bi = 0; bi < grads.d_blocks.length(); bi = bi + 1 {
    let b = grads.d_blocks[bi]
    append_tensor_f32(out, b.d_ln1_gamma)
    append_tensor_f32(out, b.d_ln1_beta)
    append_tensor_f32(out, b.d_w_q)
    append_tensor_f32(out, b.d_w_k)
    append_tensor_f32(out, b.d_w_v)
    append_tensor_f32(out, b.d_w_o)
    append_tensor_f32(out, b.d_ln2_gamma)
    append_tensor_f32(out, b.d_ln2_beta)
    append_tensor_f32(out, b.d_ff_w1)
    append_tensor_f32(out, b.d_ff_b1)
    append_tensor_f32(out, b.d_ff_w2)
    append_tensor_f32(out, b.d_ff_b2)
  }
  append_tensor_f32(out, grads.d_ln_final_gamma)
  append_tensor_f32(out, grads.d_ln_final_beta)
  append_tensor_f32(out, grads.d_lm_head)
}

///|
fn read_transformer_grads_f32(
  bytes : Bytes,
  offset : Int,
  grads : TransformerGrads,
) -> Result[Int, String] {
  let mut off = offset
  off = match read_tensor_f32(bytes, off, grads.d_token_embedding) {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  off = match read_tensor_f32(bytes, off, grads.d_pos_embedding) {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  for bi = 0; bi < grads.d_blocks.length(); bi = bi + 1 {
    let b = grads.d_blocks[bi]
    off = match read_tensor_f32(bytes, off, b.d_ln1_gamma) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.d_ln1_beta) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.d_w_q) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.d_w_k) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.d_w_v) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.d_w_o) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.d_ln2_gamma) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.d_ln2_beta) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.d_ff_w1) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.d_ff_b1) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.d_ff_w2) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    off = match read_tensor_f32(bytes, off, b.d_ff_b2) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
  }
  off = match read_tensor_f32(bytes, off, grads.d_ln_final_gamma) {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  off = match read_tensor_f32(bytes, off, grads.d_ln_final_beta) {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  read_tensor_f32(bytes, off, grads.d_lm_head)
}

///|
pub fn transformer_train_lm_profile_steps_ids_inplace(
  all_ids : Array[Int],
  params : TransformerParams,
  grads : TransformerGrads,
  config : TransformerConfig,
  warmup_steps : Int,
  steps : Int,
  batch_size : Int,
  lr : Float,
  step_offset : Int,
) -> Array[TransformerLmStepMetric] {
  let seq_len = config.max_seq_len
  let (inputs, targets) = build_lm_windows(all_ids, seq_len)
  let dummy_adamw_state = transformer_zero_adamw_state(config)
  train_lm_profile_steps_from_windows_inplace(
    inputs, targets, params, grads, dummy_adamw_state, config, warmup_steps, steps,
    batch_size, lr, step_offset, false,
  )
}

///|
pub fn transformer_train_lm_profile_steps_ids_inplace_adamw(
  all_ids : Array[Int],
  params : TransformerParams,
  grads : TransformerGrads,
  adamw_state : TransformerAdamwState,
  config : TransformerConfig,
  warmup_steps : Int,
  steps : Int,
  batch_size : Int,
  lr : Float,
  step_offset : Int,
) -> Array[TransformerLmStepMetric] {
  let seq_len = config.max_seq_len
  let (inputs, targets) = build_lm_windows(all_ids, seq_len)
  train_lm_profile_steps_from_windows_inplace(
    inputs, targets, params, grads, adamw_state, config, warmup_steps, steps, batch_size,
    lr, step_offset, true,
  )
}

///|
pub fn transformer_eval_lm_ids(
  all_ids : Array[Int],
  params : TransformerParams,
  config : TransformerConfig,
  steps : Int,
  batch_size : Int,
  step_offset : Int,
) -> (Float, Float) {
  if steps <= 0 {
    return (Float::from_int(0), Float::from_int(0))
  }
  let seq_len = config.max_seq_len
  let (inputs, targets) = build_lm_windows(all_ids, seq_len)
  if inputs.length() == 0 {
    return (Float::from_int(0), Float::from_int(0))
  }
  let effective_batch = effective_batch_size(batch_size, inputs.length())
  let mask = causal_mask(seq_len).unwrap()
  let forward_workspace = transformer_forward_workspace(
    config, effective_batch, seq_len,
  )
  let offset = if step_offset < 0 { 0 } else { step_offset }
  let mut total_loss = Float::from_int(0)
  for step = 0; step < steps; step = step + 1 {
    let batch_inputs : Array[Array[Int]] = []
    let batch_targets : Array[Array[Int]] = []
    let logical_step = offset + step
    let base = logical_step * effective_batch % inputs.length()
    for i = 0; i < effective_batch; i = i + 1 {
      let idx = (base + i) % inputs.length()
      batch_inputs.push(inputs[idx])
      batch_targets.push(targets[idx])
    }
    let (logits, _cache) = transformer_forward_with_cache_workspace(
      batch_inputs,
      params,
      config,
      Some(mask),
      forward_workspace,
    )
    let labels = flatten_lm_labels(batch_targets, seq_len)
    let batch = logits.dim(0)
    let vocab = logits.dim(2)
    let logits_c = logits.contiguous()
    let flat_shape = shape_new_unchecked([batch * seq_len, vocab])
    let flat_logits = Tensor::{
      data: logits_c.data,
      shape: flat_shape,
      offset: 0,
      strides: flat_shape.strides(),
    }
    let (loss, _) = cross_entropy_forward_backward_fixed(flat_logits, labels)
    total_loss = total_loss + loss
  }
  let avg_loss = total_loss / Float::from_int(steps)
  (avg_loss, @math.expf(avg_loss))
}

///|
pub fn transformer_lm_checkpoint_to_bytes(
  config : TransformerConfig,
  params : TransformerParams,
  adamw_state : TransformerAdamwState?,
  global_step : Int,
) -> Bytes {
  let out : Array[Byte] = []
  append_ascii_bytes(out, "NNTRCHK1")
  let flags = match adamw_state {
    Some(_) => 1
    None => 0
  }
  append_u32_le(out, flags)
  append_u32_le(out, global_step)
  append_u32_le(out, config.vocab_size)
  append_u32_le(out, config.d_model)
  append_u32_le(out, config.num_heads)
  append_u32_le(out, config.num_layers)
  append_u32_le(out, config.d_ff)
  append_u32_le(out, config.max_seq_len)
  match adamw_state {
    Some(state) => {
      append_u32_le(out, state.step)
      append_f32_le(out, state.beta1_pow)
      append_f32_le(out, state.beta2_pow)
    }
    None => ()
  }
  append_transformer_params_f32(out, params)
  match adamw_state {
    Some(state) => {
      append_transformer_grads_f32(out, state.m)
      append_transformer_grads_f32(out, state.v)
    }
    None => ()
  }
  Bytes::from_array(out)
}

///|
pub fn transformer_lm_checkpoint_from_bytes(
  config : TransformerConfig,
  bytes : Bytes,
) -> Result[TransformerLmCheckpoint, String] {
  let magic = "NNTRCHK1"
  if bytes.length() < magic.length() {
    return Err("checkpoint too short")
  }
  for i = 0; i < magic.length(); i = i + 1 {
    let c = magic.to_array()[i]
    if bytes[i] != c.to_int().to_byte() {
      return Err("invalid checkpoint magic")
    }
  }
  let offset = magic.length()
  let flags_r = read_u32_le(bytes, offset)
  let (flags, offset1) = match flags_r {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let global_r = read_u32_le(bytes, offset1)
  let (global_step, offset2) = match global_r {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let vocab_r = read_u32_le(bytes, offset2)
  let (vocab, offset3) = match vocab_r {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let d_model_r = read_u32_le(bytes, offset3)
  let (d_model, offset4) = match d_model_r {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let heads_r = read_u32_le(bytes, offset4)
  let (heads, offset5) = match heads_r {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let layers_r = read_u32_le(bytes, offset5)
  let (layers, offset6) = match layers_r {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let d_ff_r = read_u32_le(bytes, offset6)
  let (d_ff, offset7) = match d_ff_r {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let seq_r = read_u32_le(bytes, offset7)
  let (max_seq_len, offset8) = match seq_r {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let mut next_offset = offset8
  if vocab != config.vocab_size ||
    d_model != config.d_model ||
    heads != config.num_heads ||
    layers != config.num_layers ||
    d_ff != config.d_ff ||
    max_seq_len != config.max_seq_len {
    return Err("checkpoint config does not match current config")
  }
  let has_adamw = (flags & 1) == 1
  let mut adamw_step = 0
  let mut beta1_pow = Float::from_int(1)
  let mut beta2_pow = Float::from_int(1)
  if has_adamw {
    let step_r = read_u32_le(bytes, next_offset)
    let (s, o1) = match step_r {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    let b1_r = read_f32_le(bytes, o1)
    let (b1, o2) = match b1_r {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    let b2_r = read_f32_le(bytes, o2)
    let (b2, o3) = match b2_r {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    adamw_step = s
    beta1_pow = b1
    beta2_pow = b2
    next_offset = o3
  }
  let params = transformer_init_params(config, 0)
  next_offset = match read_transformer_params_f32(bytes, next_offset, params) {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let mut restored_state : TransformerAdamwState? = None
  if has_adamw {
    let state = transformer_zero_adamw_state(config)
    state.step = adamw_step
    state.beta1_pow = beta1_pow
    state.beta2_pow = beta2_pow
    next_offset = match
      read_transformer_grads_f32(bytes, next_offset, state.m) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    next_offset = match
      read_transformer_grads_f32(bytes, next_offset, state.v) {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    restored_state = Some(state)
  }
  if next_offset != bytes.length() {
    return Err(
      "checkpoint trailing bytes: " + (bytes.length() - next_offset).to_string(),
    )
  }
  Ok({ params, adamw_state: restored_state, global_step })
}

///|
/// Train a tiny transformer on text data with step-based mini-batch updates.
/// Returns loss history (one value per step).
pub fn transformer_train_lm_steps(
  text : String,
  config : TransformerConfig,
  steps : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
) -> Array[Float] {
  let metrics = transformer_train_lm_profile_steps(
    text, config, 0, steps, batch_size, lr, seed,
  )
  let loss_history : Array[Float] = []
  for i = 0; i < metrics.length(); i = i + 1 {
    loss_history.push(metrics[i].loss)
  }
  loss_history
}

///|
/// Train a tiny transformer with AdamW and return loss history.
pub fn transformer_train_lm_steps_adamw(
  text : String,
  config : TransformerConfig,
  steps : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
) -> Array[Float] {
  let metrics = transformer_train_lm_profile_steps_adamw(
    text, config, 0, steps, batch_size, lr, seed,
  )
  let loss_history : Array[Float] = []
  for i = 0; i < metrics.length(); i = i + 1 {
    loss_history.push(metrics[i].loss)
  }
  loss_history
}

///|
fn transformer_train_lm_profile_steps_from_ids_impl(
  all_ids : Array[Int],
  config : TransformerConfig,
  warmup_steps : Int,
  steps : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
  use_adamw : Bool,
) -> Array[TransformerLmStepMetric] {
  let params = transformer_init_params(config, seed)
  let grads = transformer_zero_grads(config)
  let adamw_state = transformer_zero_adamw_state(config)
  let seq_len = config.max_seq_len
  let (inputs, targets) = build_lm_windows(all_ids, seq_len)
  train_lm_profile_steps_from_windows_inplace(
    inputs, targets, params, grads, adamw_state, config, warmup_steps, steps, batch_size,
    lr, 0, use_adamw,
  )
}

///|
fn transformer_train_lm_profile_steps_impl(
  text : String,
  config : TransformerConfig,
  warmup_steps : Int,
  steps : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
  use_adamw : Bool,
) -> Array[TransformerLmStepMetric] {
  let tokenizer = char_tokenizer_from_text(text)
  let all_ids = tokenizer.encode(text)
  transformer_train_lm_profile_steps_from_ids_impl(
    all_ids, config, warmup_steps, steps, batch_size, lr, seed, use_adamw,
  )
}

///|
/// Train from pre-tokenized ids and return loss history.
pub fn transformer_train_lm_steps_ids(
  all_ids : Array[Int],
  config : TransformerConfig,
  steps : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
) -> Array[Float] {
  let metrics = transformer_train_lm_profile_steps_ids(
    all_ids, config, 0, steps, batch_size, lr, seed,
  )
  let loss_history : Array[Float] = []
  for i = 0; i < metrics.length(); i = i + 1 {
    loss_history.push(metrics[i].loss)
  }
  loss_history
}

///|
/// AdamW variant of transformer_train_lm_steps_ids.
pub fn transformer_train_lm_steps_ids_adamw(
  all_ids : Array[Int],
  config : TransformerConfig,
  steps : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
) -> Array[Float] {
  let metrics = transformer_train_lm_profile_steps_ids_adamw(
    all_ids, config, 0, steps, batch_size, lr, seed,
  )
  let loss_history : Array[Float] = []
  for i = 0; i < metrics.length(); i = i + 1 {
    loss_history.push(metrics[i].loss)
  }
  loss_history
}

///|
/// Train from pre-tokenized ids and record per-step metrics.
pub fn transformer_train_lm_profile_steps_ids(
  all_ids : Array[Int],
  config : TransformerConfig,
  warmup_steps : Int,
  steps : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
) -> Array[TransformerLmStepMetric] {
  transformer_train_lm_profile_steps_from_ids_impl(
    all_ids, config, warmup_steps, steps, batch_size, lr, seed, false,
  )
}

///|
/// AdamW variant of transformer_train_lm_profile_steps_ids.
pub fn transformer_train_lm_profile_steps_ids_adamw(
  all_ids : Array[Int],
  config : TransformerConfig,
  warmup_steps : Int,
  steps : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
) -> Array[TransformerLmStepMetric] {
  transformer_train_lm_profile_steps_from_ids_impl(
    all_ids, config, warmup_steps, steps, batch_size, lr, seed, true,
  )
}

///|
/// Train a tiny transformer on text data and record per-step metrics.
/// Returns one metric per measured step (warmup is excluded from results).
pub fn transformer_train_lm_profile_steps(
  text : String,
  config : TransformerConfig,
  warmup_steps : Int,
  steps : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
) -> Array[TransformerLmStepMetric] {
  transformer_train_lm_profile_steps_impl(
    text, config, warmup_steps, steps, batch_size, lr, seed, false,
  )
}

///|
/// AdamW variant of transformer_train_lm_profile_steps.
pub fn transformer_train_lm_profile_steps_adamw(
  text : String,
  config : TransformerConfig,
  warmup_steps : Int,
  steps : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
) -> Array[TransformerLmStepMetric] {
  transformer_train_lm_profile_steps_impl(
    text, config, warmup_steps, steps, batch_size, lr, seed, true,
  )
}

///|
/// Train with full-batch updates (compat wrapper).
/// Returns loss history (one value per epoch).
pub fn transformer_train(
  text : String,
  config : TransformerConfig,
  epochs : Int,
  lr : Float,
  seed : Int,
) -> Array[Float] {
  transformer_train_lm_steps(text, config, epochs, 1 << 30, lr, seed)
}
