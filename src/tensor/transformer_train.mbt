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
  attn_weights : Array[Array[Float]] // [batch*num_heads][seq_q*seq_k] softmax output
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
fn layer_norm_with_cache(
  t : Tensor,
  gamma : Tensor,
  beta : Tensor,
  eps : Float,
) -> (Tensor, FixedArray[Float], FixedArray[Float]) {
  let last_dim = t.dim(-1)
  let tc = t.contiguous()
  let gc = gamma.contiguous()
  let bc = beta.contiguous()
  let outer = t.numel() / last_dim
  let result = FixedArray::make(tc.numel(), Float::from_int(0))
  let means = FixedArray::make(outer, Float::from_int(0))
  let rstds = FixedArray::make(outer, Float::from_int(0))
  native_layer_norm_fwd(
    tc.data, gc.data, bc.data, result, means, rstds, outer, last_dim, eps,
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
    0, 1, n, in_dim, out_dim,
    Float::from_int(1),
    dyc.data, out_dim,
    wc.data, out_dim,
    Float::from_int(0),
    dx_data, in_dim,
  )
  // dW += x^T @ dy : [in_dim, n] @ [n, out_dim] -> [in_dim, out_dim], beta=1.0
  native_sgemm(
    1, 0, in_dim, out_dim, n,
    Float::from_int(1),
    xc.data, in_dim,
    dyc.data, out_dim,
    Float::from_int(1),
    d_w.data, out_dim,
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
  let outer = dy.numel() / last_dim
  for i = 0; i < outer; i = i + 1 {
    let base = i * last_dim
    for j = 0; j < last_dim; j = j + 1 {
      d_bias.data[j] = d_bias.data[j] + dyc.data[base + j]
    }
  }
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
    dyc.data, xc.data, mean, rstd, gc.data,
    dx_data, d_gamma.data, d_beta.data,
    outer, last_dim,
  )
  Tensor::{
    data: dx_data,
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
/// Feed-forward backward
/// dy: [batch, seq, d_model]
/// Returns dx: [batch, seq, d_model]
fn feed_forward_backward(
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
  let d_concat = batched_linear_backward(
    dy,
    cache.attn_out_pre_wo,
    params.w_o,
    grads.d_w_o,
  )

  // d_concat is [batch, seq, d_model]
  // Reshape to [batch, num_heads, seq, d_k] for per-head backward
  let d_concat_heads = reshape_for_heads(d_concat, batch, seq, num_heads, d_k)

  // Reshape cached projections to head format
  let q_heads = reshape_for_heads(cache.q_proj, batch, seq, num_heads, d_k)
  let k_heads = reshape_for_heads(cache.k_proj, batch, seq, num_heads, d_k)
  let v_heads = reshape_for_heads(cache.v_proj, batch, seq, num_heads, d_k)
  let scale = Float::from_double(1.0 / d_k.to_double().sqrt())

  // Per-head attention backward
  let dq_heads_data = FixedArray::make(
    batch * num_heads * seq * d_k,
    Float::from_int(0),
  )
  let dk_heads_data = FixedArray::make(
    batch * num_heads * seq * d_k,
    Float::from_int(0),
  )
  let dv_heads_data = FixedArray::make(
    batch * num_heads * seq * d_k,
    Float::from_int(0),
  )
  let dc = d_concat_heads.contiguous()
  let qc = q_heads.contiguous()
  let kc = k_heads.contiguous()
  let vc = v_heads.contiguous()
  let attn_w_buf = FixedArray::make(seq * seq, Float::from_int(0))
  for b = 0; b < batch; b = b + 1 {
    for h = 0; h < num_heads; h = h + 1 {
      let head_idx = b * num_heads + h
      let attn_w = cache.attn_weights[head_idx]
      let head_base = (b * num_heads + h) * seq * d_k

      // d_attn_weights = d_out @ V^T : [seq, d_k] @ [d_k, seq] -> [seq, seq]
      let d_attn_weights = FixedArray::make(seq * seq, Float::from_int(0))
      native_sgemm_offset(
        0, 1, seq, seq, d_k,
        Float::from_int(1),
        dc.data, head_base, d_k,
        vc.data, head_base, d_k,
        Float::from_int(0),
        d_attn_weights, 0, seq,
      )

      // dV += attn_weights^T @ d_out : [seq, seq]^T @ [seq, d_k] -> [seq, d_k]
      // Copy attn_w (Array[Float]) to FixedArray for SGEMM
      for idx = 0; idx < seq * seq; idx = idx + 1 {
        attn_w_buf[idx] = attn_w[idx]
      }
      native_sgemm_offset(
        1, 0, seq, d_k, seq,
        Float::from_int(1),
        attn_w_buf, 0, seq,
        dc.data, head_base, d_k,
        Float::from_int(1),
        dv_heads_data, head_base, d_k,
      )

      // Softmax backward: d_scores = attn_weights * (d_attn_weights - sum(d_attn_weights * attn_weights))
      let d_scores = FixedArray::make(seq * seq, Float::from_int(0))
      for i = 0; i < seq; i = i + 1 {
        let mut dot_sum = Float::from_int(0)
        for j = 0; j < seq; j = j + 1 {
          dot_sum = dot_sum + d_attn_weights[i * seq + j] * attn_w[i * seq + j]
        }
        for j = 0; j < seq; j = j + 1 {
          d_scores[i * seq + j] = attn_w[i * seq + j] *
            (d_attn_weights[i * seq + j] - dot_sum)
        }
      }

      // Scale backward
      for i = 0; i < seq * seq; i = i + 1 {
        d_scores[i] = d_scores[i] * scale
      }

      // dQ = d_scores @ K : [seq, seq] @ [seq, d_k] -> [seq, d_k]
      native_sgemm_offset(
        0, 0, seq, d_k, seq,
        Float::from_int(1),
        d_scores, 0, seq,
        kc.data, head_base, d_k,
        Float::from_int(0),
        dq_heads_data, head_base, d_k,
      )
      // dK = d_scores^T @ Q : [seq, seq]^T @ [seq, d_k] -> [seq, d_k]
      native_sgemm_offset(
        1, 0, seq, d_k, seq,
        Float::from_int(1),
        d_scores, 0, seq,
        qc.data, head_base, d_k,
        Float::from_int(0),
        dk_heads_data, head_base, d_k,
      )
    }
  }

  // Reshape heads back to [batch, seq, d_model]
  let dq_heads = Tensor::{
    data: dq_heads_data,
    shape: shape_new_unchecked([batch, num_heads, seq, d_k]),
    offset: 0,
    strides: shape_new_unchecked([batch, num_heads, seq, d_k]).strides(),
  }
  let dk_heads = Tensor::{
    data: dk_heads_data,
    shape: shape_new_unchecked([batch, num_heads, seq, d_k]),
    offset: 0,
    strides: shape_new_unchecked([batch, num_heads, seq, d_k]).strides(),
  }
  let dv_heads = Tensor::{
    data: dv_heads_data,
    shape: shape_new_unchecked([batch, num_heads, seq, d_k]),
    offset: 0,
    strides: shape_new_unchecked([batch, num_heads, seq, d_k]).strides(),
  }
  let dq_proj = reshape_from_heads(dq_heads, batch, seq, num_heads, d_k)
  let dk_proj = reshape_from_heads(dk_heads, batch, seq, num_heads, d_k)
  let dv_proj = reshape_from_heads(dv_heads, batch, seq, num_heads, d_k)

  // W_q, W_k, W_v backward (self-attention: input is the same for Q, K, V)
  let dx_q = batched_linear_backward(
    dq_proj,
    cache.ln1_out,
    params.w_q,
    grads.d_w_q,
  )
  let dx_k = batched_linear_backward(
    dk_proj,
    cache.ln1_out,
    params.w_k,
    grads.d_w_k,
  )
  let dx_v = batched_linear_backward(
    dv_proj,
    cache.ln1_out,
    params.w_v,
    grads.d_w_v,
  )

  // Sum the three dx contributions (self-attention)
  let dx_data = FixedArray::make(batch * seq * d_model, Float::from_int(0))
  for i = 0; i < dx_data.length(); i = i + 1 {
    dx_data[i] = dx_q.data[i] + dx_k.data[i] + dx_v.data[i]
  }
  Tensor::{
    data: dx_data,
    shape: dx_q.shape,
    offset: 0,
    strides: dx_q.shape.strides(),
  }
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
  let n = dy.numel()

  // Forward: x -> LN1 -> MHA -> +residual1 -> LN2 -> FFN -> +residual2
  // Backward:
  // dy comes from after residual2
  // residual2: dy goes to both FFN path and x_after_res1

  // FFN backward
  let d_ln2 = feed_forward_backward(dy, cache, params, grads)

  // LN2 backward
  let d_res1_from_ffn = layer_norm_backward(
    d_ln2,
    cache.x_after_res1,
    cache.ln2_mean,
    cache.ln2_rstd,
    params.ln2_gamma,
    grads.d_ln2_gamma,
    grads.d_ln2_beta,
  )

  // Add residual2: dy + d_res1_from_ffn
  let d_res1_data = FixedArray::make(n, Float::from_int(0))
  for i = 0; i < n; i = i + 1 {
    d_res1_data[i] = dy.data[i] + d_res1_from_ffn.data[i]
  }
  let d_res1 = Tensor::{
    data: d_res1_data,
    shape: dy.shape,
    offset: 0,
    strides: dy.shape.strides(),
  }

  // MHA backward
  let d_ln1 = multi_head_attention_backward(
    d_res1, cache, params, config, grads,
  )

  // LN1 backward
  let d_input_from_mha = layer_norm_backward(
    d_ln1,
    cache.x_input,
    cache.ln1_mean,
    cache.ln1_rstd,
    params.ln1_gamma,
    grads.d_ln1_gamma,
    grads.d_ln1_beta,
  )

  // Add residual1: d_res1 + d_input_from_mha
  let dx_data = FixedArray::make(n, Float::from_int(0))
  for i = 0; i < n; i = i + 1 {
    dx_data[i] = d_res1.data[i] + d_input_from_mha.data[i]
  }
  Tensor::{
    data: dx_data,
    shape: dy.shape,
    offset: 0,
    strides: dy.shape.strides(),
  }
}

///|
/// Forward pass with cache for training
pub fn transformer_forward_with_cache(
  tokens : Array[Array[Int]],
  params : TransformerParams,
  config : TransformerConfig,
  mask : Tensor?,
) -> (Tensor, TransformerCache) {
  let batch = tokens.length()
  let seq = tokens[0].length()
  let d_model = config.d_model
  let num_heads = config.num_heads
  let d_k = d_model / num_heads

  // Token embedding
  let tok_emb = embedding(tokens, params.token_embedding)
  // Add positional embedding
  let x_with_pos = add_positional_embedding(tok_emb, params.pos_embedding)
  let mut hidden = x_with_pos
  let block_caches : Array[TransformerBlockCache] = []
  for layer = 0; layer < config.num_layers; layer = layer + 1 {
    let bp = params.blocks[layer]
    let x_input = hidden

    // LN1 with cache
    let (ln1_out, ln1_mean, ln1_rstd) = layer_norm_with_cache(
      x_input,
      bp.ln1_gamma,
      bp.ln1_beta,
      config.eps,
    )

    // Q, K, V projections
    let q_proj = batched_linear(ln1_out, bp.w_q).unwrap()
    let k_proj = batched_linear(ln1_out, bp.w_k).unwrap()
    let v_proj = batched_linear(ln1_out, bp.w_v).unwrap()

    // Reshape for heads
    let q_heads = reshape_for_heads(q_proj, batch, seq, num_heads, d_k)
    let k_heads = reshape_for_heads(k_proj, batch, seq, num_heads, d_k)
    let v_heads = reshape_for_heads(v_proj, batch, seq, num_heads, d_k)

    // Attention with cached weights
    let scale = Float::from_double(1.0 / d_k.to_double().sqrt())
    let qc = q_heads.contiguous()
    let kc = k_heads.contiguous()
    let vc = v_heads.contiguous()
    let mc = match mask {
      None => None
      Some(m) => Some(m.contiguous())
    }
    let total_heads = batch * num_heads
    let scores_size = seq * seq
    let attn_weights_all : Array[Array[Float]] = Array::make(total_heads, [])
    let scores = FixedArray::make(scores_size, Float::from_int(0))
    let attn_result_data = FixedArray::make(
      batch * num_heads * seq * d_k,
      Float::from_int(0),
    )
    for b = 0; b < batch; b = b + 1 {
      for h = 0; h < num_heads; h = h + 1 {
        let head_idx = b * num_heads + h
        let head_base = head_idx * seq * d_k
        // scores = Q @ K^T * scale : [seq, d_k] @ [d_k, seq] -> [seq, seq]
        native_sgemm_offset(
          0, 1, seq, seq, d_k,
          scale,
          qc.data, head_base, d_k,
          kc.data, head_base, d_k,
          Float::from_int(0),
          scores, 0, seq,
        )
        // Apply mask
        match mc {
          Some(m) =>
            for i = 0; i < seq; i = i + 1 {
              for j = 0; j < seq; j = j + 1 {
                scores[i * seq + j] = scores[i * seq + j] + m.at2(i, j)
              }
            }
          None => ()
        }
        // Softmax
        native_softmax_inplace(scores, seq, seq)
        let scores_copy = Array::make(scores_size, Float::from_int(0))
        for idx = 0; idx < scores_size; idx = idx + 1 {
          scores_copy[idx] = scores[idx]
        }
        attn_weights_all[head_idx] = scores_copy
        // output = attn_weights @ V : [seq, seq] @ [seq, d_k] -> [seq, d_k]
        native_sgemm_offset(
          0, 0, seq, d_k, seq,
          Float::from_int(1),
          scores, 0, seq,
          vc.data, head_base, d_k,
          Float::from_int(0),
          attn_result_data, head_base, d_k,
        )
      }
    }
    let attn_heads_out = Tensor::{
      data: attn_result_data,
      shape: shape_new_unchecked([batch, num_heads, seq, d_k]),
      offset: 0,
      strides: shape_new_unchecked([batch, num_heads, seq, d_k]).strides(),
    }
    let attn_concat = reshape_from_heads(
      attn_heads_out, batch, seq, num_heads, d_k,
    )

    // W_o projection
    let attn_out = batched_linear(attn_concat, bp.w_o).unwrap()

    // Residual 1
    let x_after_res1_data = FixedArray::make(
      batch * seq * d_model,
      Float::from_int(0),
    )
    let xic = x_input.contiguous()
    let aoc = attn_out.contiguous()
    for i = 0; i < x_after_res1_data.length(); i = i + 1 {
      x_after_res1_data[i] = xic.data[i] + aoc.data[i]
    }
    let x_after_res1 = Tensor::{
      data: x_after_res1_data,
      shape: x_input.shape,
      offset: 0,
      strides: x_input.shape.strides(),
    }

    // LN2 with cache
    let (ln2_out, ln2_mean, ln2_rstd) = layer_norm_with_cache(
      x_after_res1,
      bp.ln2_gamma,
      bp.ln2_beta,
      config.eps,
    )

    // FFN with cache
    let ff_h = batched_linear(ln2_out, bp.ff_w1).unwrap()
    let ff_pre_gelu = add_bias(ff_h, bp.ff_b1)
    let ff_post_gelu = tensor_gelu(ff_pre_gelu)
    let ff_out_h = batched_linear(ff_post_gelu, bp.ff_w2).unwrap()
    let ff_out = add_bias(ff_out_h, bp.ff_b2)

    // Residual 2
    let hidden_data = FixedArray::make(batch * seq * d_model, Float::from_int(0))
    let r1c = x_after_res1.contiguous()
    let foc = ff_out.contiguous()
    for i = 0; i < hidden_data.length(); i = i + 1 {
      hidden_data[i] = r1c.data[i] + foc.data[i]
    }
    hidden = Tensor::{
      data: hidden_data,
      shape: x_input.shape,
      offset: 0,
      strides: x_input.shape.strides(),
    }
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
  let (ln_final_out, ln_final_mean, ln_final_rstd) = layer_norm_with_cache(
    hidden_before_final_ln,
    params.ln_final_gamma,
    params.ln_final_beta,
    config.eps,
  )

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

  // Compute loss
  let loss = tensor_cross_entropy(last_logits, labels).unwrap()

  // Cross-entropy backward on last position only
  let d_last_logits = cross_entropy_backward(last_logits, labels)

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

  // LM head backward: logits = ln_final_out @ lm_head
  let d_ln_final_out = batched_linear_backward(
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
  for i = 0; i < t.data.length(); i = i + 1 {
    t.data[i] = Float::from_int(0)
  }
}

///|
/// Train a tiny transformer on text data
/// Returns loss history (one value per epoch)
pub fn transformer_train(
  text : String,
  config : TransformerConfig,
  epochs : Int,
  lr : Float,
  seed : Int,
) -> Array[Float] {
  let tokenizer = char_tokenizer_from_text(text)
  let all_ids = tokenizer.encode(text)
  let params = transformer_init_params(config, seed)
  let grads = transformer_zero_grads(config)
  let loss_history : Array[Float] = []
  let seq_len = config.max_seq_len

  // Create training examples: sliding window
  let inputs : Array[Array[Int]] = []
  let targets : Array[Int] = []
  for i = 0; i + seq_len < all_ids.length(); i = i + 1 {
    let input_seq : Array[Int] = []
    for j = 0; j < seq_len; j = j + 1 {
      input_seq.push(all_ids[i + j])
    }
    inputs.push(input_seq)
    targets.push(all_ids[i + seq_len])
  }
  if inputs.length() == 0 {
    return loss_history
  }
  let mask = causal_mask(seq_len).unwrap()
  for _epoch = 0; _epoch < epochs; _epoch = _epoch + 1 {
    zero_grads(grads)

    // Forward with cache (batch = all examples)
    let (_, cache) = transformer_forward_with_cache(
      inputs,
      params,
      config,
      Some(mask),
    )

    // Backward
    let loss = transformer_backward(cache, params, config, grads, targets)
    loss_history.push(loss)

    // SGD update
    sgd_update(params, grads, lr)
  }
  loss_history
}
