///|
/// Transformer configuration
pub struct TransformerConfig {
  vocab_size : Int
  d_model : Int
  num_heads : Int
  num_layers : Int
  d_ff : Int // Feed-forward hidden dimension (typically 4 * d_model)
  max_seq_len : Int
  eps : Float // LayerNorm epsilon
}

///|
pub fn transformer_config(
  vocab_size : Int,
  d_model : Int,
  num_heads : Int,
  num_layers : Int,
  d_ff : Int,
  max_seq_len : Int,
) -> Result[TransformerConfig, String] {
  if d_model % num_heads != 0 {
    return Err("d_model must be divisible by num_heads")
  }
  Ok(TransformerConfig::{
    vocab_size,
    d_model,
    num_heads,
    num_layers,
    d_ff,
    max_seq_len,
    eps: Float::from_double(0.00001),
  })
}

///|
/// Parameters for a single Transformer block
pub struct TransformerBlockParams {
  // LayerNorm 1
  ln1_gamma : Tensor
  ln1_beta : Tensor
  // Multi-Head Attention
  w_q : Tensor // [d_model, d_model]
  w_k : Tensor
  w_v : Tensor
  w_o : Tensor
  // LayerNorm 2
  ln2_gamma : Tensor
  ln2_beta : Tensor
  // Feed-Forward Network
  ff_w1 : Tensor // [d_model, d_ff]
  ff_b1 : Tensor // [d_ff]
  ff_w2 : Tensor // [d_ff, d_model]
  ff_b2 : Tensor // [d_model]
}

///|
/// Full Transformer model parameters
pub struct TransformerParams {
  token_embedding : Tensor // [vocab_size, d_model]
  pos_embedding : Tensor // [max_seq_len, d_model]
  blocks : Array[TransformerBlockParams]
  ln_final_gamma : Tensor // [d_model]
  ln_final_beta : Tensor // [d_model]
  lm_head : Tensor // [d_model, vocab_size]
}

///|
/// Embedding lookup: indices [batch, seq] -> [batch, seq, d_model]
pub fn embedding(indices : Array[Array[Int]], weight : Tensor) -> Tensor {
  let batch = indices.length()
  let seq = indices[0].length()
  let d_model = weight.dim(1)
  let out_shape = shape_new_unchecked([batch, seq, d_model])
  let result = Array::make(batch * seq * d_model, Float::from_int(0))
  let wc = weight.contiguous()
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < seq; s = s + 1 {
      let token_idx = indices[b][s]
      let src_base = token_idx * d_model
      let dst_base = b * seq * d_model + s * d_model
      for d = 0; d < d_model; d = d + 1 {
        result[dst_base + d] = wc.data[src_base + d]
      }
    }
  }
  Tensor::{
    data: result,
    shape: out_shape,
    offset: 0,
    strides: out_shape.strides(),
  }
}

///|
/// Positional embedding: add position vectors to input
/// x: [batch, seq, d_model], pos_weight: [max_seq_len, d_model]
pub fn add_positional_embedding(x : Tensor, pos_weight : Tensor) -> Tensor {
  let batch = x.dim(0)
  let seq = x.dim(1)
  let d_model = x.dim(2)
  let xc = x.contiguous()
  let pc = pos_weight.contiguous()
  let result = Array::make(batch * seq * d_model, Float::from_int(0))
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < seq; s = s + 1 {
      let x_base = b * seq * d_model + s * d_model
      let p_base = s * d_model
      for d = 0; d < d_model; d = d + 1 {
        result[x_base + d] = xc.data[x_base + d] + pc.data[p_base + d]
      }
    }
  }
  Tensor::{
    data: result,
    shape: x.shape,
    offset: 0,
    strides: x.shape.strides(),
  }
}

///|
/// Feed-Forward Network: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)
/// x: [batch, seq, d_model]
pub fn feed_forward(
  x : Tensor,
  w1 : Tensor,
  b1 : Tensor,
  w2 : Tensor,
  b2 : Tensor,
) -> Result[Tensor, String] {
  // x @ w1 + b1
  let h = match batched_linear(x, w1) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  let h_bias = add_bias(h, b1)
  // GELU
  let h_act = tensor_gelu(h_bias)
  // h_act @ w2 + b2
  let out = match batched_linear(h_act, w2) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  Ok(add_bias(out, b2))
}

///|
/// Add bias to last dimension: x[..., i] += bias[i]
fn add_bias(x : Tensor, bias : Tensor) -> Tensor {
  let xc = x.contiguous()
  let bc = bias.contiguous()
  let last_dim = x.dim(-1)
  let result = Array::make(xc.numel(), Float::from_int(0))
  let outer = xc.numel() / last_dim
  for i = 0; i < outer; i = i + 1 {
    let base = i * last_dim
    for j = 0; j < last_dim; j = j + 1 {
      result[base + j] = xc.data[base + j] + bc.data[j]
    }
  }
  Tensor::{
    data: result,
    shape: x.shape,
    offset: 0,
    strides: x.shape.strides(),
  }
}

///|
/// Single Transformer Block (pre-norm, GPT-style)
/// x: [batch, seq, d_model]
pub fn transformer_block(
  x : Tensor,
  params : TransformerBlockParams,
  config : TransformerConfig,
  mask : Tensor?,
) -> Result[Tensor, String] {
  // Pre-norm Attention
  let ln1 = match
    tensor_layer_norm(x, params.ln1_gamma, params.ln1_beta, config.eps) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  let attn = match
    multi_head_attention(
      ln1,
      ln1,
      ln1,
      params.w_q,
      params.w_k,
      params.w_v,
      params.w_o,
      config.num_heads,
      mask,
    ) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  // Residual
  let x2 = match tensor_add(x, attn) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  // Pre-norm FFN
  let ln2 = match
    tensor_layer_norm(x2, params.ln2_gamma, params.ln2_beta, config.eps) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  let ff = match
    feed_forward(ln2, params.ff_w1, params.ff_b1, params.ff_w2, params.ff_b2) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  // Residual
  tensor_add(x2, ff)
}

///|
/// Full Transformer forward pass (decoder-only, GPT-style)
/// tokens: [batch, seq] as Array[Array[Int]]
/// Returns logits: [batch, seq, vocab_size]
pub fn transformer_forward(
  tokens : Array[Array[Int]],
  params : TransformerParams,
  config : TransformerConfig,
  mask : Tensor?,
) -> Result[Tensor, String] {
  // Token embedding
  let tok_emb = embedding(tokens, params.token_embedding)
  // Add positional embedding
  let x = add_positional_embedding(tok_emb, params.pos_embedding)
  // Apply transformer blocks
  let mut hidden = x
  for i = 0; i < config.num_layers; i = i + 1 {
    hidden = match transformer_block(hidden, params.blocks[i], config, mask) {
      Ok(t) => t
      Err(e) => return Err(e)
    }
  }
  // Final LayerNorm
  let ln_out = match
    tensor_layer_norm(
      hidden,
      params.ln_final_gamma,
      params.ln_final_beta,
      config.eps,
    ) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  // LM head: [batch, seq, d_model] @ [d_model, vocab_size] -> [batch, seq, vocab_size]
  batched_linear(ln_out, params.lm_head)
}

///|
/// Initialize transformer parameters with small random values (deterministic)
pub fn transformer_init_params(
  config : TransformerConfig,
  seed : Int,
) -> TransformerParams {
  let mut rng = seed
  let next_float = fn() -> Float {
    // Simple LCG for deterministic initialization
    rng = rng * 1103515245 + 12345
    let val = (rng / 65536 % 32768).to_double() / 32768.0
    // Xavier-like initialization scaled by 0.02
    (val * 0.04 - 0.02) |> Float::from_double
  }
  let init_tensor = fn(shape : Shape) -> Tensor {
    let data = Array::makei(shape.numel(), fn(_i) { next_float() })
    Tensor::{ data, shape, offset: 0, strides: shape.strides() }
  }
  let ones = fn(n : Int) -> Tensor {
    let shape = shape_new_unchecked([n])
    tensor_ones(shape)
  }
  let zeros = fn(n : Int) -> Tensor {
    let shape = shape_new_unchecked([n])
    tensor_zeros(shape)
  }
  // Token and position embeddings
  let token_embedding = init_tensor(
    shape_new_unchecked([config.vocab_size, config.d_model]),
  )
  let pos_embedding = init_tensor(
    shape_new_unchecked([config.max_seq_len, config.d_model]),
  )
  // Transformer blocks
  let d = config.d_model
  let d_ff = config.d_ff
  let blocks = Array::makei(config.num_layers, fn(_i) {
    TransformerBlockParams::{
      ln1_gamma: ones(d),
      ln1_beta: zeros(d),
      w_q: init_tensor(shape_new_unchecked([d, d])),
      w_k: init_tensor(shape_new_unchecked([d, d])),
      w_v: init_tensor(shape_new_unchecked([d, d])),
      w_o: init_tensor(shape_new_unchecked([d, d])),
      ln2_gamma: ones(d),
      ln2_beta: zeros(d),
      ff_w1: init_tensor(shape_new_unchecked([d, d_ff])),
      ff_b1: zeros(d_ff),
      ff_w2: init_tensor(shape_new_unchecked([d_ff, d])),
      ff_b2: zeros(d),
    }
  })
  // Final LayerNorm and LM head
  TransformerParams::{
    token_embedding,
    pos_embedding,
    blocks,
    ln_final_gamma: ones(d),
    ln_final_beta: zeros(d),
    lm_head: init_tensor(shape_new_unchecked([d, config.vocab_size])),
  }
}
