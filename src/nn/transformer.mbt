///|
/// TransformerSpec for GPU execution
pub struct TransformerSpec {
  vocab_size : Int
  d_model : Int
  num_heads : Int
  num_layers : Int
  d_ff : Int
  max_seq_len : Int
  batch_size : Int
  seq_len : Int
} derive(Show, Eq)

///|
pub fn transformer_spec_new(
  vocab_size : Int,
  d_model : Int,
  num_heads : Int,
  num_layers : Int,
  d_ff : Int,
  max_seq_len : Int,
  batch_size : Int,
  seq_len : Int,
) -> Result[TransformerSpec, MlpError] {
  if vocab_size <= 0 {
    return Err(InvalidSpec("vocab_size must be > 0"))
  }
  if d_model <= 0 {
    return Err(InvalidSpec("d_model must be > 0"))
  }
  if num_heads <= 0 {
    return Err(InvalidSpec("num_heads must be > 0"))
  }
  if d_model % num_heads != 0 {
    return Err(InvalidSpec("d_model must be divisible by num_heads"))
  }
  if num_layers <= 0 {
    return Err(InvalidSpec("num_layers must be > 0"))
  }
  if d_ff <= 0 {
    return Err(InvalidSpec("d_ff must be > 0"))
  }
  if max_seq_len <= 0 {
    return Err(InvalidSpec("max_seq_len must be > 0"))
  }
  if batch_size <= 0 {
    return Err(InvalidSpec("batch_size must be > 0"))
  }
  if seq_len <= 0 {
    return Err(InvalidSpec("seq_len must be > 0"))
  }
  if seq_len > max_seq_len {
    return Err(InvalidSpec("seq_len must be <= max_seq_len"))
  }
  Ok(
    TransformerSpec::{
      vocab_size,
      d_model,
      num_heads,
      num_layers,
      d_ff,
      max_seq_len,
      batch_size,
      seq_len,
    },
  )
}

///|
fn transformer_d_k(spec : TransformerSpec) -> Int {
  spec.d_model / spec.num_heads
}

///|
/// Buffer sizes for transformer GPU execution
pub struct TransformerGpuBufferPlan {
  // Parameter buffers (read-only)
  token_embedding_bytes : Int
  pos_embedding_bytes : Int
  causal_mask_bytes : Int
  // Per-layer parameter bytes (same for all layers)
  ln1_gamma_bytes : Int
  ln1_beta_bytes : Int
  w_q_bytes : Int
  w_k_bytes : Int
  w_v_bytes : Int
  w_o_bytes : Int
  ln2_gamma_bytes : Int
  ln2_beta_bytes : Int
  ff_w1_bytes : Int
  ff_b1_bytes : Int
  ff_w2_bytes : Int
  ff_b2_bytes : Int
  // Final layer norm + lm_head
  ln_final_gamma_bytes : Int
  ln_final_beta_bytes : Int
  lm_head_bytes : Int
  // Activation buffers (read-write)
  tokens_in_bytes : Int
  x_bytes : Int
  x_tmp_bytes : Int
  x_residual_bytes : Int
  q_bytes : Int
  k_bytes : Int
  v_bytes : Int
  attn_out_bytes : Int
  attn_proj_bytes : Int
  ff_h_bytes : Int
  logits_bytes : Int
} derive(Show, Eq)

///|
pub fn transformer_plan_buffers(
  spec : TransformerSpec,
) -> TransformerGpuBufferPlan {
  let d_k = transformer_d_k(spec)
  let bs = spec.batch_size * spec.seq_len
  {
    // Parameter buffers
    token_embedding_bytes: bytes_for_f32(spec.vocab_size * spec.d_model),
    pos_embedding_bytes: bytes_for_f32(spec.max_seq_len * spec.d_model),
    causal_mask_bytes: bytes_for_f32(spec.seq_len * spec.seq_len),
    ln1_gamma_bytes: bytes_for_f32(spec.d_model),
    ln1_beta_bytes: bytes_for_f32(spec.d_model),
    w_q_bytes: bytes_for_f32(spec.d_model * spec.d_model),
    w_k_bytes: bytes_for_f32(spec.d_model * spec.d_model),
    w_v_bytes: bytes_for_f32(spec.d_model * spec.d_model),
    w_o_bytes: bytes_for_f32(spec.d_model * spec.d_model),
    ln2_gamma_bytes: bytes_for_f32(spec.d_model),
    ln2_beta_bytes: bytes_for_f32(spec.d_model),
    ff_w1_bytes: bytes_for_f32(spec.d_model * spec.d_ff),
    ff_b1_bytes: bytes_for_f32(spec.d_ff),
    ff_w2_bytes: bytes_for_f32(spec.d_ff * spec.d_model),
    ff_b2_bytes: bytes_for_f32(spec.d_model),
    ln_final_gamma_bytes: bytes_for_f32(spec.d_model),
    ln_final_beta_bytes: bytes_for_f32(spec.d_model),
    lm_head_bytes: bytes_for_f32(spec.d_model * spec.vocab_size),
    // Activation buffers
    tokens_in_bytes: bytes_for_u32(spec.batch_size * spec.seq_len),
    x_bytes: bytes_for_f32(bs * spec.d_model),
    x_tmp_bytes: bytes_for_f32(bs * spec.d_model),
    x_residual_bytes: bytes_for_f32(bs * spec.d_model),
    q_bytes: bytes_for_f32(spec.batch_size * spec.num_heads * spec.seq_len * d_k),
    k_bytes: bytes_for_f32(spec.batch_size * spec.num_heads * spec.seq_len * d_k),
    v_bytes: bytes_for_f32(spec.batch_size * spec.num_heads * spec.seq_len * d_k),
    attn_out_bytes: bytes_for_f32(
      spec.batch_size * spec.num_heads * spec.seq_len * d_k,
    ),
    attn_proj_bytes: bytes_for_f32(bs * spec.d_model),
    ff_h_bytes: bytes_for_f32(bs * spec.d_ff),
    logits_bytes: bytes_for_f32(bs * spec.vocab_size),
  }
}

///|
/// Resource plan with BufferDescriptor for each GPU buffer
pub struct TransformerGpuResourcePlan {
  // Parameter buffers (upload once)
  token_embedding : @wgpu.BufferDescriptor
  pos_embedding : @wgpu.BufferDescriptor
  causal_mask : @wgpu.BufferDescriptor
  // Per-layer (need to be created per layer)
  layer_params : Array[TransformerLayerResourcePlan]
  // Final
  ln_final_gamma : @wgpu.BufferDescriptor
  ln_final_beta : @wgpu.BufferDescriptor
  lm_head : @wgpu.BufferDescriptor
  // Activation buffers
  tokens_in : @wgpu.BufferDescriptor
  x : @wgpu.BufferDescriptor
  x_tmp : @wgpu.BufferDescriptor
  x_residual : @wgpu.BufferDescriptor
  q : @wgpu.BufferDescriptor
  k : @wgpu.BufferDescriptor
  v : @wgpu.BufferDescriptor
  attn_out : @wgpu.BufferDescriptor
  attn_proj : @wgpu.BufferDescriptor
  ff_h : @wgpu.BufferDescriptor
  logits : @wgpu.BufferDescriptor
}

///|
pub struct TransformerLayerResourcePlan {
  ln1_gamma : @wgpu.BufferDescriptor
  ln1_beta : @wgpu.BufferDescriptor
  w_q : @wgpu.BufferDescriptor
  w_k : @wgpu.BufferDescriptor
  w_v : @wgpu.BufferDescriptor
  w_o : @wgpu.BufferDescriptor
  ln2_gamma : @wgpu.BufferDescriptor
  ln2_beta : @wgpu.BufferDescriptor
  ff_w1 : @wgpu.BufferDescriptor
  ff_b1 : @wgpu.BufferDescriptor
  ff_w2 : @wgpu.BufferDescriptor
  ff_b2 : @wgpu.BufferDescriptor
}

///|
pub fn transformer_plan_resources(
  spec : TransformerSpec,
) -> TransformerGpuResourcePlan {
  let plan = transformer_plan_buffers(spec)
  let upload = buffer_usage_storage_upload()
  let readback = buffer_usage_storage_readback()
  let layer_params = Array::makei(spec.num_layers, fn(i) {
    let prefix = "tf_l" + i.to_string() + "_"
    TransformerLayerResourcePlan::{
      ln1_gamma: buffer_desc(plan.ln1_gamma_bytes, upload, prefix + "ln1_g"),
      ln1_beta: buffer_desc(plan.ln1_beta_bytes, upload, prefix + "ln1_b"),
      w_q: buffer_desc(plan.w_q_bytes, upload, prefix + "wq"),
      w_k: buffer_desc(plan.w_k_bytes, upload, prefix + "wk"),
      w_v: buffer_desc(plan.w_v_bytes, upload, prefix + "wv"),
      w_o: buffer_desc(plan.w_o_bytes, upload, prefix + "wo"),
      ln2_gamma: buffer_desc(plan.ln2_gamma_bytes, upload, prefix + "ln2_g"),
      ln2_beta: buffer_desc(plan.ln2_beta_bytes, upload, prefix + "ln2_b"),
      ff_w1: buffer_desc(plan.ff_w1_bytes, upload, prefix + "ffw1"),
      ff_b1: buffer_desc(plan.ff_b1_bytes, upload, prefix + "ffb1"),
      ff_w2: buffer_desc(plan.ff_w2_bytes, upload, prefix + "ffw2"),
      ff_b2: buffer_desc(plan.ff_b2_bytes, upload, prefix + "ffb2"),
    }
  })
  TransformerGpuResourcePlan::{
    token_embedding: buffer_desc(
      plan.token_embedding_bytes,
      upload,
      "tf_tok_emb",
    ),
    pos_embedding: buffer_desc(plan.pos_embedding_bytes, upload, "tf_pos_emb"),
    causal_mask: buffer_desc(plan.causal_mask_bytes, upload, "tf_mask"),
    layer_params,
    ln_final_gamma: buffer_desc(
      plan.ln_final_gamma_bytes,
      upload,
      "tf_ln_final_g",
    ),
    ln_final_beta: buffer_desc(
      plan.ln_final_beta_bytes,
      upload,
      "tf_ln_final_b",
    ),
    lm_head: buffer_desc(plan.lm_head_bytes, upload, "tf_lm_head"),
    tokens_in: buffer_desc(plan.tokens_in_bytes, upload, "tf_tokens"),
    x: buffer_desc(plan.x_bytes, readback, "tf_x"),
    x_tmp: buffer_desc(plan.x_tmp_bytes, readback, "tf_x_tmp"),
    x_residual: buffer_desc(plan.x_residual_bytes, readback, "tf_x_res"),
    q: buffer_desc(plan.q_bytes, readback, "tf_q"),
    k: buffer_desc(plan.k_bytes, readback, "tf_k"),
    v: buffer_desc(plan.v_bytes, readback, "tf_v"),
    attn_out: buffer_desc(plan.attn_out_bytes, readback, "tf_attn_out"),
    attn_proj: buffer_desc(plan.attn_proj_bytes, readback, "tf_attn_proj"),
    ff_h: buffer_desc(plan.ff_h_bytes, readback, "tf_ff_h"),
    logits: buffer_desc(plan.logits_bytes, readback, "tf_logits"),
  }
}

///|
/// Dispatch plan for transformer GPU execution
pub struct TransformerGpuDispatchPlan {
  workgroup_size : Int
  // Embedding
  embedding_dispatch_x : Int
  // Element-wise (add, gelu, copy)
  add_pos_dispatch_x : Int
  // Per hidden state element
  x_elementwise_dispatch_x : Int
  // LayerNorm
  layer_norm_dispatch_x : Int
  // Batched matmul (tiled) for attention projections
  attn_proj_dispatch_x : Int
  // Batched matmul for output projection
  out_proj_dispatch_x : Int
  // FFN matmul dispatch
  ffn_up_dispatch_x : Int
  ffn_down_dispatch_x : Int
  // Reshape
  reshape_dispatch_x : Int
  // Attention core
  attention_dispatch_x : Int
  // LM head
  lm_head_dispatch_x : Int
  // FFN element-wise
  ff_h_elementwise_dispatch_x : Int
} derive(Show, Eq)

///|
pub fn transformer_dispatch_plan(
  spec : TransformerSpec,
  workgroup_size : Int,
) -> Result[TransformerGpuDispatchPlan, MlpError] {
  if workgroup_size <= 0 {
    return Err(InvalidSpec("workgroup_size must be > 0"))
  }
  let bs = spec.batch_size * spec.seq_len
  let d_k = transformer_d_k(spec)
  let x_total = bs * spec.d_model
  let q_total = spec.batch_size * spec.num_heads * spec.seq_len * d_k
  let ff_total = bs * spec.d_ff
  Ok(
    TransformerGpuDispatchPlan::{
      workgroup_size,
      embedding_dispatch_x: ceil_div(bs, workgroup_size),
      add_pos_dispatch_x: ceil_div(x_total, workgroup_size),
      x_elementwise_dispatch_x: ceil_div(x_total, workgroup_size),
      layer_norm_dispatch_x: ceil_div(bs, workgroup_size),
      attn_proj_dispatch_x: ceil_div(bs * spec.d_model, workgroup_size),
      out_proj_dispatch_x: ceil_div(bs * spec.d_model, workgroup_size),
      ffn_up_dispatch_x: ceil_div(bs * spec.d_ff, workgroup_size),
      ffn_down_dispatch_x: ceil_div(bs * spec.d_model, workgroup_size),
      reshape_dispatch_x: ceil_div(q_total, workgroup_size),
      attention_dispatch_x: ceil_div(
        spec.batch_size * spec.num_heads * spec.seq_len,
        workgroup_size,
      ),
      lm_head_dispatch_x: ceil_div(bs * spec.vocab_size, workgroup_size),
      ff_h_elementwise_dispatch_x: ceil_div(ff_total, workgroup_size),
    },
  )
}

///|
/// Shader plan for transformer GPU execution
pub struct TransformerGpuShaderPlan {
  embedding_lookup_wgsl : String
  add_vectors_wgsl : String
  layer_norm_wgsl : String
  batched_matmul_attn_wgsl : String // Q/K/V projection: [bs, d_model] @ [d_model, d_model]
  batched_matmul_ffn_up_wgsl : String // FFN up: [bs, d_model] @ [d_model, d_ff]
  batched_matmul_ffn_down_wgsl : String // FFN down: [bs, d_ff] @ [d_ff, d_model]
  batched_matmul_lm_head_wgsl : String // LM head: [bs, d_model] @ [d_model, vocab_size]
  add_bias_ffn_up_wgsl : String
  add_bias_ffn_down_wgsl : String
  gelu_wgsl : String
  reshape_for_heads_wgsl : String
  reshape_from_heads_wgsl : String
  attention_core_wgsl : String
  copy_wgsl : String
} derive(Show, Eq)

///|
pub fn transformer_shader_plan(
  spec : TransformerSpec,
  workgroup_size : Int,
) -> Result[TransformerGpuShaderPlan, MlpError] {
  if workgroup_size <= 0 {
    return Err(InvalidSpec("workgroup_size must be > 0"))
  }
  let bs = spec.batch_size * spec.seq_len
  let d_k = transformer_d_k(spec)
  let x_total = bs * spec.d_model
  let ff_total = bs * spec.d_ff
  Ok(
    TransformerGpuShaderPlan::{
      embedding_lookup_wgsl: wgsl_embedding_lookup(spec, workgroup_size),
      add_vectors_wgsl: wgsl_add_vectors(x_total, workgroup_size),
      layer_norm_wgsl: wgsl_layer_norm(
        bs,
        spec.d_model,
        Float::from_double(0.00001),
        workgroup_size,
      ),
      batched_matmul_attn_wgsl: wgsl_batched_matmul(
        bs,
        spec.d_model,
        spec.d_model,
        workgroup_size,
      ),
      batched_matmul_ffn_up_wgsl: wgsl_batched_matmul(
        bs,
        spec.d_model,
        spec.d_ff,
        workgroup_size,
      ),
      batched_matmul_ffn_down_wgsl: wgsl_batched_matmul(
        bs,
        spec.d_ff,
        spec.d_model,
        workgroup_size,
      ),
      batched_matmul_lm_head_wgsl: wgsl_batched_matmul(
        bs,
        spec.d_model,
        spec.vocab_size,
        workgroup_size,
      ),
      add_bias_ffn_up_wgsl: wgsl_add_bias(bs, spec.d_ff, workgroup_size),
      add_bias_ffn_down_wgsl: wgsl_add_bias(bs, spec.d_model, workgroup_size),
      gelu_wgsl: wgsl_gelu(ff_total, workgroup_size),
      reshape_for_heads_wgsl: wgsl_reshape_for_heads(
        spec.batch_size,
        spec.seq_len,
        spec.num_heads,
        d_k,
        workgroup_size,
      ),
      reshape_from_heads_wgsl: wgsl_reshape_from_heads(
        spec.batch_size,
        spec.seq_len,
        spec.num_heads,
        d_k,
        workgroup_size,
      ),
      attention_core_wgsl: wgsl_attention_core(
        spec.batch_size,
        spec.num_heads,
        spec.seq_len,
        spec.seq_len,
        d_k,
        workgroup_size,
      ),
      copy_wgsl: wgsl_copy(x_total, workgroup_size),
    },
  )
}
