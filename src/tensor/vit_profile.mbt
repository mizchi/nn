///|
pub struct VitProfileResult {
  // Forward phases
  fwd_extract_patches_ns : UInt64
  fwd_patch_linear_ns : UInt64
  fwd_cls_pos_ns : UInt64
  fwd_ln1_ns : UInt64
  fwd_qkv_linear_ns : UInt64
  fwd_reshape_heads_ns : UInt64
  fwd_attn_scores_ns : UInt64
  fwd_softmax_ns : UInt64
  fwd_attn_v_ns : UInt64
  fwd_reshape_back_ns : UInt64
  fwd_wo_linear_ns : UInt64
  fwd_residual1_ns : UInt64
  fwd_ln2_ns : UInt64
  fwd_ff_linear_ns : UInt64
  fwd_gelu_ns : UInt64
  fwd_residual2_ns : UInt64
  fwd_final_ln_cls_ns : UInt64
  // Backward phases
  bwd_loss_ce_ns : UInt64
  bwd_cls_head_ns : UInt64
  bwd_final_ln_ns : UInt64
  bwd_ffn_ns : UInt64
  bwd_ln2_ns : UInt64
  bwd_mha_ns : UInt64
  bwd_ln1_ns : UInt64
  bwd_residual_ns : UInt64
  bwd_patch_ns : UInt64
  // SGD update
  sgd_update_ns : UInt64
  // Total
  total_ns : UInt64
}

///|
pub fn vit_profile_result_print(r : VitProfileResult) -> Unit {
  let to_us = fn(ns : UInt64) -> String {
    let us = ns / 1000UL
    let frac = ns % 1000UL / 100UL
    us.to_string() + "." + frac.to_string()
  }
  let to_ms = fn(ns : UInt64) -> String {
    let ms = ns / 1000000UL
    let frac = ns % 1000000UL / 100000UL
    ms.to_string() + "." + frac.to_string()
  }
  let bwd_total = r.bwd_loss_ce_ns +
    r.bwd_cls_head_ns +
    r.bwd_final_ln_ns +
    r.bwd_ffn_ns +
    r.bwd_ln2_ns +
    r.bwd_mha_ns +
    r.bwd_ln1_ns +
    r.bwd_residual_ns +
    r.bwd_patch_ns
  println("=== ViT Profile (per step) ===")
  println("[Forward]")
  println("  extract_patches:  " + to_us(r.fwd_extract_patches_ns) + " us")
  println("  patch_linear:     " + to_us(r.fwd_patch_linear_ns) + " us")
  println("  cls+pos:          " + to_us(r.fwd_cls_pos_ns) + " us")
  println("  ln1:              " + to_us(r.fwd_ln1_ns) + " us")
  println("  qkv_linear:       " + to_us(r.fwd_qkv_linear_ns) + " us")
  println("  reshape_heads:    " + to_us(r.fwd_reshape_heads_ns) + " us")
  println("  attn_scores:      " + to_us(r.fwd_attn_scores_ns) + " us")
  println("  softmax:          " + to_us(r.fwd_softmax_ns) + " us")
  println("  attn_v:           " + to_us(r.fwd_attn_v_ns) + " us")
  println("  reshape_back:     " + to_us(r.fwd_reshape_back_ns) + " us")
  println("  wo_linear:        " + to_us(r.fwd_wo_linear_ns) + " us")
  println("  residual1:        " + to_us(r.fwd_residual1_ns) + " us")
  println("  ln2:              " + to_us(r.fwd_ln2_ns) + " us")
  println("  ff_linear:        " + to_us(r.fwd_ff_linear_ns) + " us")
  println("  gelu:             " + to_us(r.fwd_gelu_ns) + " us")
  println("  residual2:        " + to_us(r.fwd_residual2_ns) + " us")
  println("  final_ln+cls:     " + to_us(r.fwd_final_ln_cls_ns) + " us")
  println("[Backward] total=" + to_us(bwd_total) + " us")
  println("  loss+ce_bwd:      " + to_us(r.bwd_loss_ce_ns) + " us")
  println("  cls_head_bwd:     " + to_us(r.bwd_cls_head_ns) + " us")
  println("  final_ln_bwd:     " + to_us(r.bwd_final_ln_ns) + " us")
  println("  ffn_bwd:          " + to_us(r.bwd_ffn_ns) + " us")
  println("  ln2_bwd:          " + to_us(r.bwd_ln2_ns) + " us")
  println("  mha_bwd:          " + to_us(r.bwd_mha_ns) + " us")
  println("  ln1_bwd:          " + to_us(r.bwd_ln1_ns) + " us")
  println("  residual_bwd:     " + to_us(r.bwd_residual_ns) + " us")
  println("  patch_bwd:        " + to_us(r.bwd_patch_ns) + " us")
  println("[SGD Update]")
  println("  total:            " + to_us(r.sgd_update_ns) + " us")
  println("[Total]")
  println("  step:             " + to_ms(r.total_ns) + " ms")
}

///|
pub fn vit_profile_step(
  images : Tensor,
  labels : Array[Int],
  params : VitParams,
  grads : VitGrads,
  config : VitConfig,
  lr : Float,
) -> VitProfileResult {
  let batch = images.dim(0)
  let d_model = config.d_model
  let seq_len = config.seq_len
  let num_heads = config.num_heads
  let d_k = d_model / num_heads
  let nc = config.num_classes
  // Forward accumulators
  let mut fwd_extract_patches_ns = 0UL
  let mut fwd_patch_linear_ns = 0UL
  let mut fwd_cls_pos_ns = 0UL
  let mut fwd_ln1_ns = 0UL
  let mut fwd_qkv_linear_ns = 0UL
  let mut fwd_reshape_heads_ns = 0UL
  let mut fwd_attn_scores_ns = 0UL
  let mut fwd_softmax_ns = 0UL
  let mut fwd_attn_v_ns = 0UL
  let mut fwd_reshape_back_ns = 0UL
  let mut fwd_wo_linear_ns = 0UL
  let mut fwd_residual1_ns = 0UL
  let mut fwd_ln2_ns = 0UL
  let mut fwd_ff_linear_ns = 0UL
  let mut fwd_gelu_ns = 0UL
  let mut fwd_residual2_ns = 0UL
  let mut fwd_final_ln_cls_ns = 0UL
  // Backward accumulators
  let mut bwd_loss_ce_ns = 0UL
  let mut bwd_cls_head_ns = 0UL
  let mut bwd_final_ln_ns = 0UL
  let mut bwd_ffn_ns = 0UL
  let mut bwd_ln2_ns = 0UL
  let mut bwd_mha_ns = 0UL
  let mut bwd_ln1_ns = 0UL
  let mut bwd_residual_ns = 0UL
  let mut bwd_patch_ns = 0UL
  let t_total_start = clock_ns()
  // === Forward === (same as before)
  let t0 = clock_ns()
  let patches = extract_patches(images, config)
  let t1 = clock_ns()
  fwd_extract_patches_ns = t1 - t0
  let t0 = clock_ns()
  let patch_proj = batched_linear(patches, params.patch_proj_w).unwrap()
  let patch_embedded = add_bias(patch_proj, params.patch_proj_b)
  let t1 = clock_ns()
  fwd_patch_linear_ns = t1 - t0
  let t0 = clock_ns()
  let x_with_cls = prepend_cls_token(patch_embedded, params.cls_token, batch)
  let x_with_pos = add_positional_embedding(x_with_cls, params.pos_embedding)
  let t1 = clock_ns()
  fwd_cls_pos_ns = t1 - t0
  let mut hidden = x_with_pos
  let block_caches : Array[TransformerBlockCache] = []
  for layer = 0; layer < config.num_layers; layer = layer + 1 {
    let bp = params.blocks[layer]
    let x_input = hidden
    let t0 = clock_ns()
    let (ln1_out, ln1_mean, ln1_rstd) = layer_norm_with_cache(
      x_input,
      bp.ln1_gamma,
      bp.ln1_beta,
      config.eps,
    )
    let t1 = clock_ns()
    fwd_ln1_ns = fwd_ln1_ns + t1 - t0
    let t0 = clock_ns()
    let q_proj = batched_linear(ln1_out, bp.w_q).unwrap()
    let k_proj = batched_linear(ln1_out, bp.w_k).unwrap()
    let v_proj = batched_linear(ln1_out, bp.w_v).unwrap()
    let t1 = clock_ns()
    fwd_qkv_linear_ns = fwd_qkv_linear_ns + t1 - t0
    let t0 = clock_ns()
    let q_heads = reshape_for_heads(q_proj, batch, seq_len, num_heads, d_k)
    let k_heads = reshape_for_heads(k_proj, batch, seq_len, num_heads, d_k)
    let v_heads = reshape_for_heads(v_proj, batch, seq_len, num_heads, d_k)
    let t1 = clock_ns()
    fwd_reshape_heads_ns = fwd_reshape_heads_ns + t1 - t0
    let scale = Float::from_double(1.0 / d_k.to_double().sqrt())
    let qc = q_heads.contiguous()
    let kc = k_heads.contiguous()
    let vc = v_heads.contiguous()
    let total_heads = batch * num_heads
    let scores_size = seq_len * seq_len
    let attn_weights_all : Array[Array[Float]] = Array::make(total_heads, [])
    let scores = FixedArray::make(scores_size, Float::from_int(0))
    let attn_result_data = FixedArray::make(
      batch * num_heads * seq_len * d_k,
      Float::from_int(0),
    )
    for b = 0; b < batch; b = b + 1 {
      for h = 0; h < num_heads; h = h + 1 {
        let head_idx = b * num_heads + h
        let head_base = head_idx * seq_len * d_k
        let t0 = clock_ns()
        native_sgemm_offset(
          0,
          1,
          seq_len,
          seq_len,
          d_k,
          scale,
          qc.data,
          head_base,
          d_k,
          kc.data,
          head_base,
          d_k,
          Float::from_int(0),
          scores,
          0,
          seq_len,
        )
        let t1 = clock_ns()
        fwd_attn_scores_ns = fwd_attn_scores_ns + t1 - t0
        let t0 = clock_ns()
        native_softmax_inplace(scores, seq_len, seq_len)
        let t1 = clock_ns()
        fwd_softmax_ns = fwd_softmax_ns + t1 - t0
        let scores_copy = Array::make(scores_size, Float::from_int(0))
        for idx = 0; idx < scores_size; idx = idx + 1 {
          scores_copy[idx] = scores[idx]
        }
        attn_weights_all[head_idx] = scores_copy
        let t0 = clock_ns()
        native_sgemm_offset(
          0,
          0,
          seq_len,
          d_k,
          seq_len,
          Float::from_int(1),
          scores,
          0,
          seq_len,
          vc.data,
          head_base,
          d_k,
          Float::from_int(0),
          attn_result_data,
          head_base,
          d_k,
        )
        let t1 = clock_ns()
        fwd_attn_v_ns = fwd_attn_v_ns + t1 - t0
      }
    }
    let attn_heads_out = Tensor::{
      data: attn_result_data,
      shape: shape_new_unchecked([batch, num_heads, seq_len, d_k]),
      offset: 0,
      strides: shape_new_unchecked([batch, num_heads, seq_len, d_k]).strides(),
    }
    let t0 = clock_ns()
    let attn_concat = reshape_from_heads(
      attn_heads_out, batch, seq_len, num_heads, d_k,
    )
    let t1 = clock_ns()
    fwd_reshape_back_ns = fwd_reshape_back_ns + t1 - t0
    let t0 = clock_ns()
    let attn_out = batched_linear(attn_concat, bp.w_o).unwrap()
    let t1 = clock_ns()
    fwd_wo_linear_ns = fwd_wo_linear_ns + t1 - t0
    let t0 = clock_ns()
    let x_after_res1_data = FixedArray::make(
      batch * seq_len * d_model,
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
    let t1 = clock_ns()
    fwd_residual1_ns = fwd_residual1_ns + t1 - t0
    let t0 = clock_ns()
    let (ln2_out, ln2_mean, ln2_rstd) = layer_norm_with_cache(
      x_after_res1,
      bp.ln2_gamma,
      bp.ln2_beta,
      config.eps,
    )
    let t1 = clock_ns()
    fwd_ln2_ns = fwd_ln2_ns + t1 - t0
    let t0 = clock_ns()
    let ff_h = batched_linear(ln2_out, bp.ff_w1).unwrap()
    let ff_pre_gelu = add_bias(ff_h, bp.ff_b1)
    let t1 = clock_ns()
    fwd_ff_linear_ns = fwd_ff_linear_ns + t1 - t0
    let t0 = clock_ns()
    let ff_post_gelu = tensor_gelu(ff_pre_gelu)
    let t1 = clock_ns()
    fwd_gelu_ns = fwd_gelu_ns + t1 - t0
    let t0 = clock_ns()
    let ff_out_h = batched_linear(ff_post_gelu, bp.ff_w2).unwrap()
    let ff_out = add_bias(ff_out_h, bp.ff_b2)
    let t1 = clock_ns()
    fwd_ff_linear_ns = fwd_ff_linear_ns + t1 - t0
    let t0 = clock_ns()
    let hidden_data = FixedArray::make(
      batch * seq_len * d_model,
      Float::from_int(0),
    )
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
    let t1 = clock_ns()
    fwd_residual2_ns = fwd_residual2_ns + t1 - t0
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
  let t0 = clock_ns()
  let hidden_before_final_ln = hidden
  let (ln_final_out, ln_final_mean, ln_final_rstd) = layer_norm_with_cache(
    hidden_before_final_ln,
    params.ln_final_gamma,
    params.ln_final_beta,
    config.eps,
  )
  let lnc = ln_final_out.contiguous()
  let cls_features_data = FixedArray::make(batch * d_model, Float::from_int(0))
  for b = 0; b < batch; b = b + 1 {
    let src = b * seq_len * d_model
    let dst = b * d_model
    for d = 0; d < d_model; d = d + 1 {
      cls_features_data[dst + d] = lnc.data[src + d]
    }
  }
  let cls_features = Tensor::{
    data: cls_features_data,
    shape: shape_new_unchecked([batch, 1, d_model]),
    offset: 0,
    strides: shape_new_unchecked([batch, 1, d_model]).strides(),
  }
  let logits_3d = batched_linear(cls_features, params.cls_head_w).unwrap()
  let logits_3d = add_bias(logits_3d, params.cls_head_b)
  let l3c = logits_3d.contiguous()
  let logits_data = FixedArray::make(batch * nc, Float::from_int(0))
  for i = 0; i < batch * nc; i = i + 1 {
    logits_data[i] = l3c.data[i]
  }
  let logits = Tensor::{
    data: logits_data,
    shape: shape_new_unchecked([batch, nc]),
    offset: 0,
    strides: shape_new_unchecked([batch, nc]).strides(),
  }
  let t1 = clock_ns()
  fwd_final_ln_cls_ns = t1 - t0
  // === Backward (instrumented) ===
  // Loss + cross-entropy backward
  let t0 = clock_ns()
  let _loss = tensor_cross_entropy(logits, labels).unwrap()
  let d_logits = cross_entropy_backward(logits, labels)
  let t1 = clock_ns()
  bwd_loss_ce_ns = t1 - t0
  // Classification head backward
  let t0 = clock_ns()
  let d_logits_3d_data = FixedArray::make(batch * nc, Float::from_int(0))
  let dlc = d_logits.contiguous()
  for i = 0; i < batch * nc; i = i + 1 {
    d_logits_3d_data[i] = dlc.data[i]
  }
  let d_logits_3d = Tensor::{
    data: d_logits_3d_data,
    shape: shape_new_unchecked([batch, 1, nc]),
    offset: 0,
    strides: shape_new_unchecked([batch, 1, nc]).strides(),
  }
  add_bias_backward(d_logits_3d, grads.d_cls_head_b)
  let d_cls_features = batched_linear_backward(
    d_logits_3d,
    cls_features,
    params.cls_head_w,
    grads.d_cls_head_w,
  )
  let d_ln_final_data = FixedArray::make(
    batch * seq_len * d_model,
    Float::from_int(0),
  )
  let dcf = d_cls_features.contiguous()
  for b = 0; b < batch; b = b + 1 {
    let src = b * d_model
    let dst = b * seq_len * d_model
    for d = 0; d < d_model; d = d + 1 {
      d_ln_final_data[dst + d] = dcf.data[src + d]
    }
  }
  let d_ln_final = Tensor::{
    data: d_ln_final_data,
    shape: shape_new_unchecked([batch, seq_len, d_model]),
    offset: 0,
    strides: shape_new_unchecked([batch, seq_len, d_model]).strides(),
  }
  let t1 = clock_ns()
  bwd_cls_head_ns = t1 - t0
  // Final LN backward
  let t0 = clock_ns()
  let d_hidden = layer_norm_backward(
    d_ln_final,
    hidden_before_final_ln,
    ln_final_mean,
    ln_final_rstd,
    params.ln_final_gamma,
    grads.d_ln_final_gamma,
    grads.d_ln_final_beta,
  )
  let t1 = clock_ns()
  bwd_final_ln_ns = t1 - t0
  // Blocks backward (reverse order)
  let mut dy = d_hidden
  for i = config.num_layers - 1; i >= 0; i = i - 1 {
    let cache_b = block_caches[i]
    let bp = params.blocks[i]
    let bg = grads.d_blocks[i]
    let n = dy.numel()
    // FFN backward
    let t0 = clock_ns()
    let d_ln2 = feed_forward_backward(dy, cache_b, bp, bg)
    let t1 = clock_ns()
    bwd_ffn_ns = bwd_ffn_ns + t1 - t0
    // LN2 backward
    let t0 = clock_ns()
    let d_res1_from_ffn = layer_norm_backward(
      d_ln2,
      cache_b.x_after_res1,
      cache_b.ln2_mean,
      cache_b.ln2_rstd,
      bp.ln2_gamma,
      bg.d_ln2_gamma,
      bg.d_ln2_beta,
    )
    let t1 = clock_ns()
    bwd_ln2_ns = bwd_ln2_ns + t1 - t0
    // Residual2 add
    let t0 = clock_ns()
    let d_res1_data = FixedArray::make(n, Float::from_int(0))
    for j = 0; j < n; j = j + 1 {
      d_res1_data[j] = dy.data[j] + d_res1_from_ffn.data[j]
    }
    let d_res1 = Tensor::{
      data: d_res1_data,
      shape: dy.shape,
      offset: 0,
      strides: dy.shape.strides(),
    }
    let t1 = clock_ns()
    bwd_residual_ns = bwd_residual_ns + t1 - t0
    // MHA backward
    let t0 = clock_ns()
    let d_ln1 = multi_head_attention_backward(
      d_res1,
      cache_b,
      bp,
      config_to_transformer(config),
      bg,
    )
    let t1 = clock_ns()
    bwd_mha_ns = bwd_mha_ns + t1 - t0
    // LN1 backward
    let t0 = clock_ns()
    let d_input_from_mha = layer_norm_backward(
      d_ln1,
      cache_b.x_input,
      cache_b.ln1_mean,
      cache_b.ln1_rstd,
      bp.ln1_gamma,
      bg.d_ln1_gamma,
      bg.d_ln1_beta,
    )
    let t1 = clock_ns()
    bwd_ln1_ns = bwd_ln1_ns + t1 - t0
    // Residual1 add
    let t0 = clock_ns()
    let dx_data = FixedArray::make(n, Float::from_int(0))
    for j = 0; j < n; j = j + 1 {
      dx_data[j] = d_res1.data[j] + d_input_from_mha.data[j]
    }
    dy = Tensor::{
      data: dx_data,
      shape: dy.shape,
      offset: 0,
      strides: dy.shape.strides(),
    }
    let t1 = clock_ns()
    bwd_residual_ns = bwd_residual_ns + t1 - t0
  }
  // Patch embedding backward
  let t0 = clock_ns()
  positional_embedding_backward(dy, grads.d_pos_embedding, seq_len)
  let dyc = dy.contiguous()
  for b = 0; b < batch; b = b + 1 {
    let src = b * seq_len * d_model
    for d = 0; d < d_model; d = d + 1 {
      grads.d_cls_token.data[d] = grads.d_cls_token.data[d] + dyc.data[src + d]
    }
  }
  let num_patches = config.num_patches
  let d_patch_data = FixedArray::make(
    batch * num_patches * d_model,
    Float::from_int(0),
  )
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < num_patches; s = s + 1 {
      let src = b * seq_len * d_model + (s + 1) * d_model
      let dst = b * num_patches * d_model + s * d_model
      for d = 0; d < d_model; d = d + 1 {
        d_patch_data[dst + d] = dyc.data[src + d]
      }
    }
  }
  let d_patch_embedded = Tensor::{
    data: d_patch_data,
    shape: shape_new_unchecked([batch, num_patches, d_model]),
    offset: 0,
    strides: shape_new_unchecked([batch, num_patches, d_model]).strides(),
  }
  add_bias_backward(d_patch_embedded, grads.d_patch_proj_b)
  let _ = batched_linear_backward(
    d_patch_embedded,
    patches,
    params.patch_proj_w,
    grads.d_patch_proj_w,
  )
  let t1 = clock_ns()
  bwd_patch_ns = t1 - t0
  // === SGD Update ===
  let t0 = clock_ns()
  vit_sgd_update(params, grads, lr)
  let t1 = clock_ns()
  let sgd_update_ns = t1 - t0
  let t_total_end = clock_ns()
  VitProfileResult::{
    fwd_extract_patches_ns,
    fwd_patch_linear_ns,
    fwd_cls_pos_ns,
    fwd_ln1_ns,
    fwd_qkv_linear_ns,
    fwd_reshape_heads_ns,
    fwd_attn_scores_ns,
    fwd_softmax_ns,
    fwd_attn_v_ns,
    fwd_reshape_back_ns,
    fwd_wo_linear_ns,
    fwd_residual1_ns,
    fwd_ln2_ns,
    fwd_ff_linear_ns,
    fwd_gelu_ns,
    fwd_residual2_ns,
    fwd_final_ln_cls_ns,
    bwd_loss_ce_ns,
    bwd_cls_head_ns,
    bwd_final_ln_ns,
    bwd_ffn_ns,
    bwd_ln2_ns,
    bwd_mha_ns,
    bwd_ln1_ns,
    bwd_residual_ns,
    bwd_patch_ns,
    sgd_update_ns,
    total_ns: t_total_end - t_total_start,
  }
}

///|
fn config_to_transformer(config : VitConfig) -> TransformerConfig {
  TransformerConfig::{
    vocab_size: 0,
    d_model: config.d_model,
    num_heads: config.num_heads,
    num_layers: config.num_layers,
    d_ff: config.d_ff,
    max_seq_len: config.seq_len,
    eps: config.eps,
  }
}

///|
pub fn vit_profile_aggregate(
  results : Array[VitProfileResult],
) -> VitProfileResult {
  let n = results.length().to_uint64()
  let mut sum_fwd_extract_patches = 0UL
  let mut sum_fwd_patch_linear = 0UL
  let mut sum_fwd_cls_pos = 0UL
  let mut sum_fwd_ln1 = 0UL
  let mut sum_fwd_qkv_linear = 0UL
  let mut sum_fwd_reshape_heads = 0UL
  let mut sum_fwd_attn_scores = 0UL
  let mut sum_fwd_softmax = 0UL
  let mut sum_fwd_attn_v = 0UL
  let mut sum_fwd_reshape_back = 0UL
  let mut sum_fwd_wo_linear = 0UL
  let mut sum_fwd_residual1 = 0UL
  let mut sum_fwd_ln2 = 0UL
  let mut sum_fwd_ff_linear = 0UL
  let mut sum_fwd_gelu = 0UL
  let mut sum_fwd_residual2 = 0UL
  let mut sum_fwd_final_ln_cls = 0UL
  let mut sum_bwd_loss_ce = 0UL
  let mut sum_bwd_cls_head = 0UL
  let mut sum_bwd_final_ln = 0UL
  let mut sum_bwd_ffn = 0UL
  let mut sum_bwd_ln2 = 0UL
  let mut sum_bwd_mha = 0UL
  let mut sum_bwd_ln1 = 0UL
  let mut sum_bwd_residual = 0UL
  let mut sum_bwd_patch = 0UL
  let mut sum_sgd_update = 0UL
  let mut sum_total = 0UL
  for i = 0; i < results.length(); i = i + 1 {
    let s = results[i]
    sum_fwd_extract_patches = sum_fwd_extract_patches + s.fwd_extract_patches_ns
    sum_fwd_patch_linear = sum_fwd_patch_linear + s.fwd_patch_linear_ns
    sum_fwd_cls_pos = sum_fwd_cls_pos + s.fwd_cls_pos_ns
    sum_fwd_ln1 = sum_fwd_ln1 + s.fwd_ln1_ns
    sum_fwd_qkv_linear = sum_fwd_qkv_linear + s.fwd_qkv_linear_ns
    sum_fwd_reshape_heads = sum_fwd_reshape_heads + s.fwd_reshape_heads_ns
    sum_fwd_attn_scores = sum_fwd_attn_scores + s.fwd_attn_scores_ns
    sum_fwd_softmax = sum_fwd_softmax + s.fwd_softmax_ns
    sum_fwd_attn_v = sum_fwd_attn_v + s.fwd_attn_v_ns
    sum_fwd_reshape_back = sum_fwd_reshape_back + s.fwd_reshape_back_ns
    sum_fwd_wo_linear = sum_fwd_wo_linear + s.fwd_wo_linear_ns
    sum_fwd_residual1 = sum_fwd_residual1 + s.fwd_residual1_ns
    sum_fwd_ln2 = sum_fwd_ln2 + s.fwd_ln2_ns
    sum_fwd_ff_linear = sum_fwd_ff_linear + s.fwd_ff_linear_ns
    sum_fwd_gelu = sum_fwd_gelu + s.fwd_gelu_ns
    sum_fwd_residual2 = sum_fwd_residual2 + s.fwd_residual2_ns
    sum_fwd_final_ln_cls = sum_fwd_final_ln_cls + s.fwd_final_ln_cls_ns
    sum_bwd_loss_ce = sum_bwd_loss_ce + s.bwd_loss_ce_ns
    sum_bwd_cls_head = sum_bwd_cls_head + s.bwd_cls_head_ns
    sum_bwd_final_ln = sum_bwd_final_ln + s.bwd_final_ln_ns
    sum_bwd_ffn = sum_bwd_ffn + s.bwd_ffn_ns
    sum_bwd_ln2 = sum_bwd_ln2 + s.bwd_ln2_ns
    sum_bwd_mha = sum_bwd_mha + s.bwd_mha_ns
    sum_bwd_ln1 = sum_bwd_ln1 + s.bwd_ln1_ns
    sum_bwd_residual = sum_bwd_residual + s.bwd_residual_ns
    sum_bwd_patch = sum_bwd_patch + s.bwd_patch_ns
    sum_sgd_update = sum_sgd_update + s.sgd_update_ns
    sum_total = sum_total + s.total_ns
  }
  VitProfileResult::{
    fwd_extract_patches_ns: sum_fwd_extract_patches / n,
    fwd_patch_linear_ns: sum_fwd_patch_linear / n,
    fwd_cls_pos_ns: sum_fwd_cls_pos / n,
    fwd_ln1_ns: sum_fwd_ln1 / n,
    fwd_qkv_linear_ns: sum_fwd_qkv_linear / n,
    fwd_reshape_heads_ns: sum_fwd_reshape_heads / n,
    fwd_attn_scores_ns: sum_fwd_attn_scores / n,
    fwd_softmax_ns: sum_fwd_softmax / n,
    fwd_attn_v_ns: sum_fwd_attn_v / n,
    fwd_reshape_back_ns: sum_fwd_reshape_back / n,
    fwd_wo_linear_ns: sum_fwd_wo_linear / n,
    fwd_residual1_ns: sum_fwd_residual1 / n,
    fwd_ln2_ns: sum_fwd_ln2 / n,
    fwd_ff_linear_ns: sum_fwd_ff_linear / n,
    fwd_gelu_ns: sum_fwd_gelu / n,
    fwd_residual2_ns: sum_fwd_residual2 / n,
    fwd_final_ln_cls_ns: sum_fwd_final_ln_cls / n,
    bwd_loss_ce_ns: sum_bwd_loss_ce / n,
    bwd_cls_head_ns: sum_bwd_cls_head / n,
    bwd_final_ln_ns: sum_bwd_final_ln / n,
    bwd_ffn_ns: sum_bwd_ffn / n,
    bwd_ln2_ns: sum_bwd_ln2 / n,
    bwd_mha_ns: sum_bwd_mha / n,
    bwd_ln1_ns: sum_bwd_ln1 / n,
    bwd_residual_ns: sum_bwd_residual / n,
    bwd_patch_ns: sum_bwd_patch / n,
    sgd_update_ns: sum_sgd_update / n,
    total_ns: sum_total / n,
  }
}
