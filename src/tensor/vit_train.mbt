///|
/// Gradient struct for ViT (mirrors VitParams)
pub struct VitGrads {
  d_patch_proj_w : Tensor // [patch_dim, d_model]
  d_patch_proj_b : Tensor // [d_model]
  d_cls_token : Tensor // [1, d_model]
  d_pos_embedding : Tensor // [seq_len, d_model]
  d_blocks : Array[TransformerBlockGrads]
  d_ln_final_gamma : Tensor // [d_model]
  d_ln_final_beta : Tensor // [d_model]
  d_cls_head_w : Tensor // [d_model, num_classes]
  d_cls_head_b : Tensor // [num_classes]
}

///|
/// Cache for ViT forward pass (intermediate activations for backward)
pub struct VitCache {
  patches : Tensor // [batch, num_patches, patch_dim]
  patch_embedded : Tensor // [batch, num_patches, d_model] after linear+bias
  x_with_cls : Tensor // [batch, seq_len, d_model]
  x_with_pos : Tensor // [batch, seq_len, d_model]
  block_caches : Array[TransformerBlockCache]
  hidden_before_final_ln : Tensor // [batch, seq_len, d_model]
  ln_final_mean : Array[Float]
  ln_final_rstd : Array[Float]
  ln_final_out : Tensor // [batch, seq_len, d_model]
  cls_features : Tensor // [batch, 1, d_model] (reshaped for batched_linear)
  logits : Tensor // [batch, num_classes]
}

///|
pub fn vit_zero_grads(config : VitConfig) -> VitGrads {
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
  VitGrads::{
    d_patch_proj_w: tensor_zeros(shape_new_unchecked([config.patch_dim, d])),
    d_patch_proj_b: tensor_zeros(shape_new_unchecked([d])),
    d_cls_token: tensor_zeros(shape_new_unchecked([1, d])),
    d_pos_embedding: tensor_zeros(shape_new_unchecked([config.seq_len, d])),
    d_blocks: blocks,
    d_ln_final_gamma: tensor_zeros(shape_new_unchecked([d])),
    d_ln_final_beta: tensor_zeros(shape_new_unchecked([d])),
    d_cls_head_w: tensor_zeros(shape_new_unchecked([d, config.num_classes])),
    d_cls_head_b: tensor_zeros(shape_new_unchecked([config.num_classes])),
  }
}

///|
fn vit_zero_grads_inplace(grads : VitGrads) -> Unit {
  zero_tensor(grads.d_patch_proj_w)
  zero_tensor(grads.d_patch_proj_b)
  zero_tensor(grads.d_cls_token)
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
  zero_tensor(grads.d_cls_head_w)
  zero_tensor(grads.d_cls_head_b)
}

///|
/// ViT forward pass with cache for training
pub fn vit_forward_with_cache(
  images : Tensor,
  params : VitParams,
  config : VitConfig,
) -> (Tensor, VitCache) {
  let batch = images.dim(0)
  let d_model = config.d_model
  let seq_len = config.seq_len
  let num_heads = config.num_heads
  let d_k = d_model / num_heads
  // Extract patches: [batch, num_patches, patch_dim]
  let patches = extract_patches(images, config)
  // Patch embedding: [batch, num_patches, d_model]
  let patch_proj = batched_linear(patches, params.patch_proj_w).unwrap()
  let patch_embedded = add_bias(patch_proj, params.patch_proj_b)
  // Prepend CLS: [batch, seq_len, d_model]
  let x_with_cls = prepend_cls_token(patch_embedded, params.cls_token, batch)
  // Add positional embedding
  let x_with_pos = add_positional_embedding(x_with_cls, params.pos_embedding)
  // Transformer blocks with cache
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
    let q_heads = reshape_for_heads(q_proj, batch, seq_len, num_heads, d_k)
    let k_heads = reshape_for_heads(k_proj, batch, seq_len, num_heads, d_k)
    let v_heads = reshape_for_heads(v_proj, batch, seq_len, num_heads, d_k)
    // Attention (no mask for ViT)
    let scale = Float::from_double(1.0 / d_k.to_double().sqrt())
    let qc = q_heads.contiguous()
    let kc = k_heads.contiguous()
    let vc = v_heads.contiguous()
    let attn_weights_all : Array[Array[Float]] = []
    let attn_result_data = Array::make(
      batch * num_heads * seq_len * d_k,
      Float::from_int(0),
    )
    for b = 0; b < batch; b = b + 1 {
      for h = 0; h < num_heads; h = h + 1 {
        let head_base = (b * num_heads + h) * seq_len * d_k
        let scores = Array::make(seq_len * seq_len, Float::from_int(0))
        for i = 0; i < seq_len; i = i + 1 {
          for j = 0; j < seq_len; j = j + 1 {
            let mut dot = Float::from_int(0)
            for d = 0; d < d_k; d = d + 1 {
              dot = dot +
                qc.data[head_base + i * d_k + d] *
                kc.data[head_base + j * d_k + d]
            }
            scores[i * seq_len + j] = dot * scale
          }
        }
        // Softmax
        for i = 0; i < seq_len; i = i + 1 {
          let sbase = i * seq_len
          let mut max_val = scores[sbase]
          for j = 1; j < seq_len; j = j + 1 {
            if scores[sbase + j] > max_val {
              max_val = scores[sbase + j]
            }
          }
          let mut sum = Float::from_int(0)
          for j = 0; j < seq_len; j = j + 1 {
            let e = @math.exp((scores[sbase + j] - max_val).to_double())
              |> Float::from_double
            scores[sbase + j] = e
            sum = sum + e
          }
          for j = 0; j < seq_len; j = j + 1 {
            scores[sbase + j] = scores[sbase + j] / sum
          }
        }
        attn_weights_all.push(scores.copy())
        // Multiply by V
        for i = 0; i < seq_len; i = i + 1 {
          for d = 0; d < d_k; d = d + 1 {
            let mut sum = Float::from_int(0)
            for j = 0; j < seq_len; j = j + 1 {
              sum = sum +
                scores[i * seq_len + j] * vc.data[head_base + j * d_k + d]
            }
            attn_result_data[head_base + i * d_k + d] = sum
          }
        }
      }
    }
    let attn_heads_out = Tensor::{
      data: attn_result_data,
      shape: shape_new_unchecked([batch, num_heads, seq_len, d_k]),
      offset: 0,
      strides: shape_new_unchecked([batch, num_heads, seq_len, d_k]).strides(),
    }
    let attn_concat = reshape_from_heads(
      attn_heads_out, batch, seq_len, num_heads, d_k,
    )
    // W_o projection
    let attn_out = batched_linear(attn_concat, bp.w_o).unwrap()
    // Residual 1
    let x_after_res1_data = Array::make(
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
    let hidden_data = Array::make(batch * seq_len * d_model, Float::from_int(0))
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
  // Extract CLS token: [batch, 1, d_model] for batched_linear
  let lnc = ln_final_out.contiguous()
  let cls_features_data = Array::make(batch * d_model, Float::from_int(0))
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
  // Classification head
  let logits_3d = batched_linear(cls_features, params.cls_head_w).unwrap()
  let logits_3d = add_bias(logits_3d, params.cls_head_b)
  // Reshape [batch, 1, num_classes] -> [batch, num_classes]
  let nc = config.num_classes
  let l3c = logits_3d.contiguous()
  let logits_data = Array::make(batch * nc, Float::from_int(0))
  for i = 0; i < batch * nc; i = i + 1 {
    logits_data[i] = l3c.data[i]
  }
  let logits = Tensor::{
    data: logits_data,
    shape: shape_new_unchecked([batch, nc]),
    offset: 0,
    strides: shape_new_unchecked([batch, nc]).strides(),
  }
  let cache = VitCache::{
    patches,
    patch_embedded,
    x_with_cls,
    x_with_pos,
    block_caches,
    hidden_before_final_ln,
    ln_final_mean,
    ln_final_rstd,
    ln_final_out,
    cls_features,
    logits,
  }
  (logits, cache)
}

///|
/// ViT backward pass
/// Returns the loss value
pub fn vit_backward(
  cache : VitCache,
  params : VitParams,
  config : VitConfig,
  grads : VitGrads,
  labels : Array[Int],
) -> Float {
  let batch = cache.logits.dim(0)
  let d_model = config.d_model
  let seq_len = config.seq_len
  let nc = config.num_classes
  // Compute loss
  let loss = tensor_cross_entropy(cache.logits, labels).unwrap()
  // Cross-entropy backward: d_logits [batch, num_classes]
  let d_logits = cross_entropy_backward(cache.logits, labels)
  // Classification head backward
  // Reshape d_logits [batch, nc] -> [batch, 1, nc]
  let d_logits_3d_data = Array::make(batch * nc, Float::from_int(0))
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
  // cls_head bias backward
  add_bias_backward(d_logits_3d, grads.d_cls_head_b)
  // cls_head linear backward: d_logits_3d [batch,1,nc], cls_features [batch,1,d_model]
  let d_cls_features = batched_linear_backward(
    d_logits_3d,
    cache.cls_features,
    params.cls_head_w,
    grads.d_cls_head_w,
  )
  // CLS extract backward: expand d_cls [batch,1,d_model] -> d_ln_final [batch,seq_len,d_model]
  // Only position 0 gets gradient
  let d_ln_final_data = Array::make(
    batch * seq_len * d_model,
    Float::from_int(0),
  )
  let dcf = d_cls_features.contiguous()
  for b = 0; b < batch; b = b + 1 {
    let src = b * d_model
    let dst = b * seq_len * d_model // position 0
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
  // Final LN backward
  let d_hidden = layer_norm_backward(
    d_ln_final,
    cache.hidden_before_final_ln,
    cache.ln_final_mean,
    cache.ln_final_rstd,
    params.ln_final_gamma,
    grads.d_ln_final_gamma,
    grads.d_ln_final_beta,
  )
  // Blocks backward (reverse order)
  let transformer_config = TransformerConfig::{
    vocab_size: 0,
    d_model: config.d_model,
    num_heads: config.num_heads,
    num_layers: config.num_layers,
    d_ff: config.d_ff,
    max_seq_len: config.seq_len,
    eps: config.eps,
  }
  let mut dy = d_hidden
  for i = config.num_layers - 1; i >= 0; i = i - 1 {
    dy = transformer_block_backward(
      dy,
      cache.block_caches[i],
      params.blocks[i],
      transformer_config,
      grads.d_blocks[i],
    )
  }
  // Positional embedding backward
  positional_embedding_backward(dy, grads.d_pos_embedding, seq_len)
  // CLS token backward: accumulate dy[:,0,:] -> d_cls_token
  let dyc = dy.contiguous()
  for b = 0; b < batch; b = b + 1 {
    let src = b * seq_len * d_model // position 0
    for d = 0; d < d_model; d = d + 1 {
      grads.d_cls_token.data[d] = grads.d_cls_token.data[d] + dyc.data[src + d]
    }
  }
  // Patch embedding backward: dy[:,1:,:] -> [batch, num_patches, d_model]
  let num_patches = config.num_patches
  let d_patch_data = Array::make(
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
  // patch_proj bias backward
  add_bias_backward(d_patch_embedded, grads.d_patch_proj_b)
  // patch_proj linear backward
  let _ = batched_linear_backward(
    d_patch_embedded,
    cache.patches,
    params.patch_proj_w,
    grads.d_patch_proj_w,
  )
  loss
}

///|
/// SGD update for ViT parameters
pub fn vit_sgd_update(params : VitParams, grads : VitGrads, lr : Float) -> Unit {
  sgd_update_tensor(params.patch_proj_w, grads.d_patch_proj_w, lr)
  sgd_update_tensor(params.patch_proj_b, grads.d_patch_proj_b, lr)
  sgd_update_tensor(params.cls_token, grads.d_cls_token, lr)
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
  sgd_update_tensor(params.cls_head_w, grads.d_cls_head_w, lr)
  sgd_update_tensor(params.cls_head_b, grads.d_cls_head_b, lr)
}

///|
/// Train ViT on image classification data
/// images: [count, image_size*image_size], labels: [count]
/// Returns loss history (one value per epoch)
pub fn vit_train(
  images : Tensor,
  labels : Array[Int],
  count : Int,
  config : VitConfig,
  epochs : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
) -> Array[Float] {
  let params = vit_init_params(config, seed)
  let grads = vit_zero_grads(config)
  let loss_history : Array[Float] = []
  let img_dim = config.image_size * config.image_size
  let mut rng = seed
  for _epoch = 0; _epoch < epochs; _epoch = _epoch + 1 {
    // Simple random batch selection
    let bs = if batch_size < count { batch_size } else { count }
    let batch_data = Array::make(bs * img_dim, Float::from_int(0))
    let batch_labels : Array[Int] = Array::make(bs, 0)
    let ic = images.contiguous()
    for b = 0; b < bs; b = b + 1 {
      rng = rng * 1103515245 + 12345
      let idx = (rng / 65536 % 32768).abs() % count
      let src = idx * img_dim
      let dst = b * img_dim
      for d = 0; d < img_dim; d = d + 1 {
        batch_data[dst + d] = ic.data[src + d]
      }
      batch_labels[b] = labels[idx]
    }
    let batch_images = Tensor::{
      data: batch_data,
      shape: shape_new_unchecked([bs, img_dim]),
      offset: 0,
      strides: shape_new_unchecked([bs, img_dim]).strides(),
    }
    vit_zero_grads_inplace(grads)
    let (_, cache) = vit_forward_with_cache(batch_images, params, config)
    let loss = vit_backward(cache, params, config, grads, batch_labels)
    loss_history.push(loss)
    vit_sgd_update(params, grads, lr)
  }
  loss_history
}

///|
/// Train ViT with evaluation metrics per epoch
/// Iterates over all data in mini-batches per epoch
/// Returns (loss, accuracy) per epoch (averaged over batches)
pub fn vit_train_with_eval(
  images : Tensor,
  labels : Array[Int],
  count : Int,
  config : VitConfig,
  epochs : Int,
  batch_size : Int,
  lr : Float,
  seed : Int,
) -> (VitParams, Array[(Float, Float)]) {
  let params = vit_init_params(config, seed)
  let grads = vit_zero_grads(config)
  let metrics : Array[(Float, Float)] = []
  let img_dim = config.image_size * config.image_size
  let nc = config.num_classes
  let ic = images.contiguous()
  let mut rng = seed
  for _epoch = 0; _epoch < epochs; _epoch = _epoch + 1 {
    // Shuffle indices
    let indices = Array::makei(count, fn(i) { i })
    for i = count - 1; i > 0; i = i - 1 {
      rng = rng * 1103515245 + 12345
      let j = (rng / 65536 % 32768).abs() % (i + 1)
      let tmp = indices[i]
      indices[i] = indices[j]
      indices[j] = tmp
    }
    let mut epoch_loss = Float::from_int(0)
    let mut epoch_correct = 0
    let mut epoch_samples = 0
    let mut offset = 0
    while offset < count {
      let bs = if offset + batch_size <= count {
        batch_size
      } else {
        count - offset
      }
      let batch_data = Array::make(bs * img_dim, Float::from_int(0))
      let batch_labels : Array[Int] = Array::make(bs, 0)
      for b = 0; b < bs; b = b + 1 {
        let idx = indices[offset + b]
        let src = idx * img_dim
        let dst = b * img_dim
        for d = 0; d < img_dim; d = d + 1 {
          batch_data[dst + d] = ic.data[src + d]
        }
        batch_labels[b] = labels[idx]
      }
      let batch_images = Tensor::{
        data: batch_data,
        shape: shape_new_unchecked([bs, img_dim]),
        offset: 0,
        strides: shape_new_unchecked([bs, img_dim]).strides(),
      }
      vit_zero_grads_inplace(grads)
      let (logits, cache) = vit_forward_with_cache(batch_images, params, config)
      let loss = vit_backward(cache, params, config, grads, batch_labels)
      vit_sgd_update(params, grads, lr)
      // Accumulate metrics
      epoch_loss = epoch_loss + loss * Float::from_int(bs)
      let lc = logits.contiguous()
      for b = 0; b < bs; b = b + 1 {
        let base = b * nc
        let mut max_idx = 0
        let mut max_val = lc.data[base]
        for c = 1; c < nc; c = c + 1 {
          if lc.data[base + c] > max_val {
            max_val = lc.data[base + c]
            max_idx = c
          }
        }
        if max_idx == batch_labels[b] {
          epoch_correct = epoch_correct + 1
        }
      }
      epoch_samples = epoch_samples + bs
      offset = offset + bs
    }
    let avg_loss = epoch_loss / Float::from_int(epoch_samples)
    let avg_acc = Float::from_int(epoch_correct) / Float::from_int(epoch_samples)
    metrics.push((avg_loss, avg_acc))
  }
  (params, metrics)
}
