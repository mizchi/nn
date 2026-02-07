///|
/// ViT (Vision Transformer) configuration
pub struct VitConfig {
  image_size : Int // 28 for MNIST
  patch_size : Int // 7
  num_patches : Int // (28/7)^2 = 16
  seq_len : Int // num_patches + 1 (CLS) = 17
  patch_dim : Int // patch_size^2 = 49
  d_model : Int // 64
  num_heads : Int // 4
  num_layers : Int // 2
  d_ff : Int // 128
  num_classes : Int // 10
  eps : Float
}

///|
pub fn vit_config(
  image_size : Int,
  patch_size : Int,
  d_model : Int,
  num_heads : Int,
  num_layers : Int,
  d_ff : Int,
  num_classes : Int,
) -> Result[VitConfig, String] {
  if image_size % patch_size != 0 {
    return Err("image_size must be divisible by patch_size")
  }
  if d_model % num_heads != 0 {
    return Err("d_model must be divisible by num_heads")
  }
  let grid = image_size / patch_size
  let num_patches = grid * grid
  Ok(VitConfig::{
    image_size,
    patch_size,
    num_patches,
    seq_len: num_patches + 1,
    patch_dim: patch_size * patch_size,
    d_model,
    num_heads,
    num_layers,
    d_ff,
    num_classes,
    eps: Float::from_double(0.00001),
  })
}

///|
/// ViT model parameters
pub(all) struct VitParams {
  patch_proj_w : Tensor // [patch_dim, d_model]
  patch_proj_b : Tensor // [d_model]
  cls_token : Tensor // [1, d_model]
  pos_embedding : Tensor // [seq_len, d_model]
  blocks : Array[TransformerBlockParams]
  ln_final_gamma : Tensor // [d_model]
  ln_final_beta : Tensor // [d_model]
  cls_head_w : Tensor // [d_model, num_classes]
  cls_head_b : Tensor // [num_classes]
}

///|
pub fn vit_init_params(config : VitConfig, seed : Int) -> VitParams {
  let mut rng = seed
  let next_float = fn() -> Float {
    rng = rng * 1103515245 + 12345
    let val = (rng / 65536 % 32768).to_double() / 32768.0
    (val * 0.04 - 0.02) |> Float::from_double
  }
  let init_tensor = fn(shape : Shape) -> Tensor {
    let data = Array::makei(shape.numel(), fn(_i) { next_float() })
    Tensor::{ data, shape, offset: 0, strides: shape.strides() }
  }
  let ones = fn(n : Int) -> Tensor { tensor_ones(shape_new_unchecked([n])) }
  let zeros = fn(n : Int) -> Tensor { tensor_zeros(shape_new_unchecked([n])) }
  let d = config.d_model
  let d_ff = config.d_ff
  let patch_proj_w = init_tensor(shape_new_unchecked([config.patch_dim, d]))
  let patch_proj_b = zeros(d)
  let cls_token = init_tensor(shape_new_unchecked([1, d]))
  let pos_embedding = init_tensor(shape_new_unchecked([config.seq_len, d]))
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
  VitParams::{
    patch_proj_w,
    patch_proj_b,
    cls_token,
    pos_embedding,
    blocks,
    ln_final_gamma: ones(d),
    ln_final_beta: zeros(d),
    cls_head_w: init_tensor(shape_new_unchecked([d, config.num_classes])),
    cls_head_b: zeros(config.num_classes),
  }
}

///|
/// Extract patches from flattened images
/// images: [batch, image_size*image_size] -> [batch, num_patches, patch_dim]
pub fn extract_patches(images : Tensor, config : VitConfig) -> Tensor {
  let batch = images.dim(0)
  let ps = config.patch_size
  let grid = config.image_size / ps
  let np = config.num_patches
  let pd = config.patch_dim
  let img_size = config.image_size
  let ic = images.contiguous()
  let result = Array::make(batch * np * pd, Float::from_int(0))
  for b = 0; b < batch; b = b + 1 {
    let img_base = b * img_size * img_size
    for gy = 0; gy < grid; gy = gy + 1 {
      for gx = 0; gx < grid; gx = gx + 1 {
        let patch_idx = gy * grid + gx
        let out_base = b * np * pd + patch_idx * pd
        for py = 0; py < ps; py = py + 1 {
          for px = 0; px < ps; px = px + 1 {
            let row = gy * ps + py
            let col = gx * ps + px
            let pixel_idx = row * img_size + col
            let patch_pixel = py * ps + px
            result[out_base + patch_pixel] = ic.data[img_base + pixel_idx]
          }
        }
      }
    }
  }
  Tensor::{
    data: result,
    shape: shape_new_unchecked([batch, np, pd]),
    offset: 0,
    strides: shape_new_unchecked([batch, np, pd]).strides(),
  }
}

///|
/// Prepend CLS token to patch sequence
/// x: [batch, num_patches, d_model] -> [batch, num_patches+1, d_model]
pub fn prepend_cls_token(x : Tensor, cls_token : Tensor, batch : Int) -> Tensor {
  let num_patches = x.dim(1)
  let d_model = x.dim(2)
  let seq_len = num_patches + 1
  let xc = x.contiguous()
  let cc = cls_token.contiguous()
  let result = Array::make(batch * seq_len * d_model, Float::from_int(0))
  for b = 0; b < batch; b = b + 1 {
    let out_base = b * seq_len * d_model
    // Copy CLS token at position 0
    for d = 0; d < d_model; d = d + 1 {
      result[out_base + d] = cc.data[d]
    }
    // Copy patches at positions 1..num_patches
    let in_base = b * num_patches * d_model
    for s = 0; s < num_patches; s = s + 1 {
      let src = in_base + s * d_model
      let dst = out_base + (s + 1) * d_model
      for d = 0; d < d_model; d = d + 1 {
        result[dst + d] = xc.data[src + d]
      }
    }
  }
  Tensor::{
    data: result,
    shape: shape_new_unchecked([batch, seq_len, d_model]),
    offset: 0,
    strides: shape_new_unchecked([batch, seq_len, d_model]).strides(),
  }
}

///|
/// ViT forward pass (inference)
/// images: [batch, 784] -> logits: [batch, num_classes]
pub fn vit_forward(
  images : Tensor,
  params : VitParams,
  config : VitConfig,
) -> Result[Tensor, String] {
  let batch = images.dim(0)
  // Extract patches: [batch, 16, 49]
  let patches = extract_patches(images, config)
  // Patch embedding: [batch, 16, d_model]
  let patch_embedded = match batched_linear(patches, params.patch_proj_w) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  let patch_embedded = add_bias(patch_embedded, params.patch_proj_b)
  // Prepend CLS: [batch, 17, d_model]
  let x_with_cls = prepend_cls_token(patch_embedded, params.cls_token, batch)
  // Add positional embedding
  let x_with_pos = add_positional_embedding(x_with_cls, params.pos_embedding)
  // Transformer blocks (bidirectional, no mask)
  let transformer_config = TransformerConfig::{
    vocab_size: 0,
    d_model: config.d_model,
    num_heads: config.num_heads,
    num_layers: config.num_layers,
    d_ff: config.d_ff,
    max_seq_len: config.seq_len,
    eps: config.eps,
  }
  let mut hidden = x_with_pos
  for i = 0; i < config.num_layers; i = i + 1 {
    hidden = match
      transformer_block(hidden, params.blocks[i], transformer_config, None) {
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
  // Extract CLS token: [batch, d_model]
  let d_model = config.d_model
  let seq_len = config.seq_len
  let lnc = ln_out.contiguous()
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
    shape: shape_new_unchecked([batch, d_model]),
    offset: 0,
    strides: shape_new_unchecked([batch, d_model]).strides(),
  }
  // Classification head: [batch, d_model] @ [d_model, num_classes] + bias
  let cls_3d = cls_features
    .reshape(shape_new_unchecked([batch, 1, d_model]))
    .unwrap()
  let logits_3d = match batched_linear(cls_3d, params.cls_head_w) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  let logits_3d = add_bias(logits_3d, params.cls_head_b)
  // Reshape [batch, 1, num_classes] -> [batch, num_classes]
  let nc = config.num_classes
  let l3c = logits_3d.contiguous()
  let logits_data = Array::make(batch * nc, Float::from_int(0))
  for i = 0; i < batch * nc; i = i + 1 {
    logits_data[i] = l3c.data[i]
  }
  Ok(Tensor::{
    data: logits_data,
    shape: shape_new_unchecked([batch, nc]),
    offset: 0,
    strides: shape_new_unchecked([batch, nc]).strides(),
  })
}

///|
/// Evaluate ViT on a dataset: returns (loss, accuracy)
pub fn vit_eval(
  images : Tensor,
  params : VitParams,
  config : VitConfig,
  labels : Array[Int],
  count : Int,
) -> Result[(Float, Float), String] {
  let logits = match vit_forward(images, params, config) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  let loss = match tensor_cross_entropy(logits, labels) {
    Ok(l) => l
    Err(e) => return Err(e)
  }
  let nc = config.num_classes
  let lc = logits.contiguous()
  let mut correct = 0
  for b = 0; b < count; b = b + 1 {
    let base = b * nc
    let mut max_idx = 0
    let mut max_val = lc.data[base]
    for c = 1; c < nc; c = c + 1 {
      if lc.data[base + c] > max_val {
        max_val = lc.data[base + c]
        max_idx = c
      }
    }
    if max_idx == labels[b] {
      correct = correct + 1
    }
  }
  let accuracy = Float::from_int(correct) / Float::from_int(count)
  Ok((loss, accuracy))
}
