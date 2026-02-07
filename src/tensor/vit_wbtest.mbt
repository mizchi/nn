///|
test "vit_forward_with_cache_matches_forward" {
  let config = vit_config(28, 7, 64, 4, 2, 128, 10).unwrap()
  let params = vit_init_params(config, 123)
  let batch = 2
  // Create deterministic test images
  let img_data = Array::makei(batch * 784, fn(i) {
    Float::from_double(i.to_double() / 784.0 * 0.1)
  })
  let images = Tensor::{
    data: img_data,
    shape: shape_new_unchecked([batch, 784]),
    offset: 0,
    strides: shape_new_unchecked([batch, 784]).strides(),
  }
  // Forward (inference)
  let logits_ref = vit_forward(images, params, config).unwrap()
  // Forward with cache (training)
  let (logits_cached, _) = vit_forward_with_cache(images, params, config)
  // Every element must match
  let n = logits_ref.numel()
  assert_true(n == logits_cached.numel())
  let ref_c = logits_ref.contiguous()
  let cac_c = logits_cached.contiguous()
  let mut max_diff = 0.0
  for i = 0; i < n; i = i + 1 {
    let diff = (ref_c.data[i] - cac_c.data[i]).to_double().abs()
    if diff > max_diff {
      max_diff = diff
    }
  }
  assert_true(max_diff < 1.0e-5)
}

///|
test "vit_backward_gradient_check_cls_head_w" {
  // Numerical gradient check for cls_head_w
  let config = vit_config(4, 2, 8, 2, 1, 16, 3).unwrap()
  let params = vit_init_params(config, 42)
  let img_data = Array::makei(2 * 16, fn(i) {
    Float::from_double(i.to_double() / 16.0 * 0.5)
  })
  let images = Tensor::{
    data: img_data,
    shape: shape_new_unchecked([2, 16]),
    offset: 0,
    strides: shape_new_unchecked([2, 16]).strides(),
  }
  let labels = [0, 1]
  // Analytical gradient
  let grads = vit_zero_grads(config)
  let (_, cache) = vit_forward_with_cache(images, params, config)
  let _ = vit_backward(cache, params, config, grads, labels)
  // Numerical gradient for first few elements of cls_head_w
  let eps = Float::from_double(0.001)
  let num_check = 5
  let mut max_rel_err = 0.0
  for idx = 0; idx < num_check; idx = idx + 1 {
    let orig = params.cls_head_w.data[idx]
    params.cls_head_w.data[idx] = orig + eps
    let loss_plus = tensor_cross_entropy(
      vit_forward(images, params, config).unwrap(),
      labels,
    ).unwrap()
    params.cls_head_w.data[idx] = orig - eps
    let loss_minus = tensor_cross_entropy(
      vit_forward(images, params, config).unwrap(),
      labels,
    ).unwrap()
    params.cls_head_w.data[idx] = orig
    let numerical = (loss_plus - loss_minus) / (Float::from_int(2) * eps)
    let analytical = grads.d_cls_head_w.data[idx]
    let err = (analytical - numerical).to_double().abs()
    let denom = analytical.to_double().abs() +
      numerical.to_double().abs() +
      1.0e-8
    let rel_err = err / denom
    if rel_err > max_rel_err {
      max_rel_err = rel_err
    }
  }
  assert_true(max_rel_err < 0.05)
}

///|
test "vit_backward_gradient_check_patch_proj_w" {
  let config = vit_config(4, 2, 8, 2, 1, 16, 3).unwrap()
  let params = vit_init_params(config, 42)
  let img_data = Array::makei(2 * 16, fn(i) {
    Float::from_double(i.to_double() / 16.0 * 0.5)
  })
  let images = Tensor::{
    data: img_data,
    shape: shape_new_unchecked([2, 16]),
    offset: 0,
    strides: shape_new_unchecked([2, 16]).strides(),
  }
  let labels = [0, 1]
  let grads = vit_zero_grads(config)
  let (_, cache) = vit_forward_with_cache(images, params, config)
  let _ = vit_backward(cache, params, config, grads, labels)
  let eps = Float::from_double(0.001)
  let num_check = 5
  let mut max_rel_err = 0.0
  for idx = 0; idx < num_check; idx = idx + 1 {
    let orig = params.patch_proj_w.data[idx]
    params.patch_proj_w.data[idx] = orig + eps
    let loss_plus = tensor_cross_entropy(
      vit_forward(images, params, config).unwrap(),
      labels,
    ).unwrap()
    params.patch_proj_w.data[idx] = orig - eps
    let loss_minus = tensor_cross_entropy(
      vit_forward(images, params, config).unwrap(),
      labels,
    ).unwrap()
    params.patch_proj_w.data[idx] = orig
    let numerical = (loss_plus - loss_minus) / (Float::from_int(2) * eps)
    let analytical = grads.d_patch_proj_w.data[idx]
    let err = (analytical - numerical).to_double().abs()
    let denom = analytical.to_double().abs() +
      numerical.to_double().abs() +
      1.0e-8
    let rel_err = err / denom
    if rel_err > max_rel_err {
      max_rel_err = rel_err
    }
  }
  assert_true(max_rel_err < 0.05)
}

///|
test "vit_training_loss_decreases" {
  // Small config for fast test
  let config = vit_config(4, 2, 8, 2, 1, 16, 3).unwrap()
  // 4x4 images, 3 classes, 6 samples (2 per class)
  let num_samples = 6
  let img_dim = 16
  let img_data = Array::make(num_samples * img_dim, Float::from_int(0))
  // Class 0: bright top-left
  for s = 0; s < 2; s = s + 1 {
    let base = s * img_dim
    img_data[base + 0] = Float::from_double(1.0)
    img_data[base + 1] = Float::from_double(1.0)
    img_data[base + 4] = Float::from_double(1.0)
    img_data[base + 5] = Float::from_double(1.0)
  }
  // Class 1: bright top-right
  for s = 2; s < 4; s = s + 1 {
    let base = s * img_dim
    img_data[base + 2] = Float::from_double(1.0)
    img_data[base + 3] = Float::from_double(1.0)
    img_data[base + 6] = Float::from_double(1.0)
    img_data[base + 7] = Float::from_double(1.0)
  }
  // Class 2: bright bottom
  for s = 4; s < 6; s = s + 1 {
    let base = s * img_dim
    img_data[base + 8] = Float::from_double(1.0)
    img_data[base + 9] = Float::from_double(1.0)
    img_data[base + 12] = Float::from_double(1.0)
    img_data[base + 13] = Float::from_double(1.0)
  }
  let images = Tensor::{
    data: img_data,
    shape: shape_new_unchecked([num_samples, img_dim]),
    offset: 0,
    strides: shape_new_unchecked([num_samples, img_dim]).strides(),
  }
  let labels = [0, 0, 1, 1, 2, 2]
  // Train with direct forward/backward loop (full batch, deterministic)
  let params = vit_init_params(config, 42)
  let grads = vit_zero_grads(config)
  let lr = Float::from_double(0.05)
  let num_epochs = 50
  let losses : Array[Float] = []
  for _epoch = 0; _epoch < num_epochs; _epoch = _epoch + 1 {
    vit_zero_grads_inplace(grads)
    let (_, cache) = vit_forward_with_cache(images, params, config)
    let loss = vit_backward(cache, params, config, grads, labels)
    losses.push(loss)
    vit_sgd_update(params, grads, lr)
  }
  assert_true(losses.length() == num_epochs)
  let first_loss = losses[0]
  let last_loss = losses[losses.length() - 1]
  // Loss should decrease
  assert_true(last_loss.to_double() < first_loss.to_double())
}
