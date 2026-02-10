///|
test "vit_config" {
  let config = vit_config(28, 7, 64, 4, 2, 128, 10).unwrap()
  inspect(config.num_patches, content="16")
  inspect(config.seq_len, content="17")
  inspect(config.patch_dim, content="49")
}

///|
test "vit_config_invalid_patch" {
  let result = vit_config(28, 5, 64, 4, 2, 128, 10)
  assert_true(result is Err(_))
}

///|
test "vit_config_invalid_heads" {
  let result = vit_config(28, 7, 64, 3, 2, 128, 10)
  assert_true(result is Err(_))
}

///|
test "extract_patches" {
  let config = vit_config(4, 2, 8, 2, 1, 16, 10).unwrap()
  // 4x4 image, 2x2 patches -> 4 patches of size 4
  let f = Float::from_int
  // Row-major 4x4 image:
  // [ 0  1  2  3]
  // [ 4  5  6  7]
  // [ 8  9 10 11]
  // [12 13 14 15]
  let img_data = Array::makei(16, fn(i) { f(i) })
  let images = @tensor.tensor_new(@tensor.shape_new([1, 16]).unwrap(), img_data).unwrap()
  let patches = @tensor.extract_patches(images, config)
  inspect(patches.dim(0), content="1")
  inspect(patches.dim(1), content="4") // 4 patches
  inspect(patches.dim(2), content="4") // 2x2 = 4 pixels each
  // Patch (0,0): rows 0-1, cols 0-1 -> [0,1,4,5]
  inspect(patches.at3(0, 0, 0), content="0")
  inspect(patches.at3(0, 0, 1), content="1")
  inspect(patches.at3(0, 0, 2), content="4")
  inspect(patches.at3(0, 0, 3), content="5")
  // Patch (0,1): rows 0-1, cols 2-3 -> [2,3,6,7]
  inspect(patches.at3(0, 1, 0), content="2")
  inspect(patches.at3(0, 1, 1), content="3")
  inspect(patches.at3(0, 1, 2), content="6")
  inspect(patches.at3(0, 1, 3), content="7")
  // Patch (1,0): rows 2-3, cols 0-1 -> [8,9,12,13]
  inspect(patches.at3(0, 2, 0), content="8")
  inspect(patches.at3(0, 2, 1), content="9")
  inspect(patches.at3(0, 2, 2), content="12")
  inspect(patches.at3(0, 2, 3), content="13")
  // Patch (1,1): rows 2-3, cols 2-3 -> [10,11,14,15]
  inspect(patches.at3(0, 3, 0), content="10")
  inspect(patches.at3(0, 3, 1), content="11")
  inspect(patches.at3(0, 3, 2), content="14")
  inspect(patches.at3(0, 3, 3), content="15")
}

///|
test "prepend_cls_token_shape" {
  let f = Float::from_double
  let config = vit_config(28, 7, 8, 2, 1, 16, 10).unwrap()
  let batch = 2
  // x: [2, 16, 8]
  let x_data = Array::make(batch * 16 * 8, f(0.5))
  let x = @tensor.tensor_new(@tensor.shape_new([batch, 16, 8]).unwrap(), x_data).unwrap()
  let cls = @tensor.tensor_new(
    @tensor.shape_new([1, 8]).unwrap(),
    Array::make(8, f(1.0)),
  ).unwrap()
  let out = @tensor.prepend_cls_token(x, cls, batch)
  inspect(out.dim(0), content="2")
  inspect(out.dim(1), content="17") // 16 + 1 CLS
  inspect(out.dim(2), content="8")
  // CLS token at position 0 should be 1.0
  inspect(out.at3(0, 0, 0), content="1")
  inspect(out.at3(1, 0, 0), content="1")
  // Patches at position 1 should be 0.5
  inspect(out.at3(0, 1, 0), content="0.5")
  let _ = config
}

///|
test "vit_forward_shape" {
  let config = vit_config(28, 7, 64, 4, 2, 128, 10).unwrap()
  let params = @tensor.vit_init_params(config, 42)
  let batch = 2
  let images = @tensor.tensor_zeros(@tensor.shape_new([batch, 784]).unwrap())
  let logits = @tensor.vit_forward(images, params, config).unwrap()
  inspect(logits.dim(0), content="2")
  inspect(logits.dim(1), content="10")
}

///|
test "vit_forward_deterministic" {
  let config = vit_config(28, 7, 64, 4, 2, 128, 10).unwrap()
  let params1 = @tensor.vit_init_params(config, 42)
  let params2 = @tensor.vit_init_params(config, 42)
  let images = @tensor.tensor_zeros(@tensor.shape_new([1, 784]).unwrap())
  let logits1 = @tensor.vit_forward(images, params1, config).unwrap()
  let logits2 = @tensor.vit_forward(images, params2, config).unwrap()
  let mut max_diff = 0.0
  for i = 0; i < logits1.numel(); i = i + 1 {
    let diff = (logits1.data[i] - logits2.data[i]).to_double().abs()
    if diff > max_diff {
      max_diff = diff
    }
  }
  assert_true(max_diff < 1.0e-6)
}
