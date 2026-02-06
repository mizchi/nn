///|
test "gelu_backward_numerical" {
  let x_data : Array[Float] = [
    Float::from_double(-1.5),
    Float::from_double(-0.5),
    Float::from_double(0.0),
    Float::from_double(0.5),
    Float::from_double(1.5),
  ]
  let x = tensor_from_array(x_data)
  let dy = tensor_from_array(Array::make(5, Float::from_int(1)))
  let analytical = gelu_backward(dy, x)
  let eps = Float::from_double(0.001)
  for i = 0; i < 5; i = i + 1 {
    let x_plus = x_data.copy()
    x_plus[i] = x_plus[i] + eps
    let x_minus = x_data.copy()
    x_minus[i] = x_minus[i] - eps
    let y_plus = tensor_gelu(tensor_from_array(x_plus))
    let y_minus = tensor_gelu(tensor_from_array(x_minus))
    let numerical = (y_plus.at1(i) - y_minus.at1(i)) /
      (Float::from_int(2) * eps)
    let diff = (analytical.at1(i) - numerical).to_double().abs()
    assert_true(diff < 0.01)
  }
}

///|
test "layer_norm_backward_numerical" {
  let x_data : Array[Float] = [
    Float::from_double(1.0),
    Float::from_double(2.0),
    Float::from_double(3.0),
    Float::from_double(4.0),
  ]
  let x = Tensor::{
    data: x_data,
    shape: shape_new_unchecked([2, 2]),
    offset: 0,
    strides: [2, 1],
  }
  let gamma = tensor_from_array([
    Float::from_double(1.0),
    Float::from_double(1.0),
  ])
  let beta = tensor_from_array([
    Float::from_double(0.0),
    Float::from_double(0.0),
  ])
  let eps_ln = Float::from_double(0.00001)
  let dy_data : Array[Float] = [
    Float::from_double(1.0),
    Float::from_double(0.5),
    Float::from_double(-0.5),
    Float::from_double(1.0),
  ]
  let dy = Tensor::{
    data: dy_data,
    shape: shape_new_unchecked([2, 2]),
    offset: 0,
    strides: [2, 1],
  }
  let (_, mean, rstd) = layer_norm_with_cache(x, gamma, beta, eps_ln)
  let d_gamma = tensor_zeros(shape_new_unchecked([2]))
  let d_beta = tensor_zeros(shape_new_unchecked([2]))
  let dx = layer_norm_backward(dy, x, mean, rstd, gamma, d_gamma, d_beta)
  let eps = Float::from_double(0.001)
  for idx = 0; idx < 4; idx = idx + 1 {
    let x_plus_data = x_data.copy()
    x_plus_data[idx] = x_plus_data[idx] + eps
    let x_plus = Tensor::{
      data: x_plus_data,
      shape: shape_new_unchecked([2, 2]),
      offset: 0,
      strides: [2, 1],
    }
    let x_minus_data = x_data.copy()
    x_minus_data[idx] = x_minus_data[idx] - eps
    let x_minus = Tensor::{
      data: x_minus_data,
      shape: shape_new_unchecked([2, 2]),
      offset: 0,
      strides: [2, 1],
    }
    let y_plus = tensor_layer_norm(x_plus, gamma, beta, eps_ln).unwrap()
    let y_minus = tensor_layer_norm(x_minus, gamma, beta, eps_ln).unwrap()
    let mut loss_plus = Float::from_int(0)
    let mut loss_minus = Float::from_int(0)
    for j = 0; j < 4; j = j + 1 {
      loss_plus = loss_plus + dy_data[j] * y_plus.data[j]
      loss_minus = loss_minus + dy_data[j] * y_minus.data[j]
    }
    let numerical = (loss_plus - loss_minus) / (Float::from_int(2) * eps)
    let diff = (dx.data[idx] - numerical).to_double().abs()
    assert_true(diff < 0.01)
  }
}

///|
test "batched_linear_backward_numerical" {
  let x_data : Array[Float] = [
    Float::from_double(0.1),
    Float::from_double(0.2),
    Float::from_double(0.3),
    Float::from_double(0.4),
    Float::from_double(0.5),
    Float::from_double(0.6),
  ]
  let x = Tensor::{
    data: x_data,
    shape: shape_new_unchecked([1, 2, 3]),
    offset: 0,
    strides: [6, 3, 1],
  }
  let w_data : Array[Float] = [
    Float::from_double(0.1),
    Float::from_double(0.2),
    Float::from_double(0.3),
    Float::from_double(0.4),
    Float::from_double(0.5),
    Float::from_double(0.6),
  ]
  let w = Tensor::{
    data: w_data,
    shape: shape_new_unchecked([3, 2]),
    offset: 0,
    strides: [2, 1],
  }
  let dy_data : Array[Float] = [
    Float::from_double(1.0),
    Float::from_double(0.5),
    Float::from_double(-0.5),
    Float::from_double(1.0),
  ]
  let dy = Tensor::{
    data: dy_data,
    shape: shape_new_unchecked([1, 2, 2]),
    offset: 0,
    strides: [4, 2, 1],
  }
  let d_w = tensor_zeros(shape_new_unchecked([3, 2]))
  let dx = batched_linear_backward(dy, x, w, d_w)
  let eps = Float::from_double(0.001)
  for idx = 0; idx < 6; idx = idx + 1 {
    let xp = x_data.copy()
    xp[idx] = xp[idx] + eps
    let xm = x_data.copy()
    xm[idx] = xm[idx] - eps
    let x_plus = Tensor::{
      data: xp,
      shape: shape_new_unchecked([1, 2, 3]),
      offset: 0,
      strides: [6, 3, 1],
    }
    let x_minus = Tensor::{
      data: xm,
      shape: shape_new_unchecked([1, 2, 3]),
      offset: 0,
      strides: [6, 3, 1],
    }
    let y_plus = batched_linear(x_plus, w).unwrap()
    let y_minus = batched_linear(x_minus, w).unwrap()
    let mut loss_plus = Float::from_int(0)
    let mut loss_minus = Float::from_int(0)
    for j = 0; j < 4; j = j + 1 {
      loss_plus = loss_plus + dy_data[j] * y_plus.data[j]
      loss_minus = loss_minus + dy_data[j] * y_minus.data[j]
    }
    let numerical = (loss_plus - loss_minus) / (Float::from_int(2) * eps)
    let diff = (dx.data[idx] - numerical).to_double().abs()
    assert_true(diff < 0.01)
  }
}

///|
test "cross_entropy_backward_shape" {
  let logits_data = Array::make(8, Float::from_double(0.25))
  let logits = Tensor::{
    data: logits_data,
    shape: shape_new_unchecked([2, 4]),
    offset: 0,
    strides: [4, 1],
  }
  let labels = [1, 3]
  let d_logits = cross_entropy_backward(logits, labels)
  inspect(d_logits.dim(0), content="2")
  inspect(d_logits.dim(1), content="4")
  let mut row0_sum = Float::from_int(0)
  for v = 0; v < 4; v = v + 1 {
    row0_sum = row0_sum + d_logits.at2(0, v)
  }
  let sum_abs = row0_sum.to_double().abs()
  assert_true(sum_abs < 0.001)
}

///|
test "forward_with_cache_matches_forward" {
  let config = transformer_config(4, 8, 1, 1, 16, 8).unwrap()
  let params = transformer_init_params(config, 123)
  let tokens = [[0, 1, 2, 3, 1, 2, 3, 0]]
  let mask = causal_mask(8).unwrap()
  // forward (inference)
  let logits_ref = transformer_forward(tokens, params, config, Some(mask)).unwrap()
  // forward_with_cache (training)
  let (logits_cached, _) = transformer_forward_with_cache(
    tokens,
    params,
    config,
    Some(mask),
  )
  // Every element must match
  let n = logits_ref.numel()
  inspect(n, content=logits_cached.numel().to_string())
  let ref_c = logits_ref.contiguous()
  let cac_c = logits_cached.contiguous()
  let mut max_diff = 0.0
  for i = 0; i < n; i = i + 1 {
    let diff = (ref_c.data[i] - cac_c.data[i]).to_double().abs()
    if diff > max_diff {
      max_diff = diff
    }
  }
  // Float32 ops should be identical (same computation order)
  assert_true(max_diff < 1.0e-5)
}

///|
test "training_loss_decreases" {
  let text = "abcabcabcabcabcabc"
  let tokenizer = char_tokenizer_from_text(text)
  let vocab = tokenizer.vocab_size()
  let config = transformer_config(vocab, 8, 1, 1, 16, 8).unwrap()
  let losses = transformer_train(text, config, 20, Float::from_double(0.01), 42)
  assert_true(losses.length() > 0)
  let first_loss = losses[0]
  let last_loss = losses[losses.length() - 1]
  let reduction = (first_loss - last_loss) / first_loss
  assert_true(reduction.to_double() > 0.3)
}

///|
test "trained_model_generates_learned_pattern" {
  let text = "abcabcabcabcabcabcabcabcabc"
  let tokenizer = char_tokenizer_from_text(text)
  let vocab = tokenizer.vocab_size()
  // max_seq_len=16 so we have room to generate after a short prompt
  let config = transformer_config(vocab, 8, 1, 1, 16, 16).unwrap()
  let params = transformer_init_params(config, 42)
  let grads = transformer_zero_grads(config)
  let all_ids = tokenizer.encode(text)
  let seq_len = 8 // training window size (< max_seq_len)
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
  let mask = causal_mask(seq_len).unwrap()
  for _epoch = 0; _epoch < 50; _epoch = _epoch + 1 {
    zero_grads(grads)
    let (_, cache) = transformer_forward_with_cache(
      inputs,
      params,
      config,
      Some(mask),
    )
    let _ = transformer_backward(cache, params, config, grads, targets)
    sgd_update(params, grads, Float::from_double(0.01))
  }
  // Generate from "abcab" prompt — next should be 'c' (pattern: abc repeating)
  let prompt = tokenizer.encode("abcab")
  let generated = generate_greedy(params, config, prompt, 3).unwrap()
  let output = tokenizer.decode(generated)
  let chars = output.to_array()
  // prompt is 5 chars, generated 3 more -> 8 total
  assert_true(chars.length() == 8)
  // After "abcab", the pattern demands 'c'
  assert_true(chars[5] == 'c')
}
