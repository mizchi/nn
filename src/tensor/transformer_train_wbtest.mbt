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
  let x = tensor_new(shape_new_unchecked([2, 2]), x_data).unwrap()
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
  let dy = tensor_new(shape_new_unchecked([2, 2]), dy_data).unwrap()
  let (_, mean, rstd) = layer_norm_with_cache(x, gamma, beta, eps_ln)
  let d_gamma = tensor_zeros(shape_new_unchecked([2]))
  let d_beta = tensor_zeros(shape_new_unchecked([2]))
  let dx = layer_norm_backward(dy, x, mean, rstd, gamma, d_gamma, d_beta)
  let eps = Float::from_double(0.001)
  for idx = 0; idx < 4; idx = idx + 1 {
    let x_plus_data = x_data.copy()
    x_plus_data[idx] = x_plus_data[idx] + eps
    let x_plus = tensor_new(shape_new_unchecked([2, 2]), x_plus_data).unwrap()
    let x_minus_data = x_data.copy()
    x_minus_data[idx] = x_minus_data[idx] - eps
    let x_minus = tensor_new(shape_new_unchecked([2, 2]), x_minus_data).unwrap()
    let y_plus = tensor_layer_norm(x_plus, gamma, beta, eps_ln).unwrap()
    let y_minus = tensor_layer_norm(x_minus, gamma, beta, eps_ln).unwrap()
    let mut loss_plus = Float::from_int(0)
    let mut loss_minus = Float::from_int(0)
    for j = 0; j < 4; j = j + 1 {
      loss_plus = loss_plus + dy.data[j] * y_plus.data[j]
      loss_minus = loss_minus + dy.data[j] * y_minus.data[j]
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
  let x = tensor_new(shape_new_unchecked([1, 2, 3]), x_data).unwrap()
  let w_data : Array[Float] = [
    Float::from_double(0.1),
    Float::from_double(0.2),
    Float::from_double(0.3),
    Float::from_double(0.4),
    Float::from_double(0.5),
    Float::from_double(0.6),
  ]
  let w = tensor_new(shape_new_unchecked([3, 2]), w_data).unwrap()
  let dy_data : Array[Float] = [
    Float::from_double(1.0),
    Float::from_double(0.5),
    Float::from_double(-0.5),
    Float::from_double(1.0),
  ]
  let dy = tensor_new(shape_new_unchecked([1, 2, 2]), dy_data).unwrap()
  let d_w = tensor_zeros(shape_new_unchecked([3, 2]))
  let dx = batched_linear_backward(dy, x, w, d_w)
  let eps = Float::from_double(0.001)
  for idx = 0; idx < 6; idx = idx + 1 {
    let xp = x_data.copy()
    xp[idx] = xp[idx] + eps
    let xm = x_data.copy()
    xm[idx] = xm[idx] - eps
    let x_plus = tensor_new(shape_new_unchecked([1, 2, 3]), xp).unwrap()
    let x_minus = tensor_new(shape_new_unchecked([1, 2, 3]), xm).unwrap()
    let y_plus = batched_linear(x_plus, w).unwrap()
    let y_minus = batched_linear(x_minus, w).unwrap()
    let mut loss_plus = Float::from_int(0)
    let mut loss_minus = Float::from_int(0)
    for j = 0; j < 4; j = j + 1 {
      loss_plus = loss_plus + dy.data[j] * y_plus.data[j]
      loss_minus = loss_minus + dy.data[j] * y_minus.data[j]
    }
    let numerical = (loss_plus - loss_minus) / (Float::from_int(2) * eps)
    let diff = (dx.data[idx] - numerical).to_double().abs()
    assert_true(diff < 0.01)
  }
}

///|
test "add_bias_backward_accumulates" {
  let dy = tensor_new(shape_new_unchecked([2, 3]), [
    Float::from_double(1.0),
    Float::from_double(2.0),
    Float::from_double(3.0),
    Float::from_double(4.0),
    Float::from_double(5.0),
    Float::from_double(6.0),
  ]).unwrap()
  let d_bias = tensor_new(shape_new_unchecked([3]), [
    Float::from_double(0.5),
    Float::from_double(-0.5),
    Float::from_double(1.0),
  ]).unwrap()
  add_bias_backward(dy, d_bias)
  add_bias_backward(dy, d_bias)
  let e0 = Float::from_double(10.5)
  let e1 = Float::from_double(13.5)
  let e2 = Float::from_double(19.0)
  assert_true((d_bias.data[0] - e0).to_double().abs() < 1.0e-6)
  assert_true((d_bias.data[1] - e1).to_double().abs() < 1.0e-6)
  assert_true((d_bias.data[2] - e2).to_double().abs() < 1.0e-6)
}

///|
test "softmax_backward_rows_matches_reference" {
  let rows = 2
  let cols = 3
  let attn = FixedArray::make(rows * cols, Float::from_int(0))
  let d_attn = FixedArray::make(rows * cols, Float::from_int(0))
  attn[0] = Float::from_double(0.2)
  attn[1] = Float::from_double(0.3)
  attn[2] = Float::from_double(0.5)
  attn[3] = Float::from_double(0.1)
  attn[4] = Float::from_double(0.2)
  attn[5] = Float::from_double(0.7)
  d_attn[0] = Float::from_double(1.0)
  d_attn[1] = Float::from_double(-0.5)
  d_attn[2] = Float::from_double(0.25)
  d_attn[3] = Float::from_double(-1.0)
  d_attn[4] = Float::from_double(0.4)
  d_attn[5] = Float::from_double(0.6)
  let expected = FixedArray::make(rows * cols, Float::from_int(0))
  for i = 0; i < rows; i = i + 1 {
    let mut dot_sum = Float::from_int(0)
    for j = 0; j < cols; j = j + 1 {
      dot_sum = dot_sum + d_attn[i * cols + j] * attn[i * cols + j]
    }
    for j = 0; j < cols; j = j + 1 {
      expected[i * cols + j] = attn[i * cols + j] *
        (d_attn[i * cols + j] - dot_sum)
    }
  }
  let actual = FixedArray::make(rows * cols, Float::from_int(0))
  softmax_backward_rows(d_attn, attn, actual, rows, cols)
  for i = 0; i < rows * cols; i = i + 1 {
    let diff = (actual[i] - expected[i]).to_double().abs()
    assert_true(diff < 1.0e-6)
  }
}

///|
test "add_matrix_inplace_matches_reference" {
  let rows = 3
  let cols = 4
  let base = FixedArray::make(rows * cols, Float::from_int(0))
  let add = FixedArray::make(rows * cols, Float::from_int(0))
  for i = 0; i < rows * cols; i = i + 1 {
    base[i] = Float::from_int(i - 4) / Float::from_int(7)
    add[i] = Float::from_int(i * 3 % 11 - 5) / Float::from_int(9)
  }
  let expected = FixedArray::make(rows * cols, Float::from_int(0))
  for i = 0; i < rows * cols; i = i + 1 {
    expected[i] = base[i] + add[i]
  }
  add_matrix_inplace(base, add, rows, cols)
  for i = 0; i < rows * cols; i = i + 1 {
    let diff = (base[i] - expected[i]).to_double().abs()
    assert_true(diff < 1.0e-6)
  }
}

///|
test "ffn_backward_fused_matches_reference" {
  let config = transformer_config(8, 8, 1, 1, 16, 4).unwrap()
  let params = transformer_init_params(config, 123)
  let tokens = [[0, 1, 2, 3], [1, 2, 3, 4]]
  let mask = causal_mask(4).unwrap()
  let (_, cache_all) = transformer_forward_with_cache(
    tokens,
    params,
    config,
    Some(mask),
  )
  let block_cache = cache_all.block_caches[0]
  let block_params = params.blocks[0]
  let dy_data = Array::makei(2 * 4 * 8, fn(i) {
    Float::from_int(i % 13 - 6) / Float::from_int(11)
  })
  let dy = tensor_new(shape_new_unchecked([2, 4, 8]), dy_data).unwrap()
  let grads_ref = TransformerBlockGrads::{
    d_ln1_gamma: tensor_zeros(shape_new_unchecked([8])),
    d_ln1_beta: tensor_zeros(shape_new_unchecked([8])),
    d_w_q: tensor_zeros(shape_new_unchecked([8, 8])),
    d_w_k: tensor_zeros(shape_new_unchecked([8, 8])),
    d_w_v: tensor_zeros(shape_new_unchecked([8, 8])),
    d_w_o: tensor_zeros(shape_new_unchecked([8, 8])),
    d_ln2_gamma: tensor_zeros(shape_new_unchecked([8])),
    d_ln2_beta: tensor_zeros(shape_new_unchecked([8])),
    d_ff_w1: tensor_zeros(shape_new_unchecked([8, 16])),
    d_ff_b1: tensor_zeros(shape_new_unchecked([16])),
    d_ff_w2: tensor_zeros(shape_new_unchecked([16, 8])),
    d_ff_b2: tensor_zeros(shape_new_unchecked([8])),
  }
  let grads_fused = TransformerBlockGrads::{
    d_ln1_gamma: tensor_zeros(shape_new_unchecked([8])),
    d_ln1_beta: tensor_zeros(shape_new_unchecked([8])),
    d_w_q: tensor_zeros(shape_new_unchecked([8, 8])),
    d_w_k: tensor_zeros(shape_new_unchecked([8, 8])),
    d_w_v: tensor_zeros(shape_new_unchecked([8, 8])),
    d_w_o: tensor_zeros(shape_new_unchecked([8, 8])),
    d_ln2_gamma: tensor_zeros(shape_new_unchecked([8])),
    d_ln2_beta: tensor_zeros(shape_new_unchecked([8])),
    d_ff_w1: tensor_zeros(shape_new_unchecked([8, 16])),
    d_ff_b1: tensor_zeros(shape_new_unchecked([16])),
    d_ff_w2: tensor_zeros(shape_new_unchecked([16, 8])),
    d_ff_b2: tensor_zeros(shape_new_unchecked([8])),
  }
  let dx_ref = feed_forward_backward_reference(
    dy, block_cache, block_params, grads_ref,
  )
  let dx_fused = feed_forward_backward_fused(
    dy, block_cache, block_params, grads_fused,
  )
  let mut max_diff = 0.0
  for i = 0; i < dx_ref.numel(); i = i + 1 {
    let diff = (dx_ref.data[i] - dx_fused.data[i]).to_double().abs()
    if diff > max_diff {
      max_diff = diff
    }
  }
  assert_true(max_diff < 1.0e-5)
  let mut grad_diff = 0.0
  for i = 0; i < grads_ref.d_ff_w1.numel(); i = i + 1 {
    let diff = (grads_ref.d_ff_w1.data[i] - grads_fused.d_ff_w1.data[i])
      .to_double()
      .abs()
    if diff > grad_diff {
      grad_diff = diff
    }
  }
  for i = 0; i < grads_ref.d_ff_w2.numel(); i = i + 1 {
    let diff = (grads_ref.d_ff_w2.data[i] - grads_fused.d_ff_w2.data[i])
      .to_double()
      .abs()
    if diff > grad_diff {
      grad_diff = diff
    }
  }
  for i = 0; i < grads_ref.d_ff_b1.numel(); i = i + 1 {
    let diff = (grads_ref.d_ff_b1.data[i] - grads_fused.d_ff_b1.data[i])
      .to_double()
      .abs()
    if diff > grad_diff {
      grad_diff = diff
    }
  }
  for i = 0; i < grads_ref.d_ff_b2.numel(); i = i + 1 {
    let diff = (grads_ref.d_ff_b2.data[i] - grads_fused.d_ff_b2.data[i])
      .to_double()
      .abs()
    if diff > grad_diff {
      grad_diff = diff
    }
  }
  assert_true(grad_diff < 1.0e-5)
}

///|
test "cross_entropy_backward_shape" {
  let logits = tensor_new(
    shape_new_unchecked([2, 4]),
    Array::make(8, Float::from_double(0.25)),
  ).unwrap()
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
test "cross_entropy_forward_backward_matches_reference" {
  let logits = tensor_new(
    shape_new_unchecked([3, 5]),
    Array::makei(15, fn(i) { Float::from_int(i % 7 - 3) / Float::from_int(5) }),
  ).unwrap()
  let labels = [0, 3, 1]
  let ref_loss = tensor_cross_entropy(logits, labels).unwrap()
  let ref_grad = cross_entropy_backward(logits, labels)
  let (fused_loss, fused_grad) = cross_entropy_forward_backward(logits, labels)
  let loss_diff = (ref_loss - fused_loss).to_double().abs()
  assert_true(loss_diff < 1.0e-6)
  let mut max_grad_diff = 0.0
  for i = 0; i < ref_grad.numel(); i = i + 1 {
    let diff = (ref_grad.data[i] - fused_grad.data[i]).to_double().abs()
    if diff > max_grad_diff {
      max_grad_diff = diff
    }
  }
  assert_true(max_grad_diff < 1.0e-6)
}

///|
test "transformer_backward_lm_matches_last_token_when_seq_is_one" {
  let config = transformer_config(8, 8, 1, 1, 16, 1).unwrap()
  let params = transformer_init_params(config, 123)
  let grads_last = transformer_zero_grads(config)
  let grads_lm = transformer_zero_grads(config)
  let tokens = [[1], [2]]
  let labels_last = [3, 4]
  let labels_lm = [[3], [4]]
  let (logits_last, cache_last) = transformer_forward_with_cache(
    tokens,
    params,
    config,
    None,
  )
  ignore(logits_last)
  let loss_last = transformer_backward(
    cache_last, params, config, grads_last, labels_last,
  )
  let (logits_lm, cache_lm) = transformer_forward_with_cache(
    tokens,
    params,
    config,
    None,
  )
  ignore(logits_lm)
  let loss_lm = transformer_backward_lm(
    cache_lm, params, config, grads_lm, labels_lm,
  )
  let loss_diff = (loss_last - loss_lm).to_double().abs()
  assert_true(loss_diff < 1.0e-6)
  let mut max_grad_diff = 0.0
  for i = 0; i < grads_last.d_lm_head.numel(); i = i + 1 {
    let diff = (grads_last.d_lm_head.data[i] - grads_lm.d_lm_head.data[i])
      .to_double()
      .abs()
    if diff > max_grad_diff {
      max_grad_diff = diff
    }
  }
  assert_true(max_grad_diff < 1.0e-6)
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
test "attention_cache_uses_fixedarray" {
  let config = transformer_config(4, 8, 1, 1, 16, 8).unwrap()
  let params = transformer_init_params(config, 123)
  let tokens = [[0, 1, 2, 3, 1, 2, 3, 0]]
  let mask = causal_mask(8).unwrap()
  let (_, cache) = transformer_forward_with_cache(
    tokens,
    params,
    config,
    Some(mask),
  )
  let w0 : FixedArray[Float] = cache.block_caches[0].attn_weights[0]
  inspect(w0.length(), content="64")
}

///|
test "training_lm_minibatch_loss_decreases" {
  let text = "abcabcabcabcabcabcabcabcabcabcabcabc"
  let tokenizer = char_tokenizer_from_text(text)
  let vocab = tokenizer.vocab_size()
  let config = transformer_config(vocab, 8, 1, 1, 16, 8).unwrap()
  let losses = transformer_train_lm_steps(
    text,
    config,
    40,
    8,
    Float::from_double(0.01),
    42,
  )
  assert_true(losses.length() == 40)
  let mut head = Float::from_int(0)
  let mut tail = Float::from_int(0)
  for i = 0; i < 5; i = i + 1 {
    head = head + losses[i]
    tail = tail + losses[losses.length() - 1 - i]
  }
  head = head / Float::from_int(5)
  tail = tail / Float::from_int(5)
  assert_true(tail.to_double() < head.to_double())
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
