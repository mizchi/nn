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
test "layer_norm_backward_add_matches_reference" {
  let batch = 2
  let seq = 3
  let d_model = 4
  let n = batch * seq * d_model
  let x_data = Array::makei(n, fn(i) {
    Float::from_int(i * 5 % 19 - 9) / Float::from_int(7)
  })
  let dy_data = Array::makei(n, fn(i) {
    Float::from_int(i * 3 % 17 - 8) / Float::from_int(9)
  })
  let residual_data = Array::makei(n, fn(i) {
    Float::from_int(i * 7 % 23 - 11) / Float::from_int(11)
  })
  let gamma_data = Array::makei(d_model, fn(i) {
    Float::from_int(i * 2 % 7 + 1) / Float::from_int(5)
  })
  let beta_data = Array::makei(d_model, fn(i) {
    Float::from_int(i * 3 % 11 - 5) / Float::from_int(13)
  })
  let x = tensor_new(shape_new_unchecked([batch, seq, d_model]), x_data).unwrap()
  let dy = tensor_new(shape_new_unchecked([batch, seq, d_model]), dy_data).unwrap()
  let residual = tensor_new(
    shape_new_unchecked([batch, seq, d_model]),
    residual_data,
  ).unwrap()
  let gamma = tensor_new(shape_new_unchecked([d_model]), gamma_data).unwrap()
  let beta = tensor_new(shape_new_unchecked([d_model]), beta_data).unwrap()
  let eps_ln = Float::from_double(0.00001)
  let (_, mean, rstd) = layer_norm_with_cache(x, gamma, beta, eps_ln)
  let d_gamma_ref = tensor_zeros(shape_new_unchecked([d_model]))
  let d_beta_ref = tensor_zeros(shape_new_unchecked([d_model]))
  let dx_ref = layer_norm_backward(
    dy, x, mean, rstd, gamma, d_gamma_ref, d_beta_ref,
  )
  let out_ref_data = FixedArray::make(n, Float::from_int(0))
  for i = 0; i < n; i = i + 1 {
    out_ref_data[i] = residual.data[i] + dx_ref.data[i]
  }
  let d_gamma_fused = tensor_zeros(shape_new_unchecked([d_model]))
  let d_beta_fused = tensor_zeros(shape_new_unchecked([d_model]))
  let out_fused = layer_norm_backward_add_inplace(
    dy, residual, x, mean, rstd, gamma, d_gamma_fused, d_beta_fused,
  )
  for i = 0; i < n; i = i + 1 {
    let diff = (out_ref_data[i] - out_fused.data[i]).to_double().abs()
    assert_true(diff < 1.0e-6)
  }
  for i = 0; i < d_model; i = i + 1 {
    let dg_diff = (d_gamma_ref.data[i] - d_gamma_fused.data[i])
      .to_double()
      .abs()
    let db_diff = (d_beta_ref.data[i] - d_beta_fused.data[i]).to_double().abs()
    assert_true(dg_diff < 1.0e-6)
    assert_true(db_diff < 1.0e-6)
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
test "batched_linear_backward_fused_matches_reference" {
  let batch = 2
  let seq = 3
  let in_dim = 4
  let out_dim = 5
  let n = batch * seq
  let x_data = Array::makei(n * in_dim, fn(i) {
    Float::from_int(i * 5 % 19 - 9) / Float::from_int(11)
  })
  let w_data = Array::makei(in_dim * out_dim, fn(i) {
    Float::from_int(i * 7 % 23 - 11) / Float::from_int(13)
  })
  let dy_data = Array::makei(n * out_dim, fn(i) {
    Float::from_int(i * 3 % 17 - 8) / Float::from_int(9)
  })
  let x = tensor_new(shape_new_unchecked([batch, seq, in_dim]), x_data).unwrap()
  let w = tensor_new(shape_new_unchecked([in_dim, out_dim]), w_data).unwrap()
  let dy = tensor_new(shape_new_unchecked([batch, seq, out_dim]), dy_data).unwrap()
  let d_w_ref = tensor_zeros(shape_new_unchecked([in_dim, out_dim]))
  let dx_ref = batched_linear_backward(dy, x, w, d_w_ref)
  let d_w_fused = tensor_zeros(shape_new_unchecked([in_dim, out_dim]))
  let dx_fused = batched_linear_backward_fused(dy, x, w, d_w_fused)
  let mut max_dx_diff = 0.0
  let mut max_dw_diff = 0.0
  for i = 0; i < dx_ref.numel(); i = i + 1 {
    let diff = (dx_ref.data[i] - dx_fused.data[i]).to_double().abs()
    if diff > max_dx_diff {
      max_dx_diff = diff
    }
  }
  for i = 0; i < d_w_ref.numel(); i = i + 1 {
    let diff = (d_w_ref.data[i] - d_w_fused.data[i]).to_double().abs()
    if diff > max_dw_diff {
      max_dw_diff = diff
    }
  }
  assert_true(max_dx_diff < 1.0e-6)
  assert_true(max_dw_diff < 1.0e-6)
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
test "zero_inplace_sets_all_values_to_zero" {
  let data = FixedArray::make(17, Float::from_int(0))
  for i = 0; i < data.length(); i = i + 1 {
    data[i] = Float::from_int(i * 7 % 19 - 9) / Float::from_int(5)
  }
  zero_inplace(data, data.length())
  for i = 0; i < data.length(); i = i + 1 {
    let diff = (data[i] - Float::from_int(0)).to_double().abs()
    assert_true(diff < 1.0e-8)
  }
}

///|
test "adamw_step_inplace_matches_reference" {
  let n = 13
  let lr = Float::from_double(0.001)
  let beta1 = Float::from_double(0.9)
  let beta2 = Float::from_double(0.999)
  let eps = Float::from_double(0.00000001)
  let weight_decay = Float::from_double(0.01)
  let one = Float::from_int(1)
  let bias_c1 = one - beta1
  let bias_c2 = one - beta2
  let param = FixedArray::make(n, Float::from_int(0))
  let grad = FixedArray::make(n, Float::from_int(0))
  let m = FixedArray::make(n, Float::from_int(0))
  let v = FixedArray::make(n, Float::from_int(0))
  for i = 0; i < n; i = i + 1 {
    param[i] = Float::from_int(i * 5 % 17 - 8) / Float::from_int(7)
    grad[i] = Float::from_int(i * 3 % 13 - 6) / Float::from_int(9)
    m[i] = Float::from_int(i * 2 % 7 - 3) / Float::from_int(11)
    v[i] = Float::from_int(i * 4 % 19 + 1) / Float::from_int(100)
  }
  let p_ref = FixedArray::make(n, Float::from_int(0))
  let m_ref = FixedArray::make(n, Float::from_int(0))
  let v_ref = FixedArray::make(n, Float::from_int(0))
  for i = 0; i < n; i = i + 1 {
    p_ref[i] = param[i]
    m_ref[i] = m[i]
    v_ref[i] = v[i]
  }
  for i = 0; i < n; i = i + 1 {
    let gi = grad[i]
    let mi = beta1 * m_ref[i] + (one - beta1) * gi
    let vi = beta2 * v_ref[i] + (one - beta2) * gi * gi
    m_ref[i] = mi
    v_ref[i] = vi
    let m_hat = mi / bias_c1
    let v_hat = vi / bias_c2
    let denom = Float::from_double(v_hat.to_double().sqrt()) + eps
    let update = m_hat / denom + weight_decay * p_ref[i]
    p_ref[i] = p_ref[i] - lr * update
  }
  adamw_step_inplace(
    param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, bias_c1, bias_c2,
  )
  for i = 0; i < n; i = i + 1 {
    let p_diff = (param[i] - p_ref[i]).to_double().abs()
    let m_diff = (m[i] - m_ref[i]).to_double().abs()
    let v_diff = (v[i] - v_ref[i]).to_double().abs()
    assert_true(p_diff < 1.0e-5)
    assert_true(m_diff < 1.0e-6)
    assert_true(v_diff < 1.0e-6)
  }
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
test "attention_head_backward_matches_reference" {
  let seq = 2
  let d_k = 3
  let scale = Float::from_double(1.0 / d_k.to_double().sqrt())
  let off = 5
  let n = seq * d_k
  let total = off + n
  let d_out_all = FixedArray::make(total, Float::from_int(0))
  let q_all = FixedArray::make(total, Float::from_int(0))
  let k_all = FixedArray::make(total, Float::from_int(0))
  let v_all = FixedArray::make(total, Float::from_int(0))
  for i = 0; i < n; i = i + 1 {
    d_out_all[off + i] = Float::from_int(i - 2) / Float::from_int(5)
    q_all[off + i] = Float::from_int(i * 2 % 7 - 3) / Float::from_int(6)
    k_all[off + i] = Float::from_int(i * 3 % 11 - 5) / Float::from_int(7)
    v_all[off + i] = Float::from_int(i * 5 % 13 - 6) / Float::from_int(8)
  }
  let attn_w = FixedArray::make(seq * seq, Float::from_int(0))
  attn_w[0] = Float::from_double(0.7)
  attn_w[1] = Float::from_double(0.3)
  attn_w[2] = Float::from_double(0.4)
  attn_w[3] = Float::from_double(0.6)
  let d_out = FixedArray::make(n, Float::from_int(0))
  let q = FixedArray::make(n, Float::from_int(0))
  let k = FixedArray::make(n, Float::from_int(0))
  let v = FixedArray::make(n, Float::from_int(0))
  for i = 0; i < n; i = i + 1 {
    d_out[i] = d_out_all[off + i]
    q[i] = q_all[off + i]
    k[i] = k_all[off + i]
    v[i] = v_all[off + i]
  }
  let d_attn = FixedArray::make(seq * seq, Float::from_int(0))
  let d_scores = FixedArray::make(seq * seq, Float::from_int(0))
  let dq_ref = FixedArray::make(n, Float::from_int(0))
  let dk_ref = FixedArray::make(n, Float::from_int(0))
  let dv_ref = FixedArray::make(n, Float::from_int(0))
  sgemm(
    0,
    1,
    seq,
    seq,
    d_k,
    Float::from_int(1),
    d_out,
    d_k,
    v,
    d_k,
    Float::from_int(0),
    d_attn,
    seq,
  )
  sgemm(
    1,
    0,
    seq,
    d_k,
    seq,
    Float::from_int(1),
    attn_w,
    seq,
    d_out,
    d_k,
    Float::from_int(0),
    dv_ref,
    d_k,
  )
  softmax_backward_rows(d_attn, attn_w, d_scores, seq, seq)
  scale_inplace(d_scores, scale, seq * seq)
  sgemm(
    0,
    0,
    seq,
    d_k,
    seq,
    Float::from_int(1),
    d_scores,
    seq,
    k,
    d_k,
    Float::from_int(0),
    dq_ref,
    d_k,
  )
  sgemm(
    1,
    0,
    seq,
    d_k,
    seq,
    Float::from_int(1),
    d_scores,
    seq,
    q,
    d_k,
    Float::from_int(0),
    dk_ref,
    d_k,
  )
  let dq_all = FixedArray::make(total, Float::from_int(0))
  let dk_all = FixedArray::make(total, Float::from_int(0))
  let dv_all = FixedArray::make(total, Float::from_int(0))
  attention_head_backward(
    d_out_all, off, q_all, off, k_all, off, v_all, off, attn_w, dq_all, off, dk_all,
    off, dv_all, off, seq, d_k, scale,
  )
  for i = 0; i < n; i = i + 1 {
    let dq_diff = (dq_all[off + i] - dq_ref[i]).to_double().abs()
    let dk_diff = (dk_all[off + i] - dk_ref[i]).to_double().abs()
    let dv_diff = (dv_all[off + i] - dv_ref[i]).to_double().abs()
    assert_true(dq_diff < 1.0e-6)
    assert_true(dk_diff < 1.0e-6)
    assert_true(dv_diff < 1.0e-6)
  }
}

///|
test "attention_backward_batch_matches_head_loop" {
  let total_heads = 3
  let seq = 2
  let d_k = 3
  let scale = Float::from_double(1.0 / d_k.to_double().sqrt())
  let n = total_heads * seq * d_k
  let w_n = total_heads * seq * seq
  let d_out = FixedArray::make(n, Float::from_int(0))
  let q = FixedArray::make(n, Float::from_int(0))
  let k = FixedArray::make(n, Float::from_int(0))
  let v = FixedArray::make(n, Float::from_int(0))
  for i = 0; i < n; i = i + 1 {
    d_out[i] = Float::from_int(i * 5 % 17 - 8) / Float::from_int(9)
    q[i] = Float::from_int(i * 3 % 13 - 6) / Float::from_int(8)
    k[i] = Float::from_int(i * 7 % 19 - 9) / Float::from_int(10)
    v[i] = Float::from_int(i * 11 % 23 - 11) / Float::from_int(12)
  }
  let attn_w = FixedArray::make(w_n, Float::from_int(0))
  for h = 0; h < total_heads; h = h + 1 {
    let base = h * seq * seq
    attn_w[base] = Float::from_double(0.65)
    attn_w[base + 1] = Float::from_double(0.35)
    attn_w[base + 2] = Float::from_double(0.4)
    attn_w[base + 3] = Float::from_double(0.6)
  }
  let dq_ref = FixedArray::make(n, Float::from_int(0))
  let dk_ref = FixedArray::make(n, Float::from_int(0))
  let dv_ref = FixedArray::make(n, Float::from_int(0))
  for h = 0; h < total_heads; h = h + 1 {
    let data_off = h * seq * d_k
    let w_off = h * seq * seq
    let w_head = FixedArray::make(seq * seq, Float::from_int(0))
    for i = 0; i < seq * seq; i = i + 1 {
      w_head[i] = attn_w[w_off + i]
    }
    attention_head_backward(
      d_out, data_off, q, data_off, k, data_off, v, data_off, w_head, dq_ref, data_off,
      dk_ref, data_off, dv_ref, data_off, seq, d_k, scale,
    )
  }
  let dq = FixedArray::make(n, Float::from_int(0))
  let dk = FixedArray::make(n, Float::from_int(0))
  let dv = FixedArray::make(n, Float::from_int(0))
  attention_backward_batch(
    d_out, q, k, v, attn_w, dq, dk, dv, total_heads, seq, d_k, scale,
  )
  for i = 0; i < n; i = i + 1 {
    let dq_diff = (dq[i] - dq_ref[i]).to_double().abs()
    let dk_diff = (dk[i] - dk_ref[i]).to_double().abs()
    let dv_diff = (dv[i] - dv_ref[i]).to_double().abs()
    assert_true(dq_diff < 1.0e-6)
    assert_true(dk_diff < 1.0e-6)
    assert_true(dv_diff < 1.0e-6)
  }
}

///|
test "attention_backward_batch_interleaved_matches_reshape_path" {
  let batch = 2
  let seq = 3
  let num_heads = 2
  let d_k = 4
  let d_model = num_heads * d_k
  let total = batch * seq * d_model
  let total_heads = batch * num_heads
  let scale = Float::from_double(1.0 / d_k.to_double().sqrt())
  let d_out = tensor_new(
    shape_new_unchecked([batch, seq, d_model]),
    Array::makei(total, fn(i) {
      Float::from_int(i * 5 % 23 - 11) / Float::from_int(9)
    }),
  ).unwrap()
  let q = tensor_new(
    shape_new_unchecked([batch, seq, d_model]),
    Array::makei(total, fn(i) {
      Float::from_int(i * 3 % 19 - 9) / Float::from_int(8)
    }),
  ).unwrap()
  let k = tensor_new(
    shape_new_unchecked([batch, seq, d_model]),
    Array::makei(total, fn(i) {
      Float::from_int(i * 7 % 29 - 14) / Float::from_int(10)
    }),
  ).unwrap()
  let v = tensor_new(
    shape_new_unchecked([batch, seq, d_model]),
    Array::makei(total, fn(i) {
      Float::from_int(i * 11 % 31 - 15) / Float::from_int(11)
    }),
  ).unwrap()
  let q_heads = reshape_for_heads(q, batch, seq, num_heads, d_k)
  let k_heads = reshape_for_heads(k, batch, seq, num_heads, d_k)
  let v_heads = reshape_for_heads(v, batch, seq, num_heads, d_k)
  let d_out_heads = reshape_for_heads(d_out, batch, seq, num_heads, d_k)
  let attn_w = FixedArray::make(total_heads * seq * seq, Float::from_int(0))
  for h = 0; h < total_heads; h = h + 1 {
    let base = h * seq * seq
    // Each row sums to 1.0.
    attn_w[base] = Float::from_double(0.6)
    attn_w[base + 1] = Float::from_double(0.3)
    attn_w[base + 2] = Float::from_double(0.1)
    attn_w[base + 3] = Float::from_double(0.2)
    attn_w[base + 4] = Float::from_double(0.5)
    attn_w[base + 5] = Float::from_double(0.3)
    attn_w[base + 6] = Float::from_double(0.1)
    attn_w[base + 7] = Float::from_double(0.2)
    attn_w[base + 8] = Float::from_double(0.7)
  }

  // Reference path: reshape -> attention_backward_batch -> reshape back.
  let n_heads = total_heads * seq * d_k
  let dq_heads_ref = FixedArray::make(n_heads, Float::from_int(0))
  let dk_heads_ref = FixedArray::make(n_heads, Float::from_int(0))
  let dv_heads_ref = FixedArray::make(n_heads, Float::from_int(0))
  attention_backward_batch(
    d_out_heads.data,
    q_heads.data,
    k_heads.data,
    v_heads.data,
    attn_w,
    dq_heads_ref,
    dk_heads_ref,
    dv_heads_ref,
    total_heads,
    seq,
    d_k,
    scale,
  )
  let heads_shape = shape_new_unchecked([batch, num_heads, seq, d_k])
  let dq_ref = reshape_from_heads(
    Tensor::{
      data: dq_heads_ref,
      shape: heads_shape,
      offset: 0,
      strides: heads_shape.strides(),
    },
    batch,
    seq,
    num_heads,
    d_k,
  )
  let dk_ref = reshape_from_heads(
    Tensor::{
      data: dk_heads_ref,
      shape: heads_shape,
      offset: 0,
      strides: heads_shape.strides(),
    },
    batch,
    seq,
    num_heads,
    d_k,
  )
  let dv_ref = reshape_from_heads(
    Tensor::{
      data: dv_heads_ref,
      shape: heads_shape,
      offset: 0,
      strides: heads_shape.strides(),
    },
    batch,
    seq,
    num_heads,
    d_k,
  )

  // Candidate path: direct interleaved layout.
  let dq = FixedArray::make(total, Float::from_int(0))
  let dk = FixedArray::make(total, Float::from_int(0))
  let dv = FixedArray::make(total, Float::from_int(0))
  let d_out_c = d_out.contiguous()
  let q_c = q.contiguous()
  let k_c = k.contiguous()
  let v_c = v.contiguous()
  attention_backward_batch_interleaved(
    d_out_c.data,
    q_c.data,
    k_c.data,
    v_c.data,
    attn_w,
    dq,
    dk,
    dv,
    batch,
    num_heads,
    seq,
    d_k,
    scale,
  )
  for i = 0; i < total; i = i + 1 {
    let dq_diff = (dq[i] - dq_ref.data[i]).to_double().abs()
    let dk_diff = (dk[i] - dk_ref.data[i]).to_double().abs()
    let dv_diff = (dv[i] - dv_ref.data[i]).to_double().abs()
    assert_true(dq_diff < 1.0e-6)
    assert_true(dk_diff < 1.0e-6)
    assert_true(dv_diff < 1.0e-6)
  }
}

///|
test "attention_forward_batch_interleaved_masked_matches_reshape_path" {
  let batch = 2
  let seq = 3
  let num_heads = 2
  let d_k = 4
  let d_model = num_heads * d_k
  let total = batch * seq * d_model
  let total_heads = batch * num_heads
  let scale = Float::from_double(1.0 / d_k.to_double().sqrt())
  let q = tensor_new(
    shape_new_unchecked([batch, seq, d_model]),
    Array::makei(total, fn(i) {
      Float::from_int(i * 5 % 23 - 11) / Float::from_int(9)
    }),
  ).unwrap()
  let k = tensor_new(
    shape_new_unchecked([batch, seq, d_model]),
    Array::makei(total, fn(i) {
      Float::from_int(i * 3 % 19 - 9) / Float::from_int(8)
    }),
  ).unwrap()
  let v = tensor_new(
    shape_new_unchecked([batch, seq, d_model]),
    Array::makei(total, fn(i) {
      Float::from_int(i * 7 % 29 - 14) / Float::from_int(10)
    }),
  ).unwrap()
  let mask = causal_mask(seq).unwrap()

  // Reference path: reshape -> per-head attention loop -> reshape back.
  let q_heads = reshape_for_heads(q, batch, seq, num_heads, d_k)
  let k_heads = reshape_for_heads(k, batch, seq, num_heads, d_k)
  let v_heads = reshape_for_heads(v, batch, seq, num_heads, d_k)
  let qh = q_heads.contiguous()
  let kh = k_heads.contiguous()
  let vh = v_heads.contiguous()
  let mask_c = mask.contiguous()
  let scores_size = seq * seq
  let attn_w_ref = FixedArray::make(
    total_heads * scores_size,
    Float::from_int(0),
  )
  let out_heads_ref = FixedArray::make(
    total_heads * seq * d_k,
    Float::from_int(0),
  )
  for b = 0; b < batch; b = b + 1 {
    for h = 0; h < num_heads; h = h + 1 {
      let head_idx = b * num_heads + h
      let head_base = head_idx * seq * d_k
      let scores_off = head_idx * scores_size
      sgemm_offset(
        0,
        1,
        seq,
        seq,
        d_k,
        scale,
        qh.data,
        head_base,
        d_k,
        kh.data,
        head_base,
        d_k,
        Float::from_int(0),
        attn_w_ref,
        scores_off,
        seq,
      )
      add_matrix_inplace_offset(attn_w_ref, scores_off, mask_c.data, seq, seq)
      softmax_inplace_offset(attn_w_ref, scores_off, seq, seq)
      sgemm_offset(
        0,
        0,
        seq,
        d_k,
        seq,
        Float::from_int(1),
        attn_w_ref,
        scores_off,
        seq,
        vh.data,
        head_base,
        d_k,
        Float::from_int(0),
        out_heads_ref,
        head_base,
        d_k,
      )
    }
  }
  let heads_shape = shape_new_unchecked([batch, num_heads, seq, d_k])
  let out_ref_heads = Tensor::{
    data: out_heads_ref,
    shape: heads_shape,
    offset: 0,
    strides: heads_shape.strides(),
  }
  let out_ref = reshape_from_heads(out_ref_heads, batch, seq, num_heads, d_k)

  // Candidate path: direct interleaved layout.
  let attn_w = FixedArray::make(total_heads * scores_size, Float::from_int(0))
  let out = FixedArray::make(total, Float::from_int(0))
  let q_c = q.contiguous()
  let k_c = k.contiguous()
  let v_c = v.contiguous()
  attention_forward_batch_interleaved_masked(
    q_c.data,
    k_c.data,
    v_c.data,
    mask_c.data,
    attn_w,
    out,
    batch,
    num_heads,
    seq,
    d_k,
    scale,
  )
  for i = 0; i < total; i = i + 1 {
    let diff = (out[i] - out_ref.data[i]).to_double().abs()
    assert_true(diff < 1.0e-6)
  }
  for i = 0; i < total_heads * scores_size; i = i + 1 {
    let diff = (attn_w[i] - attn_w_ref[i]).to_double().abs()
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
  let w_all : FixedArray[Float] = cache.block_caches[0].attn_weights
  inspect(w_all.length(), content="64")
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
test "training_lm_profile_records_metrics" {
  let text = "abcabcabcabcabcabcabcabcabcabcabcabc"
  let tokenizer = char_tokenizer_from_text(text)
  let vocab = tokenizer.vocab_size()
  let config = transformer_config(vocab, 8, 1, 1, 16, 8).unwrap()
  let metrics = transformer_train_lm_profile_steps(
    text,
    config,
    2,
    6,
    8,
    Float::from_double(0.01),
    42,
  )
  assert_true(metrics.length() == 6)
  for i = 0; i < metrics.length(); i = i + 1 {
    let m = metrics[i]
    assert_true(m.loss.to_double() > 0.0)
    assert_true(m.perplexity.to_double() > 0.0)
    assert_true(m.step_ns > 0UL)
  }
}

///|
test "training_lm_profile_records_metrics_adamw" {
  let text = "abcabcabcabcabcabcabcabcabcabcabcabc"
  let tokenizer = char_tokenizer_from_text(text)
  let vocab = tokenizer.vocab_size()
  let config = transformer_config(vocab, 8, 1, 1, 16, 8).unwrap()
  let metrics = transformer_train_lm_profile_steps_adamw(
    text,
    config,
    2,
    6,
    8,
    Float::from_double(0.01),
    42,
  )
  assert_true(metrics.length() == 6)
  for i = 0; i < metrics.length(); i = i + 1 {
    let m = metrics[i]
    assert_true(m.loss.to_double() > 0.0)
    assert_true(m.perplexity.to_double() > 0.0)
    assert_true(m.step_ns > 0UL)
  }
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
