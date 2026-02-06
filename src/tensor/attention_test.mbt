///|
test "causal_mask" {
  let mask = causal_mask(3).unwrap()
  inspect(mask.dim(0), content="3")
  inspect(mask.dim(1), content="3")
  // Diagonal and below should be 0
  let zero = Float::from_int(0)
  let ok00 = Float::is_close(mask.at2(0, 0), zero)
  let ok10 = Float::is_close(mask.at2(1, 0), zero)
  let ok11 = Float::is_close(mask.at2(1, 1), zero)
  let ok20 = Float::is_close(mask.at2(2, 0), zero)
  inspect(ok00, content="true")
  inspect(ok10, content="true")
  inspect(ok11, content="true")
  inspect(ok20, content="true")
  // Above diagonal should be large negative
  let neg = mask.at2(0, 1)
  let is_masked = neg < Float::from_double(-100000000.0)
  inspect(is_masked, content="true")
}

///|
test "attention_simple" {
  let f = Float::from_int
  // Simple 2D attention: [seq=2, d_k=2]
  let shape = shape_new([2, 2]).unwrap()
  let q = tensor_new(shape, [f(1), f(0), f(0), f(1)]).unwrap()
  let k = tensor_new(shape, [f(1), f(0), f(0), f(1)]).unwrap()
  let v = tensor_new(shape, [f(1), f(2), f(3), f(4)]).unwrap()
  let out = attention(q, k, v, None).unwrap()
  inspect(out.dim(0), content="2")
  inspect(out.dim(1), content="2")
  // Output should be weighted combination of V rows
  // Q[0] = [1,0] attends more to K[0] = [1,0]
  // Q[1] = [0,1] attends more to K[1] = [0,1]
}

///|
test "attention_with_mask" {
  let f = Float::from_int
  let shape = shape_new([2, 2]).unwrap()
  let q = tensor_new(shape, [f(1), f(1), f(1), f(1)]).unwrap()
  let k = tensor_new(shape, [f(1), f(1), f(1), f(1)]).unwrap()
  let v = tensor_new(shape, [f(1), f(0), f(0), f(1)]).unwrap()
  let mask = causal_mask(2).unwrap()
  let out = attention(q, k, v, Some(mask)).unwrap()
  // With causal mask, position 0 can only attend to position 0
  // So out[0] should be v[0] = [1, 0]
  let ok0 = Float::is_close(out.at2(0, 0), f(1))
  let ok1 = Float::is_close(out.at2(0, 1), f(0))
  inspect(ok0, content="true")
  inspect(ok1, content="true")
}

///|
test "attention_batched" {
  let f = Float::from_int
  // Batched attention: [batch=2, seq=2, d_k=2]
  let shape = shape_new([2, 2, 2]).unwrap()
  // Simple identity-like Q, K
  let q = tensor_new(shape, [f(1), f(0), f(0), f(1), f(1), f(0), f(0), f(1)]).unwrap()
  let k = tensor_new(shape, [f(1), f(0), f(0), f(1), f(1), f(0), f(0), f(1)]).unwrap()
  let v = tensor_new(shape, [f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8)]).unwrap()
  let out = attention(q, k, v, None).unwrap()
  inspect(out.dim(0), content="2")
  inspect(out.dim(1), content="2")
  inspect(out.dim(2), content="2")
}

///|
test "multi_head_attention_shapes" {
  let f = Float::from_int
  let batch = 1
  let seq = 2
  let d_model = 4
  let num_heads = 2
  // Input: [batch, seq, d_model]
  let x_shape = shape_new([batch, seq, d_model]).unwrap()
  let x_data = Array::makei(batch * seq * d_model, fn(i) { f(i + 1) })
  let q = tensor_new(x_shape, x_data.copy()).unwrap()
  let k = tensor_new(x_shape, x_data.copy()).unwrap()
  let v = tensor_new(x_shape, x_data.copy()).unwrap()
  // Weights: [d_model, d_model]
  let w_shape = shape_new([d_model, d_model]).unwrap()
  let w_data = Array::make(d_model * d_model, 0.1 |> Float::from_double)
  // Initialize with small identity-like values
  for i = 0; i < d_model; i = i + 1 {
    w_data[i * d_model + i] = 1.0 |> Float::from_double
  }
  let w_q = tensor_new(w_shape, w_data.copy()).unwrap()
  let w_k = tensor_new(w_shape, w_data.copy()).unwrap()
  let w_v = tensor_new(w_shape, w_data.copy()).unwrap()
  let w_o = tensor_new(w_shape, w_data.copy()).unwrap()
  let out = multi_head_attention(q, k, v, w_q, w_k, w_v, w_o, num_heads, None).unwrap()
  // Output should be [batch, seq, d_model]
  inspect(out.dim(0), content="1")
  inspect(out.dim(1), content="2")
  inspect(out.dim(2), content="4")
}
