///|
test "embedding" {
  let f = Float::from_int
  // weight: [vocab=3, d_model=2]
  let w_shape = shape_new([3, 2]).unwrap()
  let weight = tensor_new(w_shape, [f(10), f(11), f(20), f(21), f(30), f(31)]).unwrap()
  // indices: [[0, 2], [1, 0]]  (batch=2, seq=2)
  let indices : Array[Array[Int]] = [[0, 2], [1, 0]]
  let out = embedding(indices, weight)
  inspect(out.dim(0), content="2")
  inspect(out.dim(1), content="2")
  inspect(out.dim(2), content="2")
  // batch=0, seq=0, token=0 -> [10, 11]
  inspect(out.at3(0, 0, 0), content="10")
  inspect(out.at3(0, 0, 1), content="11")
  // batch=0, seq=1, token=2 -> [30, 31]
  inspect(out.at3(0, 1, 0), content="30")
  inspect(out.at3(0, 1, 1), content="31")
  // batch=1, seq=0, token=1 -> [20, 21]
  inspect(out.at3(1, 0, 0), content="20")
}

///|
test "add_positional_embedding" {
  let f = Float::from_int
  // x: [batch=1, seq=2, d_model=2]
  let x_shape = shape_new([1, 2, 2]).unwrap()
  let x = tensor_new(x_shape, [f(1), f(2), f(3), f(4)]).unwrap()
  // pos: [max_seq=4, d_model=2]
  let p_shape = shape_new([4, 2]).unwrap()
  let pos = tensor_new(p_shape, [
    f(100),
    f(200),
    f(300),
    f(400),
    f(0),
    f(0),
    f(0),
    f(0),
  ]).unwrap()
  let out = add_positional_embedding(x, pos)
  // x[0,0,:] + pos[0,:] = [1+100, 2+200] = [101, 202]
  inspect(out.at3(0, 0, 0), content="101")
  inspect(out.at3(0, 0, 1), content="202")
  // x[0,1,:] + pos[1,:] = [3+300, 4+400] = [303, 404]
  inspect(out.at3(0, 1, 0), content="303")
  inspect(out.at3(0, 1, 1), content="404")
}

///|
test "feed_forward" {
  let f = Float::from_int
  let d_model = 2
  let d_ff = 4
  // x: [batch=1, seq=1, d_model=2]
  let x_shape = shape_new([1, 1, d_model]).unwrap()
  let x = tensor_new(x_shape, [f(1), f(2)]).unwrap()
  // w1: [d_model, d_ff], b1: [d_ff]
  let w1_shape = shape_new([d_model, d_ff]).unwrap()
  let w1_data = Array::make(d_model * d_ff, 0.5 |> Float::from_double)
  let w1 = tensor_new(w1_shape, w1_data).unwrap()
  let b1 = tensor_zeros(shape_new_unchecked([d_ff]))
  // w2: [d_ff, d_model], b2: [d_model]
  let w2_shape = shape_new([d_ff, d_model]).unwrap()
  let w2_data = Array::make(d_ff * d_model, 0.1 |> Float::from_double)
  let w2 = tensor_new(w2_shape, w2_data).unwrap()
  let b2 = tensor_zeros(shape_new_unchecked([d_model]))
  let out = feed_forward(x, w1, b1, w2, b2).unwrap()
  inspect(out.dim(0), content="1")
  inspect(out.dim(1), content="1")
  inspect(out.dim(2), content="2")
}

///|
test "transformer_block" {
  let d_model = 4
  let num_heads = 2
  let d_ff = 8
  let config = transformer_config(10, d_model, num_heads, 1, d_ff, 16).unwrap()
  let params = transformer_init_params(config, 42)
  let block_params = params.blocks[0]
  // x: [batch=1, seq=2, d_model=4]
  let x_shape = shape_new([1, 2, d_model]).unwrap()
  let x_data = Array::makei(1 * 2 * d_model, fn(i) { Float::from_int(i + 1) })
  let x = tensor_new(x_shape, x_data).unwrap()
  let out = transformer_block(x, block_params, config, None).unwrap()
  inspect(out.dim(0), content="1")
  inspect(out.dim(1), content="2")
  inspect(out.dim(2), content="4")
}

///|
test "transformer_forward" {
  let vocab_size = 8
  let d_model = 4
  let num_heads = 2
  let num_layers = 2
  let d_ff = 8
  let max_seq_len = 16
  let config = transformer_config(
    vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len,
  ).unwrap()
  let params = transformer_init_params(config, 123)
  let tokens : Array[Array[Int]] = [[1, 2, 3]]
  let mask = causal_mask(3).unwrap()
  let logits = transformer_forward(tokens, params, config, Some(mask)).unwrap()
  // logits: [batch=1, seq=3, vocab_size=8]
  inspect(logits.dim(0), content="1")
  inspect(logits.dim(1), content="3")
  inspect(logits.dim(2), content="8")
}

///|
test "transformer_forward_generates_different_logits_per_position" {
  let config = transformer_config(8, 4, 2, 1, 8, 16).unwrap()
  let params = transformer_init_params(config, 42)
  let tokens : Array[Array[Int]] = [[0, 1, 2]]
  let logits = transformer_forward(tokens, params, config, None).unwrap()
  // Different positions should produce different logits
  let pos0_0 = logits.at3(0, 0, 0)
  let pos1_0 = logits.at3(0, 1, 0)
  let pos2_0 = logits.at3(0, 2, 0)
  // At least some should differ
  let all_same = Float::is_close(pos0_0, pos1_0) &&
    Float::is_close(pos1_0, pos2_0)
  inspect(all_same, content="false")
}
