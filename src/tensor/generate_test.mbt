///|
test "tensor_argmax" {
  let f = Float::from_int
  let t = tensor_from_array([f(1), f(5), f(3), f(2)])
  inspect(tensor_argmax(t), content="1")
}

///|
test "tensor_log_softmax" {
  let f = Float::from_int
  let shape = shape_new([1, 3]).unwrap()
  let t = tensor_new(shape, [f(1), f(2), f(3)]).unwrap()
  let ls = tensor_log_softmax(t).unwrap()
  // log_softmax values should be negative and sum(exp(log_softmax)) ≈ 1
  let mut sum = Float::from_int(0)
  for j = 0; j < 3; j = j + 1 {
    let v = ls.at2(0, j)
    // All log-softmax values should be <= 0
    inspect(v <= Float::from_int(0), content="true")
    sum = sum + (@math.exp(v.to_double()) |> Float::from_double)
  }
  let ok = Float::is_close(sum, Float::from_int(1))
  inspect(ok, content="true")
}

///|
test "tensor_cross_entropy" {
  let f = Float::from_int
  let shape = shape_new([2, 4]).unwrap()
  // Logits where correct class has highest value => low loss
  let logits = tensor_new(shape, [
    f(10),
    f(1),
    f(1),
    f(1),
    f(1),
    f(1),
    f(10),
    f(1),
  ]).unwrap()
  let labels = [0, 2]
  let loss = tensor_cross_entropy(logits, labels).unwrap()
  // Loss should be small since logits peak at correct labels
  inspect(loss < Float::from_int(1), content="true")
}

///|
test "tensor_cross_entropy_high_loss" {
  let f = Float::from_int
  let shape = shape_new([1, 4]).unwrap()
  // Logits where wrong class has highest value => high loss
  let logits = tensor_new(shape, [f(0), f(0), f(10), f(0)]).unwrap()
  let labels = [0] // correct is 0, but logits peak at 2
  let loss = tensor_cross_entropy(logits, labels).unwrap()
  // Loss should be large
  inspect(loss > Float::from_int(5), content="true")
}

///|
test "generate_greedy_output_length" {
  let config = transformer_config(8, 4, 2, 1, 8, 32).unwrap()
  let params = transformer_init_params(config, 42)
  let prompt = [0, 1, 2]
  let result = generate_greedy(params, config, prompt, 5).unwrap()
  // Should have prompt_len + max_len tokens
  inspect(result.length(), content="8")
}

///|
test "generate_greedy_deterministic" {
  let config = transformer_config(8, 4, 2, 1, 8, 32).unwrap()
  let params = transformer_init_params(config, 42)
  let prompt = [0, 1]
  let r1 = generate_greedy(params, config, prompt, 3).unwrap()
  let r2 = generate_greedy(params, config, prompt, 3).unwrap()
  // Greedy should produce identical results
  let mut same = true
  for i = 0; i < r1.length(); i = i + 1 {
    if r1[i] != r2[i] {
      same = false
      break
    }
  }
  inspect(same, content="true")
}

///|
test "generate_with_temperature_output_length" {
  let config = transformer_config(8, 4, 2, 1, 8, 32).unwrap()
  let params = transformer_init_params(config, 42)
  let prompt = [0, 1]
  let temp = 1.0 |> Float::from_double
  let result = generate_with_temperature(params, config, prompt, 4, temp, 123).unwrap()
  inspect(result.length(), content="6")
}
