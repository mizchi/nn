///|
test "mlp_params_bytes_roundtrip" {
  let spec = @nn.mlp_spec_new(2, 3, 1, 1).unwrap()
  let params = @nn.mlp_init_params_with_policy(
    spec, 1, @nn.mlp_init_policy_deterministic,
  ).unwrap()
  let bytes = @nn.mlp_params_to_bytes(spec, params).unwrap()
  let restored = @nn.mlp_params_from_bytes(spec, bytes).unwrap()
  let bytes2 = @nn.mlp_params_to_bytes(spec, restored).unwrap()
  inspect(bytes.equal(bytes2), content="true")
}

///|
test "mlp_train_reduces_loss" {
  let spec = @nn.mlp_spec_new(2, 2, 2, 2).unwrap()
  let f = Float::from_int
  let inputs : Array[Float] = [f(1), f(0), f(0), f(1)]
  let labels : Array[Int] = [0, 1]
  let dataset = @nn.mlp_dataset_new(2, inputs, labels).unwrap()
  let params = @nn.mlp_init_params_with_policy(
    spec, 0, @nn.mlp_init_policy_deterministic,
  ).unwrap()
  let before = mlp_eval(spec, params, dataset).unwrap()
  let cfg = @nn.mlp_train_config(5, 2, f(1) / f(20), false, 0)
  let result = mlp_train(spec, params, dataset, cfg).unwrap()
  let after = result.metrics[result.metrics.length() - 1]
  let improved = after.loss < before.loss
  inspect(improved, content="true")
}

///|
test "mlp_eval_from_bytes_matches_eval" {
  let spec = @nn.mlp_spec_new(2, 2, 2, 1).unwrap()
  let f = Float::from_int
  let inputs : Array[Float] = [f(1), f(0), f(0), f(1)]
  let labels : Array[Int] = [0, 1]
  let dataset = @nn.mlp_dataset_new(2, inputs, labels).unwrap()
  let params = @nn.mlp_init_params_with_policy(
    spec, 0, @nn.mlp_init_policy_deterministic,
  ).unwrap()
  let expected = mlp_eval(spec, params, dataset).unwrap()
  let bytes = @nn.mlp_params_to_bytes(spec, params).unwrap()
  let actual = mlp_eval_from_bytes(spec, bytes, dataset).unwrap()
  inspect(expected.loss == actual.loss, content="true")
  inspect(expected.accuracy == actual.accuracy, content="true")
}
