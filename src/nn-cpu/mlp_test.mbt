///|
test "mlp_buffer_plan_sizes" {
  let spec = @nn.mlp_spec_new(4, 8, 2, 1).unwrap()
  let plan = @nn.mlp_plan_buffers(spec).unwrap()
  inspect(plan.input_bytes, content="16")
  inspect(plan.hidden_bytes, content="32")
  inspect(plan.output_bytes, content="8")
  inspect(plan.weight1_bytes, content="128")
  inspect(plan.bias1_bytes, content="32")
  inspect(plan.weight2_bytes, content="64")
  inspect(plan.bias2_bytes, content="8")
}

///|
test "mlp_dataset_buffer_plan_sizes" {
  let spec = @nn.mlp_spec_new(4, 3, 2, 2).unwrap()
  let plan = @nn.mlp_plan_dataset_buffers(spec, 10).unwrap()
  inspect(plan.dataset_input_bytes, content="160")
  inspect(plan.dataset_label_bytes, content="40")
  inspect(plan.indices_bytes, content="44")
  inspect(plan.shuffle_params_bytes, content="8")
}

///|
test "mlp_loss_buffer_plan_dataset" {
  let spec = @nn.mlp_spec_new(4, 3, 2, 2).unwrap()
  let plan = @nn.mlp_plan_loss_buffers_for_dataset(spec, 10).unwrap()
  inspect(plan.labels_bytes, content="40")
  inspect(plan.loss_bytes, content="8")
  inspect(plan.probs_bytes, content="16")
  inspect(plan.loss_sum_bytes, content="4")
  inspect(plan.correct_bytes, content="8")
  inspect(plan.correct_sum_bytes, content="4")
  inspect(plan.epoch_loss_bytes, content="4")
  inspect(plan.epoch_correct_bytes, content="4")
  inspect(plan.epoch_seen_bytes, content="4")
}

///|
test "mlp_resource_plan_usage" {
  let spec = @nn.mlp_spec_new(4, 8, 2, 1).unwrap()
  let plan = @nn.mlp_plan_resources(spec).unwrap()
  let upload = @wgpu.buffer_usage_or(
    @wgpu.buffer_usage_storage, @wgpu.buffer_usage_copy_dst,
  )
  let readback = @wgpu.buffer_usage_or(upload, @wgpu.buffer_usage_copy_src)
  inspect(plan.input.usage, content=upload.to_string())
  inspect(plan.output.usage, content=readback.to_string())
}

///|
test "mlp_dispatch_plan" {
  let spec = @nn.mlp_spec_new(4, 8, 2, 2).unwrap()
  let plan = @nn.mlp_dispatch_plan(spec, 4).unwrap()
  inspect(plan.workgroup_size, content="4")
  inspect(plan.layer1_dispatch_x, content="4")
  inspect(plan.layer2_dispatch_x, content="1")
}

///|
test "mlp_dispatch_plan_tiled_grid" {
  let spec = @nn.mlp_spec_new(4, 9, 2, 1).unwrap()
  let plan = @nn.mlp_dispatch_plan(spec, 4).unwrap()
  inspect(plan.layer1_dispatch_x, content="5")
  inspect(plan.layer2_dispatch_x, content="1")
}

///|
test "mlp_train_dispatch_plan" {
  let spec = @nn.mlp_spec_new(4, 3, 2, 2).unwrap()
  let plan = @nn.mlp_train_dispatch_plan(spec, 4).unwrap()
  inspect(plan.grad_update_w1_dispatch_x, content="3")
  inspect(plan.grad_update_b1_dispatch_x, content="1")
  inspect(plan.grad_update_w2_dispatch_x, content="2")
  inspect(plan.grad_update_b2_dispatch_x, content="1")
}

///|
test "mlp_shader_plan" {
  let spec = @nn.mlp_spec_new(4, 8, 2, 1).unwrap()
  let plan = @nn.mlp_shader_plan(spec, 64).unwrap()
  inspect(
    plan.layer1_wgsl.contains("const INPUT_SIZE : u32 = 4u;"),
    content="true",
  )
  inspect(plan.layer1_wgsl.contains("@workgroup_size(64)"), content="true")
  inspect(
    plan.layer1_wgsl.contains("var<workgroup> input_tile"),
    content="true",
  )
  inspect(
    plan.layer2_wgsl.contains("const OUTPUT_SIZE : u32 = 2u;"),
    content="true",
  )
  inspect(
    plan.layer2_wgsl.contains("var<workgroup> weight_tile"),
    content="true",
  )
}

///|
test "mlp_params_init_lengths" {
  let spec = @nn.mlp_spec_new(2, 3, 1, 1).unwrap()
  let params = @nn.mlp_init_params(spec, 7).unwrap()
  inspect(params.weight1.length(), content="6")
  inspect(params.bias1.length(), content="3")
  inspect(params.weight2.length(), content="3")
  inspect(params.bias2.length(), content="1")
}

///|
test "mlp_forward_cpu" {
  let spec = @nn.mlp_spec_new(2, 2, 1, 1).unwrap()
  let f = Float::from_int
  let params = @nn.mlp_params_new(
    spec,
    [f(1), f(0), f(0), f(1)],
    [f(0), f(0)],
    [f(1), f(1)],
    [f(0)],
  ).unwrap()
  let input = [f(1), f(2)]
  let result = mlp_forward(spec, params, input).unwrap()
  let ok = Float::is_close(result.output[0], f(3))
  inspect(ok, content="true")
}

///|
test "mlp_io_bytes" {
  let spec = @nn.mlp_spec_new(2, 2, 1, 1).unwrap()
  let params = @nn.mlp_init_params(spec, 1).unwrap()
  let f = Float::from_int
  let input = [f(1), f(-1)]
  let bytes = @nn.mlp_input_to_bytes(spec, input).unwrap()
  inspect(bytes.length(), content="8")
  let input_roundtrip = @nn.mlp_input_bytes_to_array(spec, bytes).unwrap()
  inspect(input_roundtrip.length(), content="2")
  let output = mlp_forward(spec, params, input).unwrap().output
  let output_bytes = @nn.mlp_output_from_bytes(spec, output).unwrap()
  let output_roundtrip = @nn.mlp_output_bytes_to_array(spec, output_bytes).unwrap()
  let ok = Float::is_close(output_roundtrip[0], output[0])
  inspect(ok, content="true")
}

///|
test "mlp_init_policy_zero" {
  let spec = @nn.mlp_spec_new(2, 2, 1, 1).unwrap()
  let params = @nn.mlp_init_params_with_policy(
    spec, 0, @nn.mlp_init_policy_zeros,
  ).unwrap()
  let zero = Float::from_int(0)
  let sum_w1 = params.weight1.fold(init=zero, fn(acc, v) { acc + v })
  let sum_b1 = params.bias1.fold(init=zero, fn(acc, v) { acc + v })
  let sum_w2 = params.weight2.fold(init=zero, fn(acc, v) { acc + v })
  let sum_b2 = params.bias2.fold(init=zero, fn(acc, v) { acc + v })
  inspect(sum_w1, content="0")
  inspect(sum_b1, content="0")
  inspect(sum_w2, content="0")
  inspect(sum_b2, content="0")
}

///|
test "mlp_init_policy_deterministic" {
  let spec = @nn.mlp_spec_new(2, 2, 1, 1).unwrap()
  let params = @nn.mlp_init_params_with_policy(
    spec, 1, @nn.mlp_init_policy_deterministic,
  ).unwrap()
  let zero = Float::from_int(0)
  let any_non_zero = params.weight1.any(fn(v) { v != zero })
  inspect(any_non_zero, content="true")
}
