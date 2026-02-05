///|
test "mlp_loss_plan_buffers" {
  let spec = mlp_spec_new(4, 3, 2, 5).unwrap()
  let plan = mlp_plan_loss_buffers(spec).unwrap()
  inspect(plan.labels_bytes, content="20")
  inspect(plan.loss_bytes, content="20")
  inspect(plan.probs_bytes, content="40")
  inspect(plan.loss_sum_bytes, content="4")
}

///|
test "mlp_loss_dispatch_x" {
  let spec = mlp_spec_new(4, 3, 2, 5).unwrap()
  let dispatch = mlp_loss_dispatch_x(spec, 64).unwrap()
  inspect(dispatch, content="1")
}

///|
test "mlp_loss_reduce_dispatch_x" {
  let spec = mlp_spec_new(4, 3, 2, 5).unwrap()
  let dispatch = mlp_loss_reduce_dispatch_x(spec, 64).unwrap()
  inspect(dispatch, content="1")
}

///|
test "mlp_loss_reduce_wgsl_uses_workgroup" {
  let spec = mlp_spec_new(4, 3, 2, 5).unwrap()
  let plan = mlp_loss_shader_plan(spec, 64).unwrap()
  inspect(plan.loss_reduce_wgsl.contains("var<workgroup>"), content="true")
  inspect(plan.loss_reduce_wgsl.contains("local_invocation_id"), content="true")
}

///|
test "mlp_loss_epoch_reduce_wgsl" {
  let spec = mlp_spec_new(4, 3, 2, 5).unwrap()
  let plan = mlp_loss_shader_plan(spec, 64).unwrap()
  inspect(plan.loss_epoch_reduce_wgsl.contains("epoch_loss"), content="true")
  inspect(plan.loss_epoch_reduce_wgsl.contains("epoch_seen"), content="true")
}

///|
test "mlp_layer2_loss_wgsl" {
  let spec = mlp_spec_new(4, 3, 2, 5).unwrap()
  let plan = mlp_loss_shader_plan(spec, 64).unwrap()
  inspect(plan.layer2_loss_wgsl.contains("layer2 + loss"), content="true")
  inspect(plan.layer2_loss_wgsl.contains("probs"), content="true")
}
