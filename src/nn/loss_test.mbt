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
