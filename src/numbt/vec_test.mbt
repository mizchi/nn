///|
fn assert_near(actual : Float, expected : Float, tol : Float) -> Unit {
  let diff = (actual - expected).abs()
  if diff > tol {
    panic()
  }
}

///|
test "vec_ops_basic" {
  let left = vec_from_array([1.0, 2.0, 3.0])
  let right = vec_from_array([3.0, 2.0, 1.0])
  let added = left + right
  let subbed = left - right
  let mulled = left * right
  assert_eq(vec_to_array(added), [4.0, 4.0, 4.0])
  assert_eq(vec_to_array(subbed), [-2.0, 0.0, 2.0])
  assert_eq(vec_to_array(mulled), [3.0, 4.0, 3.0])
}

///|
test "matmul_vec_bias_relu" {
  let mat = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let vec = vec_from_array([10.0, 20.0])
  let bias = vec_from_array([1.0, 1.0, 1.0])
  let out = vec_new(3, 0.0)
  matmul_vec_bias_relu_into(mat, vec, bias, out)
  assert_eq(vec_to_array(out), [91.0, 121.0, 151.0])
}

///|
test "softmax_into" {
  let logits = vec_from_array([0.0, 1.0])
  let probs = vec_new(2, 0.0)
  softmax_into(logits, probs)
  let values = vec_to_array(probs)
  assert_near(values[0], 0.26894, 0.0001)
  assert_near(values[1], 0.73105, 0.0001)
}

///|
test "argmax" {
  let values = vec_from_array([1.0, 3.0, 2.0])
  assert_eq(vec_argmax(values), 1)
}

