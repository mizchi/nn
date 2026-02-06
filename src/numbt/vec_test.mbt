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

// ============================================================================
// BLAS tests
// ============================================================================

///|
test "vec_dot" {
  let x = vec_from_array([1.0, 2.0, 3.0])
  let y = vec_from_array([4.0, 5.0, 6.0])
  let result = vec_dot(x, y)
  // 1*4 + 2*5 + 3*6 = 32
  assert_near(result, 32.0, 0.0001)
}

///|
test "vec_nrm2" {
  let x = vec_from_array([3.0, 4.0])
  let result = vec_nrm2(x)
  // sqrt(9 + 16) = 5
  assert_near(result, 5.0, 0.0001)
}

///|
test "matmul_vec_blas_into" {
  let mat = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let vec = vec_from_array([10.0, 20.0])
  let out = vec_new(3, 0.0)
  matmul_vec_blas_into(mat, vec, out)
  // out[j] = sum_i(vec[i] * mat[i, j])
  // out[0] = 10*1 + 20*4 = 90
  // out[1] = 10*2 + 20*5 = 120
  // out[2] = 10*3 + 20*6 = 150
  assert_eq(vec_to_array(out), [90.0, 120.0, 150.0])
}

///|
test "matmul_vec_bias_relu_blas" {
  let mat = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let vec = vec_from_array([10.0, 20.0])
  let bias = vec_from_array([1.0, 1.0, 1.0])
  let out = vec_new(3, 0.0)
  matmul_vec_bias_relu_blas_into(mat, vec, bias, out)
  assert_eq(vec_to_array(out), [91.0, 121.0, 151.0])
}

///|
test "batch_matmul" {
  let f = Float::from_int
  // Input: 2x3 (2 samples, 3 features)
  // Weight: 3x2 (3 input, 2 output)
  // Out: 2x2
  let input : Array[Float] = [f(1), f(2), f(3), f(4), f(5), f(6)]
  let weight : Array[Float] = [f(1), f(2), f(0), f(1), f(1), f(0)]
  let out : Array[Float] = [f(0), f(0), f(0), f(0)]
  batch_matmul(input, weight, out, 2, 3, 2)
  // Row 0: [1,2,3] @ [[1,2],[0,1],[1,0]] = [4, 4]
  // Row 1: [4,5,6] @ [[1,2],[0,1],[1,0]] = [10, 13]
  assert_eq(out, [f(4), f(4), f(10), f(13)])
}

///|
test "batch_matmul_bias_relu" {
  let f = Float::from_int
  let input : Array[Float] = [f(1), f(2), f(3), f(4), f(5), f(6)]
  let weight : Array[Float] = [f(1), f(2), f(0), f(1), f(1), f(0)]
  let bias : Array[Float] = [
    Float::from_int(1) / Float::from_int(2),
    Float::from_int(-10),
  ]
  let out : Array[Float] = [f(0), f(0), f(0), f(0)]
  batch_matmul_bias_relu(input, weight, bias, out, 2, 3, 2)
  // Before ReLU: [4.5, -6, 10.5, 3]
  // After ReLU: [4.5, 0, 10.5, 3]
  let half = Float::from_int(1) / Float::from_int(2)
  assert_eq(out, [f(4) + half, f(0), f(10) + half, f(3)])
}

