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

// ============================================================================
// Scalar operations tests
// ============================================================================

///|
test "vec_neg" {
  let v = vec_from_array([1.0, -2.0, 3.0])
  let neg_v = -v
  assert_eq(vec_to_array(neg_v), [-1.0, 2.0, -3.0])
}

///|
test "vec_add_scalar" {
  let v = vec_from_array([1.0, 2.0, 3.0])
  let result = vec_add_scalar(v, 10.0)
  assert_eq(vec_to_array(result), [11.0, 12.0, 13.0])
}

///|
test "vec_sub_scalar" {
  let v = vec_from_array([10.0, 20.0, 30.0])
  let result = vec_sub_scalar(v, 5.0)
  assert_eq(vec_to_array(result), [5.0, 15.0, 25.0])
}

///|
test "vec_mul_scalar" {
  let v = vec_from_array([1.0, 2.0, 3.0])
  let result = vec_mul_scalar(v, 2.0)
  assert_eq(vec_to_array(result), [2.0, 4.0, 6.0])
}

///|
test "vec_div_scalar" {
  let v = vec_from_array([10.0, 20.0, 30.0])
  let result = vec_div_scalar(v, 10.0)
  assert_eq(vec_to_array(result), [1.0, 2.0, 3.0])
}

///|
test "vec_scalar_inplace" {
  let v = vec_from_array([1.0, 2.0, 3.0])
  vec_mul_scalar_inplace(v, 2.0)
  assert_eq(vec_to_array(v), [2.0, 4.0, 6.0])
  vec_add_scalar_inplace(v, 1.0)
  assert_eq(vec_to_array(v), [3.0, 5.0, 7.0])
}

// ============================================================================
// Mat tests
// ============================================================================

///|
test "mat_new_and_access" {
  let m = mat_new(2, 3, 1.0)
  assert_eq(mat_rows(m), 2)
  assert_eq(mat_cols(m), 3)
  assert_eq(mat_at(m, 0, 0), 1.0)
  mat_set(m, 1, 2, 5.0)
  assert_eq(mat_at(m, 1, 2), 5.0)
}

///|
test "mat_zeros_ones" {
  let z = mat_zeros(2, 2)
  assert_eq(mat_to_array(z), [0.0, 0.0, 0.0, 0.0])
  let o = mat_ones(2, 2)
  assert_eq(mat_to_array(o), [1.0, 1.0, 1.0, 1.0])
}

///|
test "mat_eye" {
  let e = mat_eye(3)
  assert_eq(mat_to_array(e), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
}

///|
test "mat_transpose" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let t = mat_transpose(m)
  assert_eq(mat_rows(t), 3)
  assert_eq(mat_cols(t), 2)
  assert_eq(mat_to_array(t), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
}

///|
test "mat_ops_basic" {
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let b = mat_view([4.0, 3.0, 2.0, 1.0], 2, 2)
  let added = a + b
  let subbed = a - b
  let mulled = a * b
  assert_eq(mat_to_array(added), [5.0, 5.0, 5.0, 5.0])
  assert_eq(mat_to_array(subbed), [-3.0, -1.0, 1.0, 3.0])
  assert_eq(mat_to_array(mulled), [4.0, 6.0, 6.0, 4.0])
}

///|
test "mat_neg" {
  let m = mat_view([1.0, -2.0, 3.0, -4.0], 2, 2)
  let neg_m = -m
  assert_eq(mat_to_array(neg_m), [-1.0, 2.0, -3.0, 4.0])
}

///|
test "mat_scalar_ops" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let added = mat_add_scalar(m, 10.0)
  let mulled = mat_mul_scalar(m, 2.0)
  assert_eq(mat_to_array(added), [11.0, 12.0, 13.0, 14.0])
  assert_eq(mat_to_array(mulled), [2.0, 4.0, 6.0, 8.0])
}

///|
test "mat_row" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let row0 = mat_row(m, 0)
  let row1 = mat_row(m, 1)
  assert_eq(vec_to_array(row0), [1.0, 2.0, 3.0])
  assert_eq(vec_to_array(row1), [4.0, 5.0, 6.0])
}

///|
test "mat_matmul" {
  // A: 2x3, B: 3x2 -> C: 2x2
  let a = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  let b = mat_view([1.0, 2.0, 0.0, 1.0, 1.0, 0.0], 3, 2)
  let c = mat_matmul(a, b)
  // Row 0: [1,2,3] @ [[1,2],[0,1],[1,0]] = [4, 4]
  // Row 1: [4,5,6] @ [[1,2],[0,1],[1,0]] = [10, 13]
  assert_eq(mat_rows(c), 2)
  assert_eq(mat_cols(c), 2)
  assert_eq(mat_to_array(c), [4.0, 4.0, 10.0, 13.0])
}

// ============================================================================
// Array creation tests
// ============================================================================

///|
test "vec_zeros_ones" {
  let z = vec_zeros(3)
  let o = vec_ones(3)
  assert_eq(vec_to_array(z), [0.0, 0.0, 0.0])
  assert_eq(vec_to_array(o), [1.0, 1.0, 1.0])
}

///|
test "vec_from_fn" {
  let v = vec_from_fn(5, fn(i) { Float::from_int(i * i) })
  assert_eq(vec_to_array(v), [0.0, 1.0, 4.0, 9.0, 16.0])
}

///|
test "vec_arange" {
  // np.arange(0, 5, 1) -> [0, 1, 2, 3, 4]
  let v1 = vec_arange(0.0, 5.0, 1.0)
  assert_eq(vec_to_array(v1), [0.0, 1.0, 2.0, 3.0, 4.0])
  // np.arange(0, 1, 0.25) -> [0, 0.25, 0.5, 0.75]
  let v2 = vec_arange(0.0, 1.0, 0.25)
  assert_eq(vec_len(v2), 4)
  assert_near(vec_at(v2, 0), 0.0, 0.0001)
  assert_near(vec_at(v2, 1), 0.25, 0.0001)
  assert_near(vec_at(v2, 2), 0.5, 0.0001)
  assert_near(vec_at(v2, 3), 0.75, 0.0001)
}

///|
test "vec_linspace" {
  // np.linspace(0, 1, 5) -> [0, 0.25, 0.5, 0.75, 1.0]
  let v = vec_linspace(0.0, 1.0, 5)
  assert_eq(vec_len(v), 5)
  assert_near(vec_at(v, 0), 0.0, 0.0001)
  assert_near(vec_at(v, 1), 0.25, 0.0001)
  assert_near(vec_at(v, 2), 0.5, 0.0001)
  assert_near(vec_at(v, 3), 0.75, 0.0001)
  assert_near(vec_at(v, 4), 1.0, 0.0001)
}

// ============================================================================
// Math function tests
// ============================================================================

///|
test "vec_exp" {
  let v = vec_from_array([0.0, 1.0, 2.0])
  let r = vec_exp(v)
  assert_near(vec_at(r, 0), 1.0, 0.0001)
  assert_near(vec_at(r, 1), 2.7182, 0.001)
  assert_near(vec_at(r, 2), 7.3890, 0.001)
}

///|
test "vec_log" {
  let v = vec_from_array([1.0, 2.7182818, 7.3890561])
  let r = vec_log(v)
  assert_near(vec_at(r, 0), 0.0, 0.0001)
  assert_near(vec_at(r, 1), 1.0, 0.0001)
  assert_near(vec_at(r, 2), 2.0, 0.0001)
}

///|
test "vec_sqrt" {
  let v = vec_from_array([0.0, 1.0, 4.0, 9.0])
  let r = vec_sqrt(v)
  assert_eq(vec_to_array(r), [0.0, 1.0, 2.0, 3.0])
}

///|
test "vec_abs" {
  let v = vec_from_array([-3.0, -1.0, 0.0, 2.0])
  let r = vec_abs(v)
  assert_eq(vec_to_array(r), [3.0, 1.0, 0.0, 2.0])
}

///|
test "vec_pow" {
  let v = vec_from_array([1.0, 2.0, 3.0])
  let r = vec_pow(v, 2.0)
  assert_eq(vec_to_array(r), [1.0, 4.0, 9.0])
}

///|
test "vec_clip" {
  let v = vec_from_array([-1.0, 0.5, 2.0])
  let r = vec_clip(v, 0.0, 1.0)
  assert_eq(vec_to_array(r), [0.0, 0.5, 1.0])
}

///|
test "vec_trig" {
  let pi : Float = 3.14159265
  let half_pi = pi / Float::from_int(2)
  let v = vec_from_array([Float::from_int(0), half_pi, pi])
  let s = vec_sin(v)
  let c = vec_cos(v)
  assert_near(vec_at(s, 0), 0.0, 0.0001)
  assert_near(vec_at(s, 1), 1.0, 0.0001)
  assert_near(vec_at(s, 2), 0.0, 0.0001)
  assert_near(vec_at(c, 0), 1.0, 0.0001)
  assert_near(vec_at(c, 1), 0.0, 0.0001)
  assert_near(vec_at(c, 2), -1.0, 0.0001)
}

///|
test "vec_tanh" {
  let v = vec_from_array([0.0, 1.0, -1.0])
  let r = vec_tanh(v)
  assert_near(vec_at(r, 0), 0.0, 0.0001)
  assert_near(vec_at(r, 1), 0.7615, 0.0001)
  assert_near(vec_at(r, 2), -0.7615, 0.0001)
}

// ============================================================================
// Statistics tests
// ============================================================================

///|
test "vec_min_max" {
  let v = vec_from_array([3.0, 1.0, 4.0, 1.0, 5.0])
  assert_eq(vec_min(v), 1.0)
  assert_eq(vec_max(v), 5.0)
}

///|
test "vec_argmin" {
  let v = vec_from_array([3.0, 1.0, 4.0, 0.5, 5.0])
  assert_eq(vec_argmin(v), 3)
  assert_eq(vec_argmax(v), 4)
}

///|
test "vec_mean" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_eq(vec_mean(v), 3.0)
}

///|
test "vec_var_std" {
  // np.var([1, 2, 3, 4, 5]) = 2.0
  // np.std([1, 2, 3, 4, 5]) = sqrt(2) ≈ 1.4142
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_near(vec_var(v), 2.0, 0.0001)
  assert_near(vec_std(v), 1.4142, 0.0001)
}

///|
test "vec_prod" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0])
  assert_eq(vec_prod(v), 24.0)
}

// ============================================================================
// Mat statistics tests
// ============================================================================

///|
test "mat_sum_mean" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  // sum = 1+2+3+4+5+6 = 21
  assert_eq(mat_sum(m), 21.0)
  // mean = 21 / 6 = 3.5
  assert_eq(mat_mean(m), 3.5)
}

///|
test "mat_min_max" {
  let m = mat_view([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], 2, 3)
  assert_eq(mat_min(m), 1.0)
  assert_eq(mat_max(m), 9.0)
}

///|
test "mat_sum_axis" {
  // [[1, 2, 3],
  //  [4, 5, 6]]
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  // axis=0: [5, 7, 9]
  let axis0 = mat_sum_axis0(m)
  assert_eq(vec_to_array(axis0), [5.0, 7.0, 9.0])
  // axis=1: [6, 15]
  let axis1 = mat_sum_axis1(m)
  assert_eq(vec_to_array(axis1), [6.0, 15.0])
}

///|
test "mat_mean_axis" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
  // axis=0: [2.5, 3.5, 4.5]
  let axis0 = mat_mean_axis0(m)
  assert_eq(vec_to_array(axis0), [2.5, 3.5, 4.5])
  // axis=1: [2.0, 5.0]
  let axis1 = mat_mean_axis1(m)
  assert_eq(vec_to_array(axis1), [2.0, 5.0])
}

// ============================================================================
// Comparison tests
// ============================================================================

///|
test "vec_allclose" {
  let a = vec_from_array([1.0, 2.0, 3.0])
  let b = vec_from_array([1.0001, 1.9999, 3.0001])
  assert_true(vec_allclose(a, b, 0.001))
  assert_false(vec_allclose(a, b, 0.00001))
}

///|
test "mat_allclose" {
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let b = mat_view([1.0001, 1.9999, 3.0001, 3.9999], 2, 2)
  assert_true(mat_allclose(a, b, 0.001))
  assert_false(mat_allclose(a, b, 0.00001))
}

// ============================================================================
// Slice and reshape tests
// ============================================================================

///|
test "vec_slice" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0])
  let s = vec_slice(v, 1, 4)
  assert_eq(vec_to_array(s), [2.0, 3.0, 4.0])
}

///|
test "mat_flatten" {
  let m = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let v = mat_flatten(m)
  assert_eq(vec_to_array(v), [1.0, 2.0, 3.0, 4.0])
}

///|
test "vec_reshape" {
  let v = vec_from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let m = vec_reshape(v, 2, 3)
  assert_eq(mat_rows(m), 2)
  assert_eq(mat_cols(m), 3)
  assert_eq(mat_at(m, 0, 2), 3.0)
  assert_eq(mat_at(m, 1, 0), 4.0)
}

// ============================================================================
// NumPy-style integration tests
// ============================================================================

///|
test "numpy_style_linear_algebra" {
  // Test: A @ B = C
  // A = [[1, 0], [0, 1]] (identity)
  // B = [[1, 2], [3, 4]]
  // A @ B = B
  let a = mat_eye(2)
  let b = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let c = mat_matmul(a, b)
  assert_true(mat_allclose(b, c, 0.0001))
}

///|
test "numpy_style_broadcasting_scalar" {
  // Similar to np.array([1, 2, 3]) + 1 = [2, 3, 4]
  let v = vec_from_array([1.0, 2.0, 3.0])
  let result = vec_add_scalar(v, 1.0)
  assert_eq(vec_to_array(result), [2.0, 3.0, 4.0])
}

///|
test "numpy_style_chained_ops" {
  // np.exp(np.log([1, 2, 3])) ≈ [1, 2, 3]
  let v = vec_from_array([1.0, 2.0, 3.0])
  let result = vec_exp(vec_log(v))
  assert_true(vec_allclose(v, result, 0.0001))
}

///|
test "numpy_style_matrix_transpose_matmul" {
  // A^T @ A is symmetric positive definite
  let a = mat_view([1.0, 2.0, 3.0, 4.0], 2, 2)
  let at = mat_transpose(a)
  let ata = mat_matmul(at, a)
  // Check symmetry: ata[0,1] == ata[1,0]
  assert_eq(mat_at(ata, 0, 1), mat_at(ata, 1, 0))
}

///|
test "numpy_style_normalize" {
  // Normalize vector: v / ||v||
  let v = vec_from_array([3.0, 4.0])
  let norm = vec_nrm2(v)
  let normalized = vec_div_scalar(v, norm)
  // Should have unit length
  assert_near(vec_nrm2(normalized), 1.0, 0.0001)
}
