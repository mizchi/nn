///|
pub struct Vec {
  data : Array[Float]
  offset : Int
  len : Int
}

///|
pub struct Mat {
  data : Array[Float]
  rows : Int
  cols : Int
}

///|
fn vec_check_range(data : Array[Float], offset : Int, len : Int) -> Unit {
  if offset < 0 || len < 0 || offset + len > data.length() {
    panic()
  }
}

///|
fn vec_check_same_len(left : Vec, right : Vec) -> Unit {
  if left.len != right.len {
    panic()
  }
}

///|
fn vec_check_non_empty(vec : Vec) -> Unit {
  if vec.len <= 0 {
    panic()
  }
}

///|
pub fn vec_view(data : Array[Float], offset : Int, len : Int) -> Vec {
  vec_check_range(data, offset, len)
  { data, offset, len }
}

///|
pub fn vec_from_array(data : Array[Float]) -> Vec {
  vec_view(data, 0, data.length())
}

///|
pub fn vec_new(len : Int, value : Float) -> Vec {
  if len < 0 {
    panic()
  }
  vec_from_array(Array::make(len, value))
}

///|
pub fn vec_zeros(len : Int) -> Vec {
  vec_new(len, Float::from_int(0))
}

///|
pub fn vec_ones(len : Int) -> Vec {
  vec_new(len, Float::from_int(1))
}

///|
/// Create vec from initializer function: out[i] = f(i)
pub fn vec_from_fn(len : Int, f : (Int) -> Float) -> Vec {
  if len < 0 {
    panic()
  }
  vec_from_array(Array::makei(len, f))
}

///|
/// Create vec like numpy.arange: [start, stop) with step
pub fn vec_arange(start : Float, stop : Float, step : Float) -> Vec {
  if step == Float::from_int(0) {
    panic()
  }
  let len = ((stop - start) / step).to_int()
  if len <= 0 {
    return vec_from_array([])
  }
  vec_from_fn(len, fn(i) { start + Float::from_int(i) * step })
}

///|
/// Create vec like numpy.linspace: num evenly spaced values in [start, stop]
pub fn vec_linspace(start : Float, stop : Float, num : Int) -> Vec {
  if num <= 0 {
    panic()
  }
  if num == 1 {
    return vec_from_array([start])
  }
  let step = (stop - start) / Float::from_int(num - 1)
  vec_from_fn(num, fn(i) { start + Float::from_int(i) * step })
}

///|
pub fn vec_len(vec : Vec) -> Int {
  vec.len
}

///|
pub fn vec_at(vec : Vec, index : Int) -> Float {
  vec.data[vec.offset + index]
}

///|
pub fn vec_set(vec : Vec, index : Int, value : Float) -> Unit {
  vec.data[vec.offset + index] = value
}

///|
pub fn vec_fill(vec : Vec, value : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, value)
  }
}

///|
pub fn vec_to_array(vec : Vec) -> Array[Float] {
  Array::makei(vec.len, fn(i) { vec_at(vec, i) })
}

///|
pub fn vec_add_into(out : Vec, left : Vec, right : Vec) -> Unit {
  vec_check_same_len(out, left)
  vec_check_same_len(left, right)
  for i = 0; i < left.len; i = i + 1 {
    vec_set(out, i, vec_at(left, i) + vec_at(right, i))
  }
}

///|
pub fn vec_sub_into(out : Vec, left : Vec, right : Vec) -> Unit {
  vec_check_same_len(out, left)
  vec_check_same_len(left, right)
  for i = 0; i < left.len; i = i + 1 {
    vec_set(out, i, vec_at(left, i) - vec_at(right, i))
  }
}

///|
pub fn vec_mul_into(out : Vec, left : Vec, right : Vec) -> Unit {
  vec_check_same_len(out, left)
  vec_check_same_len(left, right)
  for i = 0; i < left.len; i = i + 1 {
    vec_set(out, i, vec_at(left, i) * vec_at(right, i))
  }
}

///|
pub fn vec_div_into(out : Vec, left : Vec, right : Vec) -> Unit {
  vec_check_same_len(out, left)
  vec_check_same_len(left, right)
  for i = 0; i < left.len; i = i + 1 {
    vec_set(out, i, vec_at(left, i) / vec_at(right, i))
  }
}

///|
pub fn vec_add(left : Vec, right : Vec) -> Vec {
  vec_check_same_len(left, right)
  let out = vec_new(left.len, Float::from_int(0))
  vec_add_into(out, left, right)
  out
}

///|
pub fn vec_sub(left : Vec, right : Vec) -> Vec {
  vec_check_same_len(left, right)
  let out = vec_new(left.len, Float::from_int(0))
  vec_sub_into(out, left, right)
  out
}

///|
pub fn vec_mul(left : Vec, right : Vec) -> Vec {
  vec_check_same_len(left, right)
  let out = vec_new(left.len, Float::from_int(0))
  vec_mul_into(out, left, right)
  out
}

///|
pub fn vec_div(left : Vec, right : Vec) -> Vec {
  vec_check_same_len(left, right)
  let out = vec_new(left.len, Float::from_int(0))
  vec_div_into(out, left, right)
  out
}

///|
pub fn vec_sum(vec : Vec) -> Float {
  let mut sum = Float::from_int(0)
  for i = 0; i < vec.len; i = i + 1 {
    sum = sum + vec_at(vec, i)
  }
  sum
}

///|
pub fn vec_max(vec : Vec) -> Float {
  vec_check_non_empty(vec)
  let mut max = vec_at(vec, 0)
  for i = 1; i < vec.len; i = i + 1 {
    let v = vec_at(vec, i)
    if v > max {
      max = v
    }
  }
  max
}

///|
pub fn vec_argmax(vec : Vec) -> Int {
  vec_check_non_empty(vec)
  let mut max = vec_at(vec, 0)
  let mut index = 0
  for i = 1; i < vec.len; i = i + 1 {
    let v = vec_at(vec, i)
    if v > max {
      max = v
      index = i
    }
  }
  index
}

///|
pub fn vec_min(vec : Vec) -> Float {
  vec_check_non_empty(vec)
  let mut min = vec_at(vec, 0)
  for i = 1; i < vec.len; i = i + 1 {
    let v = vec_at(vec, i)
    if v < min {
      min = v
    }
  }
  min
}

///|
pub fn vec_argmin(vec : Vec) -> Int {
  vec_check_non_empty(vec)
  let mut min = vec_at(vec, 0)
  let mut index = 0
  for i = 1; i < vec.len; i = i + 1 {
    let v = vec_at(vec, i)
    if v < min {
      min = v
      index = i
    }
  }
  index
}

///|
pub fn vec_mean(vec : Vec) -> Float {
  vec_check_non_empty(vec)
  vec_sum(vec) / Float::from_int(vec.len)
}

///|
/// Variance: E[(X - mean)^2]
pub fn vec_var(vec : Vec) -> Float {
  vec_check_non_empty(vec)
  let mean = vec_mean(vec)
  let mut sum_sq = Float::from_int(0)
  for i = 0; i < vec.len; i = i + 1 {
    let diff = vec_at(vec, i) - mean
    sum_sq = sum_sq + diff * diff
  }
  sum_sq / Float::from_int(vec.len)
}

///|
/// Standard deviation: sqrt(variance)
pub fn vec_std(vec : Vec) -> Float {
  vec_var(vec).sqrt()
}

///|
/// Product of all elements
pub fn vec_prod(vec : Vec) -> Float {
  vec_check_non_empty(vec)
  let mut prod = Float::from_int(1)
  for i = 0; i < vec.len; i = i + 1 {
    prod = prod * vec_at(vec, i)
  }
  prod
}

///|
pub fn vec_scale_inplace(vec : Vec, scale : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, vec_at(vec, i) * scale)
  }
}

///|
pub fn vec_exp_into(out : Vec, input : Vec) -> Unit {
  vec_check_same_len(out, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(out, i, @math.expf(vec_at(input, i)))
  }
}

///|
pub fn vec_exp(input : Vec) -> Vec {
  let out = vec_zeros(input.len)
  vec_exp_into(out, input)
  out
}

///|
pub fn vec_log_into(out : Vec, input : Vec) -> Unit {
  vec_check_same_len(out, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(out, i, @math.lnf(vec_at(input, i)))
  }
}

///|
pub fn vec_log(input : Vec) -> Vec {
  let out = vec_zeros(input.len)
  vec_log_into(out, input)
  out
}

///|
pub fn vec_sqrt_into(out : Vec, input : Vec) -> Unit {
  vec_check_same_len(out, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(out, i, vec_at(input, i).sqrt())
  }
}

///|
pub fn vec_sqrt(input : Vec) -> Vec {
  let out = vec_zeros(input.len)
  vec_sqrt_into(out, input)
  out
}

///|
pub fn vec_abs_into(out : Vec, input : Vec) -> Unit {
  vec_check_same_len(out, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(out, i, vec_at(input, i).abs())
  }
}

///|
pub fn vec_abs(input : Vec) -> Vec {
  let out = vec_zeros(input.len)
  vec_abs_into(out, input)
  out
}

///|
pub fn vec_pow_into(out : Vec, input : Vec, exp : Float) -> Unit {
  vec_check_same_len(out, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(out, i, @math.powf(vec_at(input, i), exp))
  }
}

///|
pub fn vec_pow(input : Vec, exp : Float) -> Vec {
  let out = vec_zeros(input.len)
  vec_pow_into(out, input, exp)
  out
}

///|
/// Clip values to [min, max] range
pub fn vec_clip_into(out : Vec, input : Vec, min : Float, max : Float) -> Unit {
  vec_check_same_len(out, input)
  for i = 0; i < input.len; i = i + 1 {
    let v = vec_at(input, i)
    let clipped = if v < min { min } else if v > max { max } else { v }
    vec_set(out, i, clipped)
  }
}

///|
pub fn vec_clip(input : Vec, min : Float, max : Float) -> Vec {
  let out = vec_zeros(input.len)
  vec_clip_into(out, input, min, max)
  out
}

///|
pub fn vec_sin_into(out : Vec, input : Vec) -> Unit {
  vec_check_same_len(out, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(out, i, @math.sinf(vec_at(input, i)))
  }
}

///|
pub fn vec_sin(input : Vec) -> Vec {
  let out = vec_zeros(input.len)
  vec_sin_into(out, input)
  out
}

///|
pub fn vec_cos_into(out : Vec, input : Vec) -> Unit {
  vec_check_same_len(out, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(out, i, @math.cosf(vec_at(input, i)))
  }
}

///|
pub fn vec_cos(input : Vec) -> Vec {
  let out = vec_zeros(input.len)
  vec_cos_into(out, input)
  out
}

///|
pub fn vec_tanh_into(out : Vec, input : Vec) -> Unit {
  vec_check_same_len(out, input)
  for i = 0; i < input.len; i = i + 1 {
    vec_set(out, i, @math.tanhf(vec_at(input, i)))
  }
}

///|
pub fn vec_tanh(input : Vec) -> Vec {
  let out = vec_zeros(input.len)
  vec_tanh_into(out, input)
  out
}

///|
pub fn relu(value : Float) -> Float {
  let zero = Float::from_int(0)
  if value < zero {
    zero
  } else {
    value
  }
}

///|
pub fn relu_grad(value : Float) -> Float {
  let zero = Float::from_int(0)
  if value > zero {
    Float::from_int(1)
  } else {
    zero
  }
}

///|
pub fn softmax_into(logits : Vec, probs : Vec) -> Unit {
  vec_check_same_len(logits, probs)
  let max = vec_max(logits)
  let mut sum = Float::from_int(0)
  for i = 0; i < logits.len; i = i + 1 {
    let v = @math.expf(vec_at(logits, i) - max)
    vec_set(probs, i, v)
    sum = sum + v
  }
  for i = 0; i < logits.len; i = i + 1 {
    vec_set(probs, i, vec_at(probs, i) / sum)
  }
}

///|
pub fn cross_entropy_loss(probs : Vec, label : Int) -> Float {
  let eps = Float::from_int(1) / Float::from_int(1000000)
  let p = vec_at(probs, label) + eps
  -@math.lnf(p)
}

///|
fn mat_check_shape(mat : Mat, data : Array[Float]) -> Unit {
  if mat.rows <= 0 || mat.cols <= 0 {
    panic()
  }
  if mat.rows * mat.cols != data.length() {
    panic()
  }
}

///|
fn mat_check_vec_dims(mat : Mat, vec : Vec, out : Vec) -> Unit {
  if vec.len != mat.rows || out.len != mat.cols {
    panic()
  }
}

///|
fn mat_check_same_shape(a : Mat, b : Mat) -> Unit {
  if a.rows != b.rows || a.cols != b.cols {
    panic()
  }
}

///|
pub fn mat_view(data : Array[Float], rows : Int, cols : Int) -> Mat {
  let mat = { data, rows, cols }
  mat_check_shape(mat, data)
  mat
}

///|
pub fn mat_new(rows : Int, cols : Int, value : Float) -> Mat {
  if rows <= 0 || cols <= 0 {
    panic()
  }
  mat_view(Array::make(rows * cols, value), rows, cols)
}

///|
pub fn mat_zeros(rows : Int, cols : Int) -> Mat {
  mat_new(rows, cols, Float::from_int(0))
}

///|
pub fn mat_ones(rows : Int, cols : Int) -> Mat {
  mat_new(rows, cols, Float::from_int(1))
}

///|
pub fn mat_eye(n : Int) -> Mat {
  if n <= 0 {
    panic()
  }
  let data = Array::make(n * n, Float::from_int(0))
  for i = 0; i < n; i = i + 1 {
    data[i * n + i] = Float::from_int(1)
  }
  mat_view(data, n, n)
}

///|
pub fn mat_rows(mat : Mat) -> Int {
  mat.rows
}

///|
pub fn mat_cols(mat : Mat) -> Int {
  mat.cols
}

///|
pub fn mat_at(mat : Mat, row : Int, col : Int) -> Float {
  mat.data[row * mat.cols + col]
}

///|
pub fn mat_set(mat : Mat, row : Int, col : Int, value : Float) -> Unit {
  mat.data[row * mat.cols + col] = value
}

///|
pub fn mat_fill(mat : Mat, value : Float) -> Unit {
  for i = 0; i < mat.data.length(); i = i + 1 {
    mat.data[i] = value
  }
}

///|
pub fn mat_to_array(mat : Mat) -> Array[Float] {
  Array::makei(mat.data.length(), fn(i) { mat.data[i] })
}

///|
pub fn mat_clone(mat : Mat) -> Mat {
  mat_view(mat_to_array(mat), mat.rows, mat.cols)
}

///|
/// Get row as Vec (view into original data)
pub fn mat_row(mat : Mat, row : Int) -> Vec {
  vec_view(mat.data, row * mat.cols, mat.cols)
}

///|
/// Transpose matrix (creates new matrix)
pub fn mat_transpose(mat : Mat) -> Mat {
  let out = mat_zeros(mat.cols, mat.rows)
  for i = 0; i < mat.rows; i = i + 1 {
    for j = 0; j < mat.cols; j = j + 1 {
      mat_set(out, j, i, mat_at(mat, i, j))
    }
  }
  out
}

///|
pub fn mat_add_into(out : Mat, a : Mat, b : Mat) -> Unit {
  mat_check_same_shape(out, a)
  mat_check_same_shape(a, b)
  for i = 0; i < a.data.length(); i = i + 1 {
    out.data[i] = a.data[i] + b.data[i]
  }
}

///|
pub fn mat_sub_into(out : Mat, a : Mat, b : Mat) -> Unit {
  mat_check_same_shape(out, a)
  mat_check_same_shape(a, b)
  for i = 0; i < a.data.length(); i = i + 1 {
    out.data[i] = a.data[i] - b.data[i]
  }
}

///|
pub fn mat_mul_into(out : Mat, a : Mat, b : Mat) -> Unit {
  mat_check_same_shape(out, a)
  mat_check_same_shape(a, b)
  for i = 0; i < a.data.length(); i = i + 1 {
    out.data[i] = a.data[i] * b.data[i]
  }
}

///|
pub fn mat_div_into(out : Mat, a : Mat, b : Mat) -> Unit {
  mat_check_same_shape(out, a)
  mat_check_same_shape(a, b)
  for i = 0; i < a.data.length(); i = i + 1 {
    out.data[i] = a.data[i] / b.data[i]
  }
}

///|
pub fn mat_add(a : Mat, b : Mat) -> Mat {
  mat_check_same_shape(a, b)
  let out = mat_zeros(a.rows, a.cols)
  mat_add_into(out, a, b)
  out
}

///|
pub fn mat_sub(a : Mat, b : Mat) -> Mat {
  mat_check_same_shape(a, b)
  let out = mat_zeros(a.rows, a.cols)
  mat_sub_into(out, a, b)
  out
}

///|
pub fn mat_mul(a : Mat, b : Mat) -> Mat {
  mat_check_same_shape(a, b)
  let out = mat_zeros(a.rows, a.cols)
  mat_mul_into(out, a, b)
  out
}

///|
pub fn mat_div(a : Mat, b : Mat) -> Mat {
  mat_check_same_shape(a, b)
  let out = mat_zeros(a.rows, a.cols)
  mat_div_into(out, a, b)
  out
}

///|
pub impl Add for Mat with add(self : Mat, other : Mat) -> Mat {
  mat_add(self, other)
}

///|
pub impl Sub for Mat with sub(self : Mat, other : Mat) -> Mat {
  mat_sub(self, other)
}

///|
pub impl Mul for Mat with mul(self : Mat, other : Mat) -> Mat {
  mat_mul(self, other)
}

///|
pub impl Div for Mat with div(self : Mat, other : Mat) -> Mat {
  mat_div(self, other)
}

///|
pub impl Neg for Mat with neg(self : Mat) -> Mat {
  mat_neg(self)
}

///|
pub fn mat_neg(mat : Mat) -> Mat {
  let out = mat_zeros(mat.rows, mat.cols)
  for i = 0; i < mat.data.length(); i = i + 1 {
    out.data[i] = -mat.data[i]
  }
  out
}

///|
pub fn mat_add_scalar(mat : Mat, scalar : Float) -> Mat {
  let out = mat_zeros(mat.rows, mat.cols)
  for i = 0; i < mat.data.length(); i = i + 1 {
    out.data[i] = mat.data[i] + scalar
  }
  out
}

///|
pub fn mat_sub_scalar(mat : Mat, scalar : Float) -> Mat {
  let out = mat_zeros(mat.rows, mat.cols)
  for i = 0; i < mat.data.length(); i = i + 1 {
    out.data[i] = mat.data[i] - scalar
  }
  out
}

///|
pub fn mat_mul_scalar(mat : Mat, scalar : Float) -> Mat {
  let out = mat_zeros(mat.rows, mat.cols)
  for i = 0; i < mat.data.length(); i = i + 1 {
    out.data[i] = mat.data[i] * scalar
  }
  out
}

///|
pub fn mat_div_scalar(mat : Mat, scalar : Float) -> Mat {
  let out = mat_zeros(mat.rows, mat.cols)
  for i = 0; i < mat.data.length(); i = i + 1 {
    out.data[i] = mat.data[i] / scalar
  }
  out
}

///|
pub fn mat_scale_inplace(mat : Mat, scalar : Float) -> Unit {
  for i = 0; i < mat.data.length(); i = i + 1 {
    mat.data[i] = mat.data[i] * scalar
  }
}

///|
/// Matrix multiplication using BLAS: C = A @ B
pub fn mat_matmul(a : Mat, b : Mat) -> Mat {
  if a.cols != b.rows {
    panic()
  }
  let out = mat_zeros(a.rows, b.cols)
  @blas.sgemm(a.data, b.data, out.data, a.rows, b.cols, a.cols)
  out
}

///|
/// Sum of all elements
pub fn mat_sum(mat : Mat) -> Float {
  let mut sum = Float::from_int(0)
  for i = 0; i < mat.data.length(); i = i + 1 {
    sum = sum + mat.data[i]
  }
  sum
}

///|
/// Mean of all elements
pub fn mat_mean(mat : Mat) -> Float {
  mat_sum(mat) / Float::from_int(mat.data.length())
}

///|
/// Max of all elements
pub fn mat_max(mat : Mat) -> Float {
  if mat.data.length() == 0 {
    panic()
  }
  let mut max = mat.data[0]
  for i = 1; i < mat.data.length(); i = i + 1 {
    if mat.data[i] > max {
      max = mat.data[i]
    }
  }
  max
}

///|
/// Min of all elements
pub fn mat_min(mat : Mat) -> Float {
  if mat.data.length() == 0 {
    panic()
  }
  let mut min = mat.data[0]
  for i = 1; i < mat.data.length(); i = i + 1 {
    if mat.data[i] < min {
      min = mat.data[i]
    }
  }
  min
}

///|
/// Sum along axis 0 (columns): result[j] = sum_i(mat[i,j])
pub fn mat_sum_axis0(mat : Mat) -> Vec {
  let out = vec_zeros(mat.cols)
  for i = 0; i < mat.rows; i = i + 1 {
    for j = 0; j < mat.cols; j = j + 1 {
      vec_set(out, j, vec_at(out, j) + mat_at(mat, i, j))
    }
  }
  out
}

///|
/// Sum along axis 1 (rows): result[i] = sum_j(mat[i,j])
pub fn mat_sum_axis1(mat : Mat) -> Vec {
  let out = vec_zeros(mat.rows)
  for i = 0; i < mat.rows; i = i + 1 {
    let mut sum = Float::from_int(0)
    for j = 0; j < mat.cols; j = j + 1 {
      sum = sum + mat_at(mat, i, j)
    }
    vec_set(out, i, sum)
  }
  out
}

///|
/// Mean along axis 0
pub fn mat_mean_axis0(mat : Mat) -> Vec {
  let sums = mat_sum_axis0(mat)
  vec_div_scalar(sums, Float::from_int(mat.rows))
}

///|
/// Mean along axis 1
pub fn mat_mean_axis1(mat : Mat) -> Vec {
  let sums = mat_sum_axis1(mat)
  vec_div_scalar(sums, Float::from_int(mat.cols))
}

// ============================================================================
// Comparison functions
// ============================================================================

///|
/// Check if all elements are equal within tolerance
pub fn vec_allclose(a : Vec, b : Vec, atol : Float) -> Bool {
  if a.len != b.len {
    return false
  }
  for i = 0; i < a.len; i = i + 1 {
    if (vec_at(a, i) - vec_at(b, i)).abs() > atol {
      return false
    }
  }
  true
}

///|
/// Check if all elements are equal within tolerance
pub fn mat_allclose(a : Mat, b : Mat, atol : Float) -> Bool {
  if a.rows != b.rows || a.cols != b.cols {
    return false
  }
  for i = 0; i < a.data.length(); i = i + 1 {
    if (a.data[i] - b.data[i]).abs() > atol {
      return false
    }
  }
  true
}

///|
/// Check if any element is NaN
pub fn vec_has_nan(v : Vec) -> Bool {
  for i = 0; i < v.len; i = i + 1 {
    if vec_at(v, i).is_nan() {
      return true
    }
  }
  false
}

///|
/// Check if any element is infinite
pub fn vec_has_inf(v : Vec) -> Bool {
  for i = 0; i < v.len; i = i + 1 {
    if vec_at(v, i).is_inf() {
      return true
    }
  }
  false
}

// ============================================================================
// Slice and reshape
// ============================================================================

///|
/// Create a slice of vec [start, end)
pub fn vec_slice(v : Vec, start : Int, end : Int) -> Vec {
  if start < 0 || end > v.len || start >= end {
    panic()
  }
  vec_view(v.data, v.offset + start, end - start)
}

///|
/// Flatten mat to vec (row-major order)
pub fn mat_flatten(mat : Mat) -> Vec {
  vec_from_array(mat_to_array(mat))
}

///|
/// Reshape vec to mat (must have matching element count)
pub fn vec_reshape(v : Vec, rows : Int, cols : Int) -> Mat {
  if rows * cols != v.len {
    panic()
  }
  mat_view(vec_to_array(v), rows, cols)
}

///|
pub fn matmul_vec_into(mat : Mat, vec : Vec, out : Vec) -> Unit {
  mat_check_vec_dims(mat, vec, out)
  for j = 0; j < mat.cols; j = j + 1 {
    let mut acc = Float::from_int(0)
    for i = 0; i < mat.rows; i = i + 1 {
      acc = acc + vec_at(vec, i) * mat.data[i * mat.cols + j]
    }
    vec_set(out, j, acc)
  }
}

///|
pub fn matmul_vec_bias_into(
  mat : Mat,
  vec : Vec,
  bias : Vec,
  out : Vec,
) -> Unit {
  mat_check_vec_dims(mat, vec, out)
  vec_check_same_len(bias, out)
  for j = 0; j < mat.cols; j = j + 1 {
    let mut acc = vec_at(bias, j)
    for i = 0; i < mat.rows; i = i + 1 {
      acc = acc + vec_at(vec, i) * mat.data[i * mat.cols + j]
    }
    vec_set(out, j, acc)
  }
}

///|
pub fn matmul_vec_bias_relu_into(
  mat : Mat,
  vec : Vec,
  bias : Vec,
  out : Vec,
) -> Unit {
  mat_check_vec_dims(mat, vec, out)
  vec_check_same_len(bias, out)
  for j = 0; j < mat.cols; j = j + 1 {
    let mut acc = vec_at(bias, j)
    for i = 0; i < mat.rows; i = i + 1 {
      acc = acc + vec_at(vec, i) * mat.data[i * mat.cols + j]
    }
    vec_set(out, j, relu(acc))
  }
}

///|
pub impl Add for Vec with add(self : Vec, other : Vec) -> Vec {
  vec_add(self, other)
}

///|
pub impl Sub for Vec with sub(self : Vec, other : Vec) -> Vec {
  vec_sub(self, other)
}

///|
pub impl Mul for Vec with mul(self : Vec, other : Vec) -> Vec {
  vec_mul(self, other)
}

///|
pub impl Div for Vec with div(self : Vec, other : Vec) -> Vec {
  vec_div(self, other)
}

///|
pub impl Neg for Vec with neg(self : Vec) -> Vec {
  vec_neg(self)
}

// ============================================================================
// Scalar operations
// ============================================================================

///|
/// Negate all elements: out = -input
pub fn vec_neg(input : Vec) -> Vec {
  let out = vec_new(input.len, Float::from_int(0))
  for i = 0; i < input.len; i = i + 1 {
    vec_set(out, i, -vec_at(input, i))
  }
  out
}

///|
/// Add scalar to all elements: out = vec + scalar
pub fn vec_add_scalar(vec : Vec, scalar : Float) -> Vec {
  let out = vec_new(vec.len, Float::from_int(0))
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(out, i, vec_at(vec, i) + scalar)
  }
  out
}

///|
/// Subtract scalar from all elements: out = vec - scalar
pub fn vec_sub_scalar(vec : Vec, scalar : Float) -> Vec {
  let out = vec_new(vec.len, Float::from_int(0))
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(out, i, vec_at(vec, i) - scalar)
  }
  out
}

///|
/// Multiply all elements by scalar: out = vec * scalar
pub fn vec_mul_scalar(vec : Vec, scalar : Float) -> Vec {
  let out = vec_new(vec.len, Float::from_int(0))
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(out, i, vec_at(vec, i) * scalar)
  }
  out
}

///|
/// Divide all elements by scalar: out = vec / scalar
pub fn vec_div_scalar(vec : Vec, scalar : Float) -> Vec {
  let out = vec_new(vec.len, Float::from_int(0))
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(out, i, vec_at(vec, i) / scalar)
  }
  out
}

///|
/// In-place negate: vec = -vec
pub fn vec_neg_inplace(vec : Vec) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, -vec_at(vec, i))
  }
}

///|
/// In-place add scalar: vec = vec + scalar
pub fn vec_add_scalar_inplace(vec : Vec, scalar : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, vec_at(vec, i) + scalar)
  }
}

///|
/// In-place sub scalar: vec = vec - scalar
pub fn vec_sub_scalar_inplace(vec : Vec, scalar : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, vec_at(vec, i) - scalar)
  }
}

///|
/// In-place mul scalar: vec = vec * scalar
pub fn vec_mul_scalar_inplace(vec : Vec, scalar : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, vec_at(vec, i) * scalar)
  }
}

///|
/// In-place div scalar: vec = vec / scalar
pub fn vec_div_scalar_inplace(vec : Vec, scalar : Float) -> Unit {
  for i = 0; i < vec.len; i = i + 1 {
    vec_set(vec, i, vec_at(vec, i) / scalar)
  }
}

// ============================================================================
// BLAS Level 1 (vector-vector)
// ============================================================================

///|
/// Dot product: x · y
pub fn vec_dot(x : Vec, y : Vec) -> Float {
  vec_check_same_len(x, y)
  @blas.sdot(x.data, y.data)
}

///|
/// L2 norm: ||x||_2
pub fn vec_nrm2(x : Vec) -> Float {
  @blas.snrm2(x.data)
}

///|
/// AXPY: y = α*x + y
pub fn vec_axpy(alpha : Float, x : Vec, y : Vec) -> Unit {
  vec_check_same_len(x, y)
  @blas.saxpy(alpha, x.data, y.data)
}

// ============================================================================
// BLAS Level 2 (matrix-vector)
// ============================================================================

///|
/// Matrix-vector multiply using BLAS sgemv: out = mat^T @ vec
pub fn matmul_vec_blas_into(mat : Mat, vec : Vec, out : Vec) -> Unit {
  mat_check_vec_dims(mat, vec, out)
  @blas.sgemv_trans(mat.data, vec.data, out.data, mat.rows, mat.cols)
}

///|
/// Matrix-vector multiply + bias using BLAS
pub fn matmul_vec_bias_blas_into(
  mat : Mat,
  vec : Vec,
  bias : Vec,
  out : Vec,
) -> Unit {
  matmul_vec_blas_into(mat, vec, out)
  for i = 0; i < out.len; i = i + 1 {
    vec_set(out, i, vec_at(out, i) + vec_at(bias, i))
  }
}

///|
/// Matrix-vector multiply + bias + ReLU using BLAS
pub fn matmul_vec_bias_relu_blas_into(
  mat : Mat,
  vec : Vec,
  bias : Vec,
  out : Vec,
) -> Unit {
  matmul_vec_blas_into(mat, vec, out)
  for i = 0; i < out.len; i = i + 1 {
    vec_set(out, i, relu(vec_at(out, i) + vec_at(bias, i)))
  }
}

// ============================================================================
// BLAS Level 3 (matrix-matrix) - Batch operations
// ============================================================================

///|
/// Batch matrix multiply using BLAS sgemm: Out = Input @ Weight
/// Input: batch x in_dim, Weight: in_dim x out_dim, Out: batch x out_dim
pub fn batch_matmul(
  input : Array[Float],
  weight : Array[Float],
  out : Array[Float],
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  @blas.sgemm(input, weight, out, batch, out_dim, in_dim)
}

///|
/// Batch matrix multiply + bias
pub fn batch_matmul_bias(
  input : Array[Float],
  weight : Array[Float],
  bias : Array[Float],
  out : Array[Float],
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  batch_matmul(input, weight, out, batch, in_dim, out_dim)
  for b = 0; b < batch; b = b + 1 {
    for j = 0; j < out_dim; j = j + 1 {
      out[b * out_dim + j] = out[b * out_dim + j] + bias[j]
    }
  }
}

///|
/// Batch matrix multiply + bias + ReLU
pub fn batch_matmul_bias_relu(
  input : Array[Float],
  weight : Array[Float],
  bias : Array[Float],
  out : Array[Float],
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  batch_matmul(input, weight, out, batch, in_dim, out_dim)
  for b = 0; b < batch; b = b + 1 {
    for j = 0; j < out_dim; j = j + 1 {
      let idx = b * out_dim + j
      out[idx] = relu(out[idx] + bias[j])
    }
  }
}
