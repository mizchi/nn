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
pub fn mat_view(data : Array[Float], rows : Int, cols : Int) -> Mat {
  let mat = { data, rows, cols }
  mat_check_shape(mat, data)
  mat
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

// ============================================================================
// BLAS-accelerated operations
// ============================================================================

///|
/// Matrix-vector multiply using BLAS sgemv: out = mat^T @ vec
/// mat: rows x cols, vec: rows, out: cols (row-major)
/// We compute out[j] = sum_i(vec[i] * mat[i, j])
pub fn matmul_vec_blas_into(mat : Mat, vec : Vec, out : Vec) -> Unit {
  mat_check_vec_dims(mat, vec, out)
  // mat.data is row-major: [row0_col0, row0_col1, ..., row1_col0, ...]
  // dims: mat.rows x mat.cols, vec: mat.rows, out: mat.cols
  // We compute out[j] = sum_i(vec[i] * mat[i, j])
  // This is transposed sgemv: out = A^T @ vec
  @blas.sgemv_trans(mat.data, vec.data, out.data, mat.rows, mat.cols)
}

///|
/// Matrix-vector multiply + bias using BLAS: y = A @ x + bias
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
// Batch BLAS operations (for efficient multi-sample processing)
// ============================================================================

///|
/// Batch matrix multiply using BLAS sgemm: Out = Input @ Weight
/// Input: batch x in_dim, Weight: in_dim x out_dim, Out: batch x out_dim
pub fn batch_matmul_blas(
  input : Array[Float],
  weight : Array[Float],
  out : Array[Float],
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  // sgemm: C = A @ B where A is m x k, B is k x n, C is m x n
  // Here: A = Input (batch x in_dim), B = Weight (in_dim x out_dim), C = Out (batch x out_dim)
  @blas.sgemm(input, weight, out, batch, out_dim, in_dim)
}

///|
/// Batch matrix multiply + bias: Out[b, j] = sum_i(Input[b, i] * Weight[i, j]) + Bias[j]
pub fn batch_matmul_bias_blas(
  input : Array[Float],
  weight : Array[Float],
  bias : Array[Float],
  out : Array[Float],
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  batch_matmul_blas(input, weight, out, batch, in_dim, out_dim)
  // Add bias to each row
  for b = 0; b < batch; b = b + 1 {
    for j = 0; j < out_dim; j = j + 1 {
      out[b * out_dim + j] = out[b * out_dim + j] + bias[j]
    }
  }
}

///|
/// Batch matrix multiply + bias + ReLU
pub fn batch_matmul_bias_relu_blas(
  input : Array[Float],
  weight : Array[Float],
  bias : Array[Float],
  out : Array[Float],
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  batch_matmul_blas(input, weight, out, batch, in_dim, out_dim)
  // Add bias and apply ReLU to each element
  for b = 0; b < batch; b = b + 1 {
    for j = 0; j < out_dim; j = j + 1 {
      let idx = b * out_dim + j
      out[idx] = relu(out[idx] + bias[j])
    }
  }
}
