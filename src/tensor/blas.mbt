///|
#borrow(a, b, c)
extern "C" fn native_sgemm(
  trans_a : Int,
  trans_b : Int,
  m : Int,
  n : Int,
  k : Int,
  alpha : Float,
  a : FixedArray[Float],
  lda : Int,
  b : FixedArray[Float],
  ldb : Int,
  beta : Float,
  c : FixedArray[Float],
  ldc : Int,
) -> Unit = "tensor_sgemm"

///|
extern "C" fn native_clock_ns() -> UInt64 = "timer_clock_ns"

///|
pub fn clock_ns() -> UInt64 {
  native_clock_ns()
}

///|
#borrow(input, output)
extern "C" fn native_gelu_forward(
  input : FixedArray[Float],
  output : FixedArray[Float],
  n : Int,
) -> Unit = "tensor_gelu_forward"

///|
#borrow(dy, x_pre, dx)
extern "C" fn native_gelu_backward(
  dy : FixedArray[Float],
  x_pre : FixedArray[Float],
  dx : FixedArray[Float],
  n : Int,
) -> Unit = "tensor_gelu_backward"

///|
#borrow(data)
extern "C" fn native_softmax_inplace(
  data : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> Unit = "tensor_softmax_inplace"

///|
#borrow(input, output)
extern "C" fn native_log_softmax(
  input : FixedArray[Float],
  output : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> Unit = "tensor_log_softmax"

///|
#borrow(input, gamma, beta, output, mean_out, rstd_out)
extern "C" fn native_layer_norm_fwd(
  input : FixedArray[Float],
  gamma : FixedArray[Float],
  beta : FixedArray[Float],
  output : FixedArray[Float],
  mean_out : FixedArray[Float],
  rstd_out : FixedArray[Float],
  outer : Int,
  last_dim : Int,
  eps : Float,
) -> Unit = "tensor_layer_norm_fwd"

///|
#borrow(dy, x, mean, rstd, gamma, dx, d_gamma, d_beta)
extern "C" fn native_layer_norm_bwd(
  dy : FixedArray[Float],
  x : FixedArray[Float],
  mean : FixedArray[Float],
  rstd : FixedArray[Float],
  gamma : FixedArray[Float],
  dx : FixedArray[Float],
  d_gamma : FixedArray[Float],
  d_beta : FixedArray[Float],
  outer : Int,
  last_dim : Int,
) -> Unit = "tensor_layer_norm_bwd"

///|
#borrow(src, dst)
extern "C" fn native_reshape_for_heads(
  src : FixedArray[Float],
  dst : FixedArray[Float],
  batch : Int,
  seq : Int,
  num_heads : Int,
  d_k : Int,
) -> Unit = "tensor_reshape_for_heads"

///|
#borrow(src, dst)
extern "C" fn native_reshape_from_heads(
  src : FixedArray[Float],
  dst : FixedArray[Float],
  batch : Int,
  seq : Int,
  num_heads : Int,
  d_k : Int,
) -> Unit = "tensor_reshape_from_heads"

///|
pub fn sgemm(
  trans_a : Int,
  trans_b : Int,
  m : Int,
  n : Int,
  k : Int,
  alpha : Float,
  a : FixedArray[Float],
  lda : Int,
  b : FixedArray[Float],
  ldb : Int,
  beta : Float,
  c : FixedArray[Float],
  ldc : Int,
) -> Unit {
  native_sgemm(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

///|
#borrow(x, y)
extern "C" fn native_saxpy(
  n : Int,
  alpha : Float,
  x : FixedArray[Float],
  y : FixedArray[Float],
) -> Unit = "tensor_saxpy"

///|
/// y[i] += alpha * x[i] (BLAS saxpy)
pub fn saxpy(
  n : Int,
  alpha : Float,
  x : FixedArray[Float],
  y : FixedArray[Float],
) -> Unit {
  native_saxpy(n, alpha, x, y)
}

///|
#borrow(dy, x, dx)
extern "C" fn native_relu_backward(
  dy : FixedArray[Float],
  x : FixedArray[Float],
  dx : FixedArray[Float],
  n : Int,
) -> Unit = "tensor_relu_backward"

///|
pub fn relu_backward_inplace(
  dy : FixedArray[Float],
  x : FixedArray[Float],
  dx : FixedArray[Float],
  n : Int,
) -> Unit {
  native_relu_backward(dy, x, dx, n)
}

///|
#borrow(x, bias, out)
extern "C" fn native_bias_add_fwd(
  x : FixedArray[Float],
  bias : FixedArray[Float],
  out : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit = "tensor_bias_add_fwd"

///|
pub fn bias_add_forward(
  x : FixedArray[Float],
  bias : FixedArray[Float],
  out : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit {
  native_bias_add_fwd(x, bias, out, total, last_dim)
}

///|
#borrow(dy, db)
extern "C" fn native_bias_add_bwd(
  dy : FixedArray[Float],
  db : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit = "tensor_bias_add_bwd"

///|
pub fn bias_add_backward(
  dy : FixedArray[Float],
  db : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit {
  native_bias_add_bwd(dy, db, total, last_dim)
}

///|
#borrow(x)
extern "C" fn native_scale_inplace(
  x : FixedArray[Float],
  s : Float,
  n : Int,
) -> Unit = "tensor_scale_inplace"

///|
pub fn scale_inplace(x : FixedArray[Float], s : Float, n : Int) -> Unit {
  native_scale_inplace(x, s, n)
}

///|
#borrow(a, b, c)
extern "C" fn native_sgemm_offset(
  trans_a : Int,
  trans_b : Int,
  m : Int,
  n : Int,
  k : Int,
  alpha : Float,
  a : FixedArray[Float],
  a_offset : Int,
  lda : Int,
  b : FixedArray[Float],
  b_offset : Int,
  ldb : Int,
  beta : Float,
  c : FixedArray[Float],
  c_offset : Int,
  ldc : Int,
) -> Unit = "tensor_sgemm_offset"
