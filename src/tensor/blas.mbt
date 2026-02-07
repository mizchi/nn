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
#borrow(y, bias)
extern "C" fn native_bias_add_inplace(
  y : FixedArray[Float],
  bias : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit = "tensor_bias_add_inplace"

///|
pub fn bias_add_inplace(
  y : FixedArray[Float],
  bias : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit {
  native_bias_add_inplace(y, bias, total, last_dim)
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
#borrow(x, out)
extern "C" fn native_relu_forward(
  x : FixedArray[Float],
  out : FixedArray[Float],
  n : Int,
) -> Unit = "tensor_relu_forward"

///|
pub fn relu_forward(
  x : FixedArray[Float],
  out : FixedArray[Float],
  n : Int,
) -> Unit {
  native_relu_forward(x, out, n)
}

///|
#borrow(dst, src)
extern "C" fn native_accumulate(
  dst : FixedArray[Float],
  src : FixedArray[Float],
  n : Int,
) -> Unit = "tensor_accumulate"

///|
/// Softmax in-place over rows: data is [rows, cols], each row is softmaxed.
pub fn softmax_inplace(
  data : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> Unit {
  native_softmax_inplace(data, rows, cols)
}

///|
/// dst[i] += src[i] (BLAS saxpy with alpha=1)
pub fn accumulate_inplace(
  dst : FixedArray[Float],
  src : FixedArray[Float],
  n : Int,
) -> Unit {
  native_accumulate(dst, src, n)
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

///|
/// sgemm with all buffers as offsets into (possibly same) FixedArrays.
pub fn sgemm_offset(
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
) -> Unit {
  native_sgemm_offset(
    trans_a, trans_b, m, n, k, alpha, a, a_offset, lda, b, b_offset, ldb, beta, c,
    c_offset, ldc,
  )
}

///|
#borrow(arena)
extern "C" fn native_relu_backward_arena(
  arena : FixedArray[Float],
  dy_off : Int,
  x_off : Int,
  dx_off : Int,
  n : Int,
) -> Unit = "tensor_relu_backward_arena"

///|
pub fn relu_backward_arena(
  arena : FixedArray[Float],
  dy_off : Int,
  x_off : Int,
  dx_off : Int,
  n : Int,
) -> Unit {
  native_relu_backward_arena(arena, dy_off, x_off, dx_off, n)
}

///|
#borrow(dy, arena)
extern "C" fn native_relu_backward_hybrid(
  dy : FixedArray[Float],
  arena : FixedArray[Float],
  x_off : Int,
  dx_off : Int,
  n : Int,
) -> Unit = "tensor_relu_backward_hybrid"

///|
/// ReLU backward: dy from separate buffer, x and dx from arena offsets.
pub fn relu_backward_hybrid(
  dy : FixedArray[Float],
  arena : FixedArray[Float],
  x_off : Int,
  dx_off : Int,
  n : Int,
) -> Unit {
  native_relu_backward_hybrid(dy, arena, x_off, dx_off, n)
}

///|
#borrow(arena)
extern "C" fn native_relu_forward_arena(
  arena : FixedArray[Float],
  x_off : Int,
  out_off : Int,
  n : Int,
) -> Unit = "tensor_relu_forward_arena"

///|
pub fn relu_forward_arena(
  arena : FixedArray[Float],
  x_off : Int,
  out_off : Int,
  n : Int,
) -> Unit {
  native_relu_forward_arena(arena, x_off, out_off, n)
}

///|
#borrow(arena, out)
extern "C" fn native_relu_forward_hybrid(
  arena : FixedArray[Float],
  x_off : Int,
  out : FixedArray[Float],
  n : Int,
) -> Unit = "tensor_relu_forward_hybrid"

///|
/// ReLU forward: x from arena at offset, output to separate buffer.
pub fn relu_forward_hybrid(
  arena : FixedArray[Float],
  x_off : Int,
  out : FixedArray[Float],
  n : Int,
) -> Unit {
  native_relu_forward_hybrid(arena, x_off, out, n)
}

///|
#borrow(arena, bias)
extern "C" fn native_bias_add_inplace_arena(
  arena : FixedArray[Float],
  y_off : Int,
  bias : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit = "tensor_bias_add_inplace_arena"

///|
pub fn bias_add_inplace_arena(
  arena : FixedArray[Float],
  y_off : Int,
  bias : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit {
  native_bias_add_inplace_arena(arena, y_off, bias, total, last_dim)
}

///|
#borrow(arena, db)
extern "C" fn native_bias_add_bwd_arena(
  arena : FixedArray[Float],
  dy_off : Int,
  db : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit = "tensor_bias_add_bwd_arena"

///|
pub fn bias_add_backward_arena(
  arena : FixedArray[Float],
  dy_off : Int,
  db : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit {
  native_bias_add_bwd_arena(arena, dy_off, db, total, last_dim)
}

// ========== C-managed ReLU workspace ==========

///|
#borrow(a, b)
extern "C" fn native_sgemm_to_relu_pre(
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
  total : Int,
) -> Unit = "tensor_sgemm_to_relu_pre"

///|
/// sgemm that writes output to C-managed relu_pre buffer.
pub fn sgemm_to_relu_pre(
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
  total : Int,
) -> Unit {
  native_sgemm_to_relu_pre(
    trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, total,
  )
}

///|
#borrow(bias)
extern "C" fn native_bias_add_relu_pre(
  bias : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit = "tensor_bias_add_relu_pre"

///|
/// Bias add inplace on C-managed relu_pre buffer.
pub fn bias_add_relu_pre(
  bias : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit {
  native_bias_add_relu_pre(bias, total, last_dim)
}

///|
#borrow(out)
extern "C" fn native_relu_from_pre(out : FixedArray[Float], n : Int) -> Unit = "tensor_relu_from_pre"

///|
/// ReLU forward: reads from C-managed relu_pre, writes to output buffer.
pub fn relu_from_pre(out : FixedArray[Float], n : Int) -> Unit {
  native_relu_from_pre(out, n)
}

///|
#borrow(dy, dx)
extern "C" fn native_relu_backward_from_pre(
  dy : FixedArray[Float],
  dx : FixedArray[Float],
  n : Int,
) -> Unit = "tensor_relu_backward_from_pre"

///|
/// ReLU backward: reads dy from MoonBit, reads x from C-managed relu_pre, writes dx.
pub fn relu_backward_from_pre(
  dy : FixedArray[Float],
  dx : FixedArray[Float],
  n : Int,
) -> Unit {
  native_relu_backward_from_pre(dy, dx, n)
}

// ========== C-managed gradient pass-through buffer ==========

///|
#borrow(a, b)
extern "C" fn native_sgemm_to_grad_buf(
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
  total : Int,
) -> Unit = "tensor_sgemm_to_grad_buf"

///|
/// sgemm that writes output to C-managed gradient buffer.
pub fn sgemm_to_grad_buf(
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
  total : Int,
) -> Unit {
  native_sgemm_to_grad_buf(
    trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, total,
  )
}

///|
#borrow(dst)
extern "C" fn native_grad_buf_to_fixed(
  dst : FixedArray[Float],
  n : Int,
) -> Unit = "tensor_grad_buf_to_fixed"

///|
/// Copy from C-managed gradient buffer to FixedArray.
pub fn grad_buf_to_fixed(dst : FixedArray[Float], n : Int) -> Unit {
  native_grad_buf_to_fixed(dst, n)
}

///|
#borrow(dx)
extern "C" fn native_relu_backward_managed(
  dx : FixedArray[Float],
  n : Int,
) -> Unit = "tensor_relu_backward_managed"

///|
/// ReLU backward: reads dy from C-managed grad_buf, reads x from C-managed relu_pre,
/// writes dx to MoonBit buffer. Both input buffers are outside GC.
pub fn relu_backward_managed(dx : FixedArray[Float], n : Int) -> Unit {
  native_relu_backward_managed(dx, n)
}

// ========== C-managed relu_dx buffer ==========

///|
extern "C" fn native_relu_backward_fully_managed(n : Int) -> Unit = "tensor_relu_backward_fully_managed"

///|
/// ReLU backward fully managed: all three buffers (dy, x, dx) are C-managed.
/// No MoonBit allocations needed in the backward closure.
pub fn relu_backward_fully_managed(n : Int) -> Unit {
  native_relu_backward_fully_managed(n)
}

///|
#borrow(b, c)
extern "C" fn native_sgemm_relu_dx_a(
  trans_a : Int,
  trans_b : Int,
  m : Int,
  n : Int,
  k : Int,
  alpha : Float,
  lda : Int,
  b : FixedArray[Float],
  ldb : Int,
  beta : Float,
  c : FixedArray[Float],
  ldc : Int,
) -> Unit = "tensor_sgemm_relu_dx_a"

///|
/// sgemm where A is C-managed g_relu_dx buffer.
pub fn sgemm_relu_dx_a(
  trans_a : Int,
  trans_b : Int,
  m : Int,
  n : Int,
  k : Int,
  alpha : Float,
  lda : Int,
  b : FixedArray[Float],
  ldb : Int,
  beta : Float,
  c : FixedArray[Float],
  ldc : Int,
) -> Unit {
  native_sgemm_relu_dx_a(
    trans_a, trans_b, m, n, k, alpha, lda, b, ldb, beta, c, ldc,
  )
}

///|
#borrow(db)
extern "C" fn native_bias_add_bwd_relu_dx(
  db : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit = "tensor_bias_add_bwd_relu_dx"

///|
/// bias_add_backward reading dy from C-managed g_relu_dx.
pub fn bias_add_bwd_relu_dx(
  db : FixedArray[Float],
  total : Int,
  last_dim : Int,
) -> Unit {
  native_bias_add_bwd_relu_dx(db, total, last_dim)
}
