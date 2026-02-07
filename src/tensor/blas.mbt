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
#borrow(dy, x, w, d_w, dx)
extern "C" fn native_batched_linear_backward(
  dy : FixedArray[Float],
  x : FixedArray[Float],
  w : FixedArray[Float],
  d_w : FixedArray[Float],
  dx : FixedArray[Float],
  n : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit = "tensor_batched_linear_backward"

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
#borrow(data)
extern "C" fn native_softmax_inplace_offset(
  data : FixedArray[Float],
  offset : Int,
  rows : Int,
  cols : Int,
) -> Unit = "tensor_softmax_inplace_offset"

///|
#borrow(d_attn, attn, out)
extern "C" fn native_softmax_backward_rows(
  d_attn : FixedArray[Float],
  attn : FixedArray[Float],
  out : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> Unit = "tensor_softmax_backward_rows"

///|
#borrow(d_out, q, k, v, attn_w, dq, dk, dv)
extern "C" fn native_attention_head_backward(
  d_out : FixedArray[Float],
  d_out_off : Int,
  q : FixedArray[Float],
  q_off : Int,
  k : FixedArray[Float],
  k_off : Int,
  v : FixedArray[Float],
  v_off : Int,
  attn_w : FixedArray[Float],
  dq : FixedArray[Float],
  dq_off : Int,
  dk : FixedArray[Float],
  dk_off : Int,
  dv : FixedArray[Float],
  dv_off : Int,
  seq : Int,
  d_k : Int,
  scale : Float,
) -> Unit = "tensor_attention_head_backward"

///|
#borrow(d_out, q, k, v, attn_w, dq, dk, dv)
extern "C" fn native_attention_backward_batch(
  d_out : FixedArray[Float],
  q : FixedArray[Float],
  k : FixedArray[Float],
  v : FixedArray[Float],
  attn_w : FixedArray[Float],
  dq : FixedArray[Float],
  dk : FixedArray[Float],
  dv : FixedArray[Float],
  total_heads : Int,
  seq : Int,
  d_k : Int,
  scale : Float,
) -> Unit = "tensor_attention_backward_batch"

///|
#borrow(d_out, q, k, v, attn_w, dq, dk, dv)
extern "C" fn native_attention_backward_batch_interleaved(
  d_out : FixedArray[Float],
  q : FixedArray[Float],
  k : FixedArray[Float],
  v : FixedArray[Float],
  attn_w : FixedArray[Float],
  dq : FixedArray[Float],
  dk : FixedArray[Float],
  dv : FixedArray[Float],
  batch : Int,
  num_heads : Int,
  seq : Int,
  d_k : Int,
  scale : Float,
) -> Unit = "tensor_attention_backward_batch_interleaved"

///|
#borrow(dst, add)
extern "C" fn native_add_matrix_inplace(
  dst : FixedArray[Float],
  add : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> Unit = "tensor_add_matrix_inplace"

///|
#borrow(dst, add)
extern "C" fn native_add_matrix_inplace_offset(
  dst : FixedArray[Float],
  dst_off : Int,
  add : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> Unit = "tensor_add_matrix_inplace_offset"

///|
#borrow(logits, labels, d_logits)
extern "C" fn native_cross_entropy_fwd_bwd(
  logits : FixedArray[Float],
  labels : FixedArray[Int],
  d_logits : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> Float = "tensor_cross_entropy_fwd_bwd"

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
#borrow(dy, add_inout, x, mean, rstd, gamma, d_gamma, d_beta)
extern "C" fn native_layer_norm_bwd_add_inplace(
  dy : FixedArray[Float],
  add_inout : FixedArray[Float],
  x : FixedArray[Float],
  mean : FixedArray[Float],
  rstd : FixedArray[Float],
  gamma : FixedArray[Float],
  d_gamma : FixedArray[Float],
  d_beta : FixedArray[Float],
  outer : Int,
  last_dim : Int,
) -> Unit = "tensor_layer_norm_bwd_add_inplace"

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
/// Fused batched linear backward:
/// dx = dy @ w^T, d_w += x^T @ dy.
pub fn batched_linear_backward_raw(
  dy : FixedArray[Float],
  x : FixedArray[Float],
  w : FixedArray[Float],
  d_w : FixedArray[Float],
  dx : FixedArray[Float],
  n : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  native_batched_linear_backward(dy, x, w, d_w, dx, n, in_dim, out_dim)
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
#borrow(dst)
extern "C" fn native_zero(dst : FixedArray[Float], n : Int) -> Unit = "tensor_zero"

///|
#borrow(param, grad, m, v)
extern "C" fn native_adamw_step(
  param : FixedArray[Float],
  grad : FixedArray[Float],
  m : FixedArray[Float],
  v : FixedArray[Float],
  n : Int,
  lr : Float,
  beta1 : Float,
  beta2 : Float,
  eps : Float,
  weight_decay : Float,
  bias_c1 : Float,
  bias_c2 : Float,
) -> Unit = "tensor_adamw_step"

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
pub fn softmax_inplace_offset(
  data : FixedArray[Float],
  offset : Int,
  rows : Int,
  cols : Int,
) -> Unit {
  native_softmax_inplace_offset(data, offset, rows, cols)
}

///|
/// Row-wise softmax backward:
/// out[row, col] = attn[row, col] * (d_attn[row, col] - sum_j d_attn[row, j] * attn[row, j])
pub fn softmax_backward_rows(
  d_attn : FixedArray[Float],
  attn : FixedArray[Float],
  out : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> Unit {
  native_softmax_backward_rows(d_attn, attn, out, rows, cols)
}

///|
/// Fused backward for one attention head.
/// Inputs: d_out/q/k/v [seq, d_k], attn_w [seq, seq].
/// Outputs: dq/dk/dv [seq, d_k].
pub fn attention_head_backward(
  d_out : FixedArray[Float],
  d_out_off : Int,
  q : FixedArray[Float],
  q_off : Int,
  k : FixedArray[Float],
  k_off : Int,
  v : FixedArray[Float],
  v_off : Int,
  attn_w : FixedArray[Float],
  dq : FixedArray[Float],
  dq_off : Int,
  dk : FixedArray[Float],
  dk_off : Int,
  dv : FixedArray[Float],
  dv_off : Int,
  seq : Int,
  d_k : Int,
  scale : Float,
) -> Unit {
  native_attention_head_backward(
    d_out, d_out_off, q, q_off, k, k_off, v, v_off, attn_w, dq, dq_off, dk, dk_off,
    dv, dv_off, seq, d_k, scale,
  )
}

///|
/// Fused backward for all attention heads.
pub fn attention_backward_batch(
  d_out : FixedArray[Float],
  q : FixedArray[Float],
  k : FixedArray[Float],
  v : FixedArray[Float],
  attn_w : FixedArray[Float],
  dq : FixedArray[Float],
  dk : FixedArray[Float],
  dv : FixedArray[Float],
  total_heads : Int,
  seq : Int,
  d_k : Int,
  scale : Float,
) -> Unit {
  native_attention_backward_batch(
    d_out, q, k, v, attn_w, dq, dk, dv, total_heads, seq, d_k, scale,
  )
}

///|
/// Fused backward for interleaved layout:
/// d_out/q/k/v, dq/dk/dv are [batch, seq, d_model], with d_model = num_heads * d_k.
pub fn attention_backward_batch_interleaved(
  d_out : FixedArray[Float],
  q : FixedArray[Float],
  k : FixedArray[Float],
  v : FixedArray[Float],
  attn_w : FixedArray[Float],
  dq : FixedArray[Float],
  dk : FixedArray[Float],
  dv : FixedArray[Float],
  batch : Int,
  num_heads : Int,
  seq : Int,
  d_k : Int,
  scale : Float,
) -> Unit {
  native_attention_backward_batch_interleaved(
    d_out, q, k, v, attn_w, dq, dk, dv, batch, num_heads, seq, d_k, scale,
  )
}

///|
pub fn add_matrix_inplace(
  dst : FixedArray[Float],
  add : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> Unit {
  native_add_matrix_inplace(dst, add, rows, cols)
}

///|
pub fn add_matrix_inplace_offset(
  dst : FixedArray[Float],
  dst_off : Int,
  add : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> Unit {
  native_add_matrix_inplace_offset(dst, dst_off, add, rows, cols)
}

///|
/// Cross-entropy forward + backward over rows.
/// Returns average loss and writes d_logits = (softmax - one_hot(labels)) / rows.
pub fn cross_entropy_fwd_bwd(
  logits : FixedArray[Float],
  labels : FixedArray[Int],
  d_logits : FixedArray[Float],
  rows : Int,
  cols : Int,
) -> Float {
  native_cross_entropy_fwd_bwd(logits, labels, d_logits, rows, cols)
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
pub fn zero_inplace(dst : FixedArray[Float], n : Int) -> Unit {
  native_zero(dst, n)
}

///|
/// AdamW step in-place for one tensor.
pub fn adamw_step_inplace(
  param : FixedArray[Float],
  grad : FixedArray[Float],
  m : FixedArray[Float],
  v : FixedArray[Float],
  n : Int,
  lr : Float,
  beta1 : Float,
  beta2 : Float,
  eps : Float,
  weight_decay : Float,
  bias_c1 : Float,
  bias_c2 : Float,
) -> Unit {
  native_adamw_step(
    param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, bias_c1, bias_c2,
  )
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

// ========== Fused FFN backward ==========

///|
#borrow(dy, ff_post_gelu, ff_pre_gelu, ln2_out, w2, w1, d_w2, d_b2, d_w1, d_b1, dx)
extern "C" fn native_fused_ffn_backward(
  dy : FixedArray[Float],
  ff_post_gelu : FixedArray[Float],
  ff_pre_gelu : FixedArray[Float],
  ln2_out : FixedArray[Float],
  w2 : FixedArray[Float],
  w1 : FixedArray[Float],
  d_w2 : FixedArray[Float],
  d_b2 : FixedArray[Float],
  d_w1 : FixedArray[Float],
  d_b1 : FixedArray[Float],
  dx : FixedArray[Float],
  n : Int,
  d_model : Int,
  d_ff : Int,
) -> Unit = "tensor_fused_ffn_backward"

///|
/// Fused FFN backward for Transformer block:
/// dy -> (d_w2, d_b2) -> GELU backward -> (d_w1, d_b1, dx).
/// Accumulates gradients into d_w1/d_b1/d_w2/d_b2, writes dx.
pub fn fused_ffn_backward(
  dy : FixedArray[Float],
  ff_post_gelu : FixedArray[Float],
  ff_pre_gelu : FixedArray[Float],
  ln2_out : FixedArray[Float],
  w2 : FixedArray[Float],
  w1 : FixedArray[Float],
  d_w2 : FixedArray[Float],
  d_b2 : FixedArray[Float],
  d_w1 : FixedArray[Float],
  d_b1 : FixedArray[Float],
  dx : FixedArray[Float],
  n : Int,
  d_model : Int,
  d_ff : Int,
) -> Unit {
  native_fused_ffn_backward(
    dy, ff_post_gelu, ff_pre_gelu, ln2_out, w2, w1, d_w2, d_b2, d_w1, d_b1, dx, n,
    d_model, d_ff,
  )
}

// ========== Fused Two-Layer MLP backward ==========

///|
#borrow(out)
extern "C" fn native_relu_from_pre_cached(
  out : FixedArray[Float],
  n : Int,
) -> Unit = "tensor_relu_from_pre_cached"

///|
/// ReLU forward with caching: reads from C-managed relu_pre, writes to both out AND
/// C-managed relu_out. The cached relu_out is used in backward to avoid reading GC old-gen.
pub fn relu_from_pre_cached(out : FixedArray[Float], n : Int) -> Unit {
  native_relu_from_pre_cached(out, n)
}

///|
#borrow(dy, x_data, w2_data, dw1, db1, dw2, db2)
extern "C" fn native_fused_two_layer_relu_backward(
  dy : FixedArray[Float],
  x_data : FixedArray[Float],
  w2_data : FixedArray[Float],
  dw1 : FixedArray[Float],
  db1 : FixedArray[Float],
  dw2 : FixedArray[Float],
  db2 : FixedArray[Float],
  batch : Int,
  input_dim : Int,
  hidden_dim : Int,
  output_dim : Int,
) -> Unit = "tensor_fused_two_layer_relu_backward"

///|
/// Fused 2-layer MLP backward: linear2 backward + relu backward + linear1 backward
/// all in a single C call. Reads relu output from C-managed g_relu_out (not MoonBit old-gen).
pub fn fused_two_layer_relu_backward(
  dy : FixedArray[Float],
  x_data : FixedArray[Float],
  w2_data : FixedArray[Float],
  dw1 : FixedArray[Float],
  db1 : FixedArray[Float],
  dw2 : FixedArray[Float],
  db2 : FixedArray[Float],
  batch : Int,
  input_dim : Int,
  hidden_dim : Int,
  output_dim : Int,
) -> Unit {
  native_fused_two_layer_relu_backward(
    dy, x_data, w2_data, dw1, db1, dw2, db2, batch, input_dim, hidden_dim, output_dim,
  )
}

///|
#borrow(out5)
extern "C" fn native_fused_two_layer_last_profile(
  out5 : FixedArray[UInt64],
) -> Unit = "tensor_fused_two_layer_last_profile"

///|
/// Get last fused_two_layer backward profile in nanoseconds:
/// [dw2, db2, da1, relu, dw1_db1]
pub fn fused_two_layer_last_profile(out5 : FixedArray[UInt64]) -> Unit {
  native_fused_two_layer_last_profile(out5)
}

// ========== Fused Linear+ReLU backward ==========

///|
#borrow(x_data, w_data, dw_data, db_data, dx_data)
extern "C" fn native_fused_linear_relu_backward(
  x_data : FixedArray[Float],
  w_data : FixedArray[Float],
  dw_data : FixedArray[Float],
  db_data : FixedArray[Float],
  dx_data : FixedArray[Float],
  n_rows : Int,
  in_features : Int,
  out_features : Int,
  compute_dx : Int,
) -> Unit = "tensor_fused_linear_relu_backward"

///|
/// Fused Linear+ReLU backward: performs relu_backward + sgemm dW + bias_backward + sgemm dx
/// all in a single C call. Reads from C-managed g_grad_buf and g_relu_pre.
/// Avoids GC interaction between heavy compute operations.
pub fn fused_linear_relu_backward(
  x_data : FixedArray[Float],
  w_data : FixedArray[Float],
  dw_data : FixedArray[Float],
  db_data : FixedArray[Float],
  dx_data : FixedArray[Float],
  n_rows : Int,
  in_features : Int,
  out_features : Int,
  compute_dx : Int,
) -> Unit {
  native_fused_linear_relu_backward(
    x_data, w_data, dw_data, db_data, dx_data, n_rows, in_features, out_features,
    compute_dx,
  )
}
