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
