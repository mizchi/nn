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
