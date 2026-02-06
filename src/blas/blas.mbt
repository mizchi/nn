///|
/// BLAS (Basic Linear Algebra Subprograms) bindings for MoonBit
/// Uses Apple Accelerate.framework on macOS

/// sgemm: C = alpha * op(A) @ op(B) + beta * C
/// A: M x K, B: K x N, C: M x N (row-major)
extern "C" fn native_sgemm(
  trans_a : Int,
  trans_b : Int,
  m : Int,
  n : Int,
  k : Int,
  alpha : Float,
  a : FixedArray[Byte],
  lda : Int,
  b : FixedArray[Byte],
  ldb : Int,
  beta : Float,
  c : FixedArray[Byte],
  ldc : Int,
) -> Unit = "blas_sgemm"

/// sgemv: y = alpha * op(A) @ x + beta * y
/// A: M x N, x: N, y: M (row-major)
extern "C" fn native_sgemv(
  trans : Int,
  m : Int,
  n : Int,
  alpha : Float,
  a : FixedArray[Byte],
  lda : Int,
  x : FixedArray[Byte],
  incx : Int,
  beta : Float,
  y : FixedArray[Byte],
  incy : Int,
) -> Unit = "blas_sgemv"

/// saxpy: y = alpha * x + y
extern "C" fn native_saxpy(
  n : Int,
  alpha : Float,
  x : FixedArray[Byte],
  incx : Int,
  y : FixedArray[Byte],
  incy : Int,
) -> Unit = "blas_saxpy"

/// sscal: x = alpha * x
extern "C" fn native_sscal(
  n : Int,
  alpha : Float,
  x : FixedArray[Byte],
  incx : Int,
) -> Unit = "blas_sscal"

/// sdot: x . y
extern "C" fn native_sdot(
  n : Int,
  x : FixedArray[Byte],
  incx : Int,
  y : FixedArray[Byte],
  incy : Int,
) -> Float = "blas_sdot"

/// snrm2: ||x||_2
extern "C" fn native_snrm2(
  n : Int,
  x : FixedArray[Byte],
  incx : Int,
) -> Float = "blas_snrm2"

/// scopy: y = x
extern "C" fn native_scopy(
  n : Int,
  x : FixedArray[Byte],
  incx : Int,
  y : FixedArray[Byte],
  incy : Int,
) -> Unit = "blas_scopy"

///|
fn float_array_to_fixed_bytes(arr : Array[Float]) -> FixedArray[Byte] {
  let bytes : FixedArray[Byte] = FixedArray::make(arr.length() * 4, b'\x00')
  for i, v in arr {
    let bits = v.reinterpret_as_int()
    bytes[i * 4] = (bits & 0xFF).to_byte()
    bytes[i * 4 + 1] = ((bits >> 8) & 0xFF).to_byte()
    bytes[i * 4 + 2] = ((bits >> 16) & 0xFF).to_byte()
    bytes[i * 4 + 3] = ((bits >> 24) & 0xFF).to_byte()
  }
  bytes
}

///|
fn fixed_bytes_to_float_array(bytes : FixedArray[Byte], count : Int) -> Array[Float] {
  let arr : Array[Float] = Array::make(count, Float::from_int(0))
  for i in 0..<count {
    let bits = bytes[i * 4].to_int() |
      (bytes[i * 4 + 1].to_int() << 8) |
      (bytes[i * 4 + 2].to_int() << 16) |
      (bytes[i * 4 + 3].to_int() << 24)
    arr[i] = Float::reinterpret_from_int(bits)
  }
  arr
}

///|
/// Matrix multiply: C = A @ B
/// A: m x k, B: k x n, C: m x n (row-major)
pub fn sgemm(
  a : Array[Float],
  b : Array[Float],
  c : Array[Float],
  m : Int,
  n : Int,
  k : Int,
) -> Unit {
  let a_bytes = float_array_to_fixed_bytes(a)
  let b_bytes = float_array_to_fixed_bytes(b)
  let c_bytes = float_array_to_fixed_bytes(c)
  native_sgemm(0, 0, m, n, k, 1.0, a_bytes, k, b_bytes, n, 0.0, c_bytes, n)
  // Copy back to c
  let result = fixed_bytes_to_float_array(c_bytes, m * n)
  for i, v in result {
    c[i] = v
  }
}

///|
/// Matrix-vector multiply: y = A @ x
/// A: m x n, x: n, y: m (row-major)
pub fn sgemv(
  a : Array[Float],
  x : Array[Float],
  y : Array[Float],
  m : Int,
  n : Int,
) -> Unit {
  let a_bytes = float_array_to_fixed_bytes(a)
  let x_bytes = float_array_to_fixed_bytes(x)
  let y_bytes = float_array_to_fixed_bytes(y)
  native_sgemv(0, m, n, 1.0, a_bytes, n, x_bytes, 1, 0.0, y_bytes, 1)
  let result = fixed_bytes_to_float_array(y_bytes, m)
  for i, v in result {
    y[i] = v
  }
}

///|
/// Matrix-vector multiply (transposed): y = A^T @ x
/// A: m x n, x: m, y: n (row-major)
pub fn sgemv_trans(
  a : Array[Float],
  x : Array[Float],
  y : Array[Float],
  m : Int,
  n : Int,
) -> Unit {
  let a_bytes = float_array_to_fixed_bytes(a)
  let x_bytes = float_array_to_fixed_bytes(x)
  let y_bytes = float_array_to_fixed_bytes(y)
  native_sgemv(1, m, n, 1.0, a_bytes, n, x_bytes, 1, 0.0, y_bytes, 1)
  let result = fixed_bytes_to_float_array(y_bytes, n)
  for i, v in result {
    y[i] = v
  }
}

///|
/// Vector add: y = alpha * x + y
pub fn saxpy(alpha : Float, x : Array[Float], y : Array[Float]) -> Unit {
  let n = x.length()
  let x_bytes = float_array_to_fixed_bytes(x)
  let y_bytes = float_array_to_fixed_bytes(y)
  native_saxpy(n, alpha, x_bytes, 1, y_bytes, 1)
  let result = fixed_bytes_to_float_array(y_bytes, n)
  for i, v in result {
    y[i] = v
  }
}

///|
/// Dot product: x . y
pub fn sdot(x : Array[Float], y : Array[Float]) -> Float {
  let n = x.length()
  let x_bytes = float_array_to_fixed_bytes(x)
  let y_bytes = float_array_to_fixed_bytes(y)
  native_sdot(n, x_bytes, 1, y_bytes, 1)
}

///|
/// L2 norm: ||x||_2
pub fn snrm2(x : Array[Float]) -> Float {
  let n = x.length()
  let x_bytes = float_array_to_fixed_bytes(x)
  native_snrm2(n, x_bytes, 1)
}

// ============================================================================
// Optimized: C-side float buffer operations (zero-copy after init)
// ============================================================================

///|
/// Opaque handle to C-side float buffer
pub struct FloatBuffer(Int64)

///|
extern "C" fn native_alloc_floats(count : Int) -> Int64 = "blas_alloc_floats"

///|
extern "C" fn native_free_floats(buf : Int64) -> Unit = "blas_free_floats"

///|
extern "C" fn native_copy_from_bytes(
  dst : Int64,
  src : FixedArray[Byte],
  count : Int,
) -> Unit = "blas_copy_from_bytes"

///|
extern "C" fn native_copy_to_bytes(
  dst : FixedArray[Byte],
  src : Int64,
  count : Int,
) -> Unit = "blas_copy_to_bytes"

///|
extern "C" fn native_get_float(buf : Int64, idx : Int) -> Float = "blas_get_float"

///|
extern "C" fn native_set_float(buf : Int64, idx : Int, val : Float) -> Unit = "blas_set_float"

///|
extern "C" fn native_sgemm_direct(
  trans_a : Int,
  trans_b : Int,
  m : Int,
  n : Int,
  k : Int,
  alpha : Float,
  a : Int64,
  lda : Int,
  b : Int64,
  ldb : Int,
  beta : Float,
  c : Int64,
  ldc : Int,
) -> Unit = "blas_sgemm_direct"

///|
extern "C" fn native_layer1_fused(
  input : Int64,
  weight : Int64,
  bias : Int64,
  output : Int64,
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit = "blas_layer1_fused"

///|
extern "C" fn native_layer2_fused(
  input : Int64,
  weight : Int64,
  bias : Int64,
  output : Int64,
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit = "blas_layer2_fused"

///|
/// Allocate a C-side float buffer
pub fn alloc_floats(count : Int) -> FloatBuffer {
  FloatBuffer(native_alloc_floats(count))
}

///|
/// Free a C-side float buffer
pub fn free_floats(buf : FloatBuffer) -> Unit {
  native_free_floats(buf.0)
}

///|
/// Copy from MoonBit Array[Float] to C buffer
pub fn copy_to_buffer(buf : FloatBuffer, arr : Array[Float]) -> Unit {
  let bytes = float_array_to_fixed_bytes(arr)
  native_copy_from_bytes(buf.0, bytes, arr.length())
}

///|
/// Copy from C buffer to MoonBit Array[Float]
pub fn copy_from_buffer(arr : Array[Float], buf : FloatBuffer) -> Unit {
  let bytes : FixedArray[Byte] = FixedArray::make(arr.length() * 4, b'\x00')
  native_copy_to_bytes(bytes, buf.0, arr.length())
  for i in 0..<arr.length() {
    let bits = bytes[i * 4].to_int() |
      (bytes[i * 4 + 1].to_int() << 8) |
      (bytes[i * 4 + 2].to_int() << 16) |
      (bytes[i * 4 + 3].to_int() << 24)
    arr[i] = Float::reinterpret_from_int(bits)
  }
}

///|
/// Get float at index from C buffer
pub fn get_float(buf : FloatBuffer, idx : Int) -> Float {
  native_get_float(buf.0, idx)
}

///|
/// Set float at index in C buffer
pub fn set_float(buf : FloatBuffer, idx : Int, val : Float) -> Unit {
  native_set_float(buf.0, idx, val)
}

///|
/// Matrix multiply on C buffers: C = A @ B
/// A: m x k, B: k x n, C: m x n (row-major)
pub fn sgemm_direct(
  a : FloatBuffer,
  b : FloatBuffer,
  c : FloatBuffer,
  m : Int,
  n : Int,
  k : Int,
) -> Unit {
  native_sgemm_direct(0, 0, m, n, k, 1.0, a.0, k, b.0, n, 0.0, c.0, n)
}

///|
/// Fused layer1: output = ReLU(input @ weight + bias)
/// input: batch x in_dim, weight: in_dim x out_dim, bias: out_dim, output: batch x out_dim
pub fn layer1_fused(
  input : FloatBuffer,
  weight : FloatBuffer,
  bias : FloatBuffer,
  output : FloatBuffer,
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  native_layer1_fused(input.0, weight.0, bias.0, output.0, batch, in_dim, out_dim)
}

///|
/// Fused layer2: output = input @ weight + bias (no activation)
/// input: batch x in_dim, weight: in_dim x out_dim, bias: out_dim, output: batch x out_dim
pub fn layer2_fused(
  input : FloatBuffer,
  weight : FloatBuffer,
  bias : FloatBuffer,
  output : FloatBuffer,
  batch : Int,
  in_dim : Int,
  out_dim : Int,
) -> Unit {
  native_layer2_fused(input.0, weight.0, bias.0, output.0, batch, in_dim, out_dim)
}

// ============================================================================
// MLP forward with C-side buffers (optimized: single copy at start/end)
// ============================================================================

///|
pub struct MlpBuffers {
  input : FloatBuffer
  weight1 : FloatBuffer
  bias1 : FloatBuffer
  hidden : FloatBuffer
  weight2 : FloatBuffer
  bias2 : FloatBuffer
  output : FloatBuffer
  batch : Int
  input_dim : Int
  hidden_dim : Int
  output_dim : Int
}

///|
/// Create MLP buffers in C memory for zero-copy forward
pub fn mlp_buffers_create(
  batch : Int,
  input_dim : Int,
  hidden_dim : Int,
  output_dim : Int,
) -> MlpBuffers {
  {
    input: alloc_floats(batch * input_dim),
    weight1: alloc_floats(input_dim * hidden_dim),
    bias1: alloc_floats(hidden_dim),
    hidden: alloc_floats(batch * hidden_dim),
    weight2: alloc_floats(hidden_dim * output_dim),
    bias2: alloc_floats(output_dim),
    output: alloc_floats(batch * output_dim),
    batch,
    input_dim,
    hidden_dim,
    output_dim,
  }
}

///|
/// Free MLP buffers
pub fn mlp_buffers_free(bufs : MlpBuffers) -> Unit {
  free_floats(bufs.input)
  free_floats(bufs.weight1)
  free_floats(bufs.bias1)
  free_floats(bufs.hidden)
  free_floats(bufs.weight2)
  free_floats(bufs.bias2)
  free_floats(bufs.output)
}

///|
/// Initialize MLP buffers with data (called once before benchmarking)
pub fn mlp_buffers_init(
  bufs : MlpBuffers,
  input_data : Array[Float],
  weight1_data : Array[Float],
  bias1_data : Array[Float],
  weight2_data : Array[Float],
  bias2_data : Array[Float],
) -> Unit {
  copy_to_buffer(bufs.input, input_data)
  copy_to_buffer(bufs.weight1, weight1_data)
  copy_to_buffer(bufs.bias1, bias1_data)
  copy_to_buffer(bufs.weight2, weight2_data)
  copy_to_buffer(bufs.bias2, bias2_data)
}

///|
/// Run MLP forward pass using C-side buffers (no data copy)
pub fn mlp_forward_fused(bufs : MlpBuffers) -> Unit {
  // Layer 1: hidden = ReLU(input @ weight1 + bias1)
  layer1_fused(
    bufs.input,
    bufs.weight1,
    bufs.bias1,
    bufs.hidden,
    bufs.batch,
    bufs.input_dim,
    bufs.hidden_dim,
  )
  // Layer 2: output = hidden @ weight2 + bias2
  layer2_fused(
    bufs.hidden,
    bufs.weight2,
    bufs.bias2,
    bufs.output,
    bufs.batch,
    bufs.hidden_dim,
    bufs.output_dim,
  )
}

///|
/// Copy output from C buffers back to MoonBit array
pub fn mlp_buffers_get_output(bufs : MlpBuffers, out : Array[Float]) -> Unit {
  copy_from_buffer(out, bufs.output)
}

// ============================================================================
// Batch Training with BLAS (forward + backward + update in C)
// ============================================================================

///|
extern "C" fn native_train_step(
  input : Int64,
  labels : Int64,
  weight1 : Int64,
  bias1 : Int64,
  weight2 : Int64,
  bias2 : Int64,
  hidden : Int64,
  output : Int64,
  probs : Int64,
  grad_w1 : Int64,
  grad_b1 : Int64,
  grad_w2 : Int64,
  grad_b2 : Int64,
  delta2 : Int64,
  delta1 : Int64,
  result : Int64,
  lr : Float,
  batch : Int,
  input_dim : Int,
  hidden_dim : Int,
  output_dim : Int,
) -> Unit = "blas_train_step"

///|
extern "C" fn native_zero(buf : Int64, n : Int) -> Unit = "blas_zero"

///|
/// Opaque handle to C-side int buffer (for labels)
pub struct IntBuffer(Int64)

///|
extern "C" fn native_alloc_ints(count : Int) -> Int64 = "blas_alloc_ints"

///|
extern "C" fn native_free_ints(buf : Int64) -> Unit = "blas_free_ints"

///|
extern "C" fn native_set_int(buf : Int64, idx : Int, val : Int) -> Unit = "blas_set_int"

///|
extern "C" fn native_get_int(buf : Int64, idx : Int) -> Int = "blas_get_int"

///|
/// Allocate a C-side int buffer
pub fn alloc_ints(count : Int) -> IntBuffer {
  IntBuffer(native_alloc_ints(count))
}

///|
/// Free a C-side int buffer
pub fn free_ints(buf : IntBuffer) -> Unit {
  native_free_ints(buf.0)
}

///|
/// Set int at index in C buffer
pub fn set_int(buf : IntBuffer, idx : Int, val : Int) -> Unit {
  native_set_int(buf.0, idx, val)
}

///|
/// Training buffers for batch BLAS training
pub struct MlpTrainBuffers {
  // Parameters (in C memory)
  weight1 : FloatBuffer
  bias1 : FloatBuffer
  weight2 : FloatBuffer
  bias2 : FloatBuffer
  // Workspace for forward pass
  input : FloatBuffer
  hidden : FloatBuffer
  output : FloatBuffer
  probs : FloatBuffer
  // Workspace for backward pass
  grad_w1 : FloatBuffer
  grad_b1 : FloatBuffer
  grad_w2 : FloatBuffer
  grad_b2 : FloatBuffer
  delta2 : FloatBuffer
  delta1 : FloatBuffer
  // Labels and result
  labels : IntBuffer
  result : FloatBuffer
  // Dimensions
  batch : Int
  input_dim : Int
  hidden_dim : Int
  output_dim : Int
}

///|
/// Create training buffers
pub fn mlp_train_buffers_create(
  batch : Int,
  input_dim : Int,
  hidden_dim : Int,
  output_dim : Int,
) -> MlpTrainBuffers {
  {
    weight1: alloc_floats(input_dim * hidden_dim),
    bias1: alloc_floats(hidden_dim),
    weight2: alloc_floats(hidden_dim * output_dim),
    bias2: alloc_floats(output_dim),
    input: alloc_floats(batch * input_dim),
    hidden: alloc_floats(batch * hidden_dim),
    output: alloc_floats(batch * output_dim),
    probs: alloc_floats(batch * output_dim),
    grad_w1: alloc_floats(input_dim * hidden_dim),
    grad_b1: alloc_floats(hidden_dim),
    grad_w2: alloc_floats(hidden_dim * output_dim),
    grad_b2: alloc_floats(output_dim),
    delta2: alloc_floats(batch * output_dim),
    delta1: alloc_floats(batch * hidden_dim),
    labels: alloc_ints(batch),
    result: alloc_floats(2),
    batch,
    input_dim,
    hidden_dim,
    output_dim,
  }
}

///|
/// Free training buffers
pub fn mlp_train_buffers_free(bufs : MlpTrainBuffers) -> Unit {
  free_floats(bufs.weight1)
  free_floats(bufs.bias1)
  free_floats(bufs.weight2)
  free_floats(bufs.bias2)
  free_floats(bufs.input)
  free_floats(bufs.hidden)
  free_floats(bufs.output)
  free_floats(bufs.probs)
  free_floats(bufs.grad_w1)
  free_floats(bufs.grad_b1)
  free_floats(bufs.grad_w2)
  free_floats(bufs.grad_b2)
  free_floats(bufs.delta2)
  free_floats(bufs.delta1)
  free_ints(bufs.labels)
  free_floats(bufs.result)
}

///|
/// Initialize weights in training buffers
pub fn mlp_train_buffers_init_weights(
  bufs : MlpTrainBuffers,
  weight1 : Array[Float],
  bias1 : Array[Float],
  weight2 : Array[Float],
  bias2 : Array[Float],
) -> Unit {
  copy_to_buffer(bufs.weight1, weight1)
  copy_to_buffer(bufs.bias1, bias1)
  copy_to_buffer(bufs.weight2, weight2)
  copy_to_buffer(bufs.bias2, bias2)
}

///|
/// Copy weights from training buffers back to arrays
pub fn mlp_train_buffers_get_weights(
  bufs : MlpTrainBuffers,
  weight1 : Array[Float],
  bias1 : Array[Float],
  weight2 : Array[Float],
  bias2 : Array[Float],
) -> Unit {
  copy_from_buffer(weight1, bufs.weight1)
  copy_from_buffer(bias1, bufs.bias1)
  copy_from_buffer(weight2, bufs.weight2)
  copy_from_buffer(bias2, bufs.bias2)
}

///|
/// Zero gradient buffers before each batch
fn zero_grads(bufs : MlpTrainBuffers) -> Unit {
  native_zero(bufs.grad_w1.0, bufs.input_dim * bufs.hidden_dim)
  native_zero(bufs.grad_b1.0, bufs.hidden_dim)
  native_zero(bufs.grad_w2.0, bufs.hidden_dim * bufs.output_dim)
  native_zero(bufs.grad_b2.0, bufs.output_dim)
}

///|
/// Run a complete training step: forward + backward + update
/// Returns (loss_sum, correct_count)
pub fn mlp_train_step(
  bufs : MlpTrainBuffers,
  input : Array[Float],
  labels : Array[Int],
  lr : Float,
) -> (Float, Int) {
  // Copy input data
  copy_to_buffer(bufs.input, input)
  // Copy labels
  for i = 0; i < labels.length(); i = i + 1 {
    set_int(bufs.labels, i, labels[i])
  }
  // Zero gradients
  zero_grads(bufs)
  // Run training step
  native_train_step(
    bufs.input.0,
    bufs.labels.0,
    bufs.weight1.0,
    bufs.bias1.0,
    bufs.weight2.0,
    bufs.bias2.0,
    bufs.hidden.0,
    bufs.output.0,
    bufs.probs.0,
    bufs.grad_w1.0,
    bufs.grad_b1.0,
    bufs.grad_w2.0,
    bufs.grad_b2.0,
    bufs.delta2.0,
    bufs.delta1.0,
    bufs.result.0,
    lr,
    bufs.batch,
    bufs.input_dim,
    bufs.hidden_dim,
    bufs.output_dim,
  )
  // Get results
  let loss_sum = get_float(bufs.result, 0)
  let correct = get_float(bufs.result, 1).to_int()
  (loss_sum, correct)
}
