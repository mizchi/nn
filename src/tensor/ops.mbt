///|
/// Element-wise addition: a + b
pub fn tensor_add(a : Tensor, b : Tensor) -> Result[Tensor, String] {
  if !a.shape.eq(b.shape) {
    return Err(
      "tensor_add shape mismatch: " +
      a.shape.to_string() +
      " vs " +
      b.shape.to_string(),
    )
  }
  let result = FixedArray::make(a.numel(), Float::from_int(0))
  let ac = a.contiguous()
  let bc = b.contiguous()
  for i = 0; i < result.length(); i = i + 1 {
    result[i] = ac.data[i] + bc.data[i]
  }
  Ok(tensor_new_fixed(a.shape, result))
}

///|
/// Element-wise subtraction: a - b
pub fn tensor_sub(a : Tensor, b : Tensor) -> Result[Tensor, String] {
  if !a.shape.eq(b.shape) {
    return Err(
      "tensor_sub shape mismatch: " +
      a.shape.to_string() +
      " vs " +
      b.shape.to_string(),
    )
  }
  let result = FixedArray::make(a.numel(), Float::from_int(0))
  let ac = a.contiguous()
  let bc = b.contiguous()
  for i = 0; i < result.length(); i = i + 1 {
    result[i] = ac.data[i] - bc.data[i]
  }
  Ok(tensor_new_fixed(a.shape, result))
}

///|
/// Element-wise multiplication: a * b
pub fn tensor_mul(a : Tensor, b : Tensor) -> Result[Tensor, String] {
  if !a.shape.eq(b.shape) {
    return Err(
      "tensor_mul shape mismatch: " +
      a.shape.to_string() +
      " vs " +
      b.shape.to_string(),
    )
  }
  let result = FixedArray::make(a.numel(), Float::from_int(0))
  let ac = a.contiguous()
  let bc = b.contiguous()
  for i = 0; i < result.length(); i = i + 1 {
    result[i] = ac.data[i] * bc.data[i]
  }
  Ok(tensor_new_fixed(a.shape, result))
}

///|
/// Scalar multiplication
pub fn tensor_scale(t : Tensor, s : Float) -> Tensor {
  let result = FixedArray::make(t.numel(), Float::from_int(0))
  let tc = t.contiguous()
  for i = 0; i < result.length(); i = i + 1 {
    result[i] = tc.data[i] * s
  }
  Tensor::{
    data: result,
    shape: t.shape,
    offset: 0,
    strides: t.shape.strides(),
  }
}

///|
/// Matrix multiplication for 2D tensors: [m, k] @ [k, n] -> [m, n]
pub fn tensor_matmul_2d(a : Tensor, b : Tensor) -> Result[Tensor, String] {
  if a.ndim() != 2 || b.ndim() != 2 {
    return Err("matmul_2d requires 2D tensors")
  }
  let m = a.dim(0)
  let k1 = a.dim(1)
  let k2 = b.dim(0)
  let n = b.dim(1)
  if k1 != k2 {
    return Err(
      "matmul inner dimensions mismatch: " +
      k1.to_string() +
      " vs " +
      k2.to_string(),
    )
  }
  let out_shape = match shape_new([m, n]) {
    Ok(s) => s
    Err(e) => return Err(e)
  }
  let ac = a.contiguous()
  let bc = b.contiguous()
  let result = FixedArray::make(m * n, Float::from_int(0))
  native_sgemm(
    0, 0, m, n, k1,
    Float::from_int(1),
    ac.data, k1,
    bc.data, n,
    Float::from_int(0),
    result, n,
  )
  Ok(tensor_new_fixed(out_shape, result))
}

///|
/// Batched matrix multiplication: [..., m, k] @ [..., k, n] -> [..., m, n]
pub fn tensor_matmul(a : Tensor, b : Tensor) -> Result[Tensor, String] {
  if a.ndim() == 2 && b.ndim() == 2 {
    return tensor_matmul_2d(a, b)
  }
  // For now, support 3D batched matmul: [batch, m, k] @ [batch, k, n]
  if a.ndim() == 3 && b.ndim() == 3 {
    let batch = a.dim(0)
    if b.dim(0) != batch {
      return Err("batch dimensions must match")
    }
    let m = a.dim(1)
    let k1 = a.dim(2)
    let k2 = b.dim(1)
    let n = b.dim(2)
    if k1 != k2 {
      return Err("matmul inner dimensions mismatch")
    }
    let out_shape = match shape_new([batch, m, n]) {
      Ok(s) => s
      Err(e) => return Err(e)
    }
    let ac = a.contiguous()
    let bc = b.contiguous()
    let result = FixedArray::make(batch * m * n, Float::from_int(0))
    for bi = 0; bi < batch; bi = bi + 1 {
      native_sgemm_offset(
        0, 0, m, n, k1,
        Float::from_int(1),
        ac.data, bi * m * k1, k1,
        bc.data, bi * k1 * n, n,
        Float::from_int(0),
        result, bi * m * n, n,
      )
    }
    return Ok(tensor_new_fixed(out_shape, result))
  }
  // 3D @ 2D broadcast: [batch, m, k] @ [k, n] -> [batch, m, n]
  if a.ndim() == 3 && b.ndim() == 2 {
    let batch = a.dim(0)
    let m = a.dim(1)
    let k1 = a.dim(2)
    let k2 = b.dim(0)
    let n = b.dim(1)
    if k1 != k2 {
      return Err("matmul inner dimensions mismatch")
    }
    let out_shape = match shape_new([batch, m, n]) {
      Ok(s) => s
      Err(e) => return Err(e)
    }
    let ac = a.contiguous()
    let bc = b.contiguous()
    let result = FixedArray::make(batch * m * n, Float::from_int(0))
    for bi = 0; bi < batch; bi = bi + 1 {
      native_sgemm_offset(
        0, 0, m, n, k1,
        Float::from_int(1),
        ac.data, bi * m * k1, k1,
        bc.data, 0, n,
        Float::from_int(0),
        result, bi * m * n, n,
      )
    }
    return Ok(tensor_new_fixed(out_shape, result))
  }
  Err(
    "unsupported matmul shapes: " +
    a.shape.to_string() +
    " @ " +
    b.shape.to_string(),
  )
}

///|
/// Softmax along last dimension
pub fn tensor_softmax(t : Tensor) -> Result[Tensor, String] {
  if t.ndim() == 0 {
    return Err("softmax requires at least 1D tensor")
  }
  let tc = t.contiguous()
  let result = FixedArray::make(tc.numel(), Float::from_int(0))
  let last_dim = t.dim(-1)
  let outer = t.numel() / last_dim
  for i = 0; i < outer; i = i + 1 {
    let base = i * last_dim
    // Find max for numerical stability
    let mut max_val = tc.data[base]
    for j = 1; j < last_dim; j = j + 1 {
      if tc.data[base + j] > max_val {
        max_val = tc.data[base + j]
      }
    }
    // Compute exp and sum
    let mut sum = Float::from_int(0)
    for j = 0; j < last_dim; j = j + 1 {
      let e = @math.exp((tc.data[base + j] - max_val).to_double())
        |> Float::from_double
      result[base + j] = e
      sum = sum + e
    }
    // Normalize
    for j = 0; j < last_dim; j = j + 1 {
      result[base + j] = result[base + j] / sum
    }
  }
  Ok(tensor_new_fixed(t.shape, result))
}

///|
/// ReLU activation
pub fn tensor_relu(t : Tensor) -> Tensor {
  let tc = t.contiguous()
  let result = FixedArray::make(tc.numel(), Float::from_int(0))
  let zero = Float::from_int(0)
  for i = 0; i < result.length(); i = i + 1 {
    result[i] = if tc.data[i] > zero { tc.data[i] } else { zero }
  }
  Tensor::{
    data: result,
    shape: t.shape,
    offset: 0,
    strides: t.shape.strides(),
  }
}

///|
/// GELU activation (approximate)
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn tensor_gelu(t : Tensor) -> Tensor {
  let tc = t.contiguous()
  let result = FixedArray::make(tc.numel(), Float::from_int(0))
  let sqrt_2_pi = 0.7978845608028654 // sqrt(2/pi)
  let coef = 0.044715
  for i = 0; i < result.length(); i = i + 1 {
    let x = tc.data[i].to_double()
    let inner = sqrt_2_pi * (x + coef * x * x * x)
    let tanh_val = @math.tanh(inner)
    result[i] = (0.5 * x * (1.0 + tanh_val)) |> Float::from_double
  }
  Tensor::{
    data: result,
    shape: t.shape,
    offset: 0,
    strides: t.shape.strides(),
  }
}

///|
/// Layer normalization over last dimension
pub fn tensor_layer_norm(
  t : Tensor,
  gamma : Tensor,
  beta : Tensor,
  eps : Float,
) -> Result[Tensor, String] {
  let last_dim = t.dim(-1)
  if gamma.numel() != last_dim || beta.numel() != last_dim {
    return Err("gamma/beta must match last dimension")
  }
  let tc = t.contiguous()
  let gc = gamma.contiguous()
  let bc = beta.contiguous()
  let result = FixedArray::make(tc.numel(), Float::from_int(0))
  let outer = t.numel() / last_dim
  for i = 0; i < outer; i = i + 1 {
    let base = i * last_dim
    // Compute mean
    let mut mean = Float::from_int(0)
    for j = 0; j < last_dim; j = j + 1 {
      mean = mean + tc.data[base + j]
    }
    mean = mean / Float::from_int(last_dim)
    // Compute variance
    let mut variance = Float::from_int(0)
    for j = 0; j < last_dim; j = j + 1 {
      let diff = tc.data[base + j] - mean
      variance = variance + diff * diff
    }
    variance = variance / Float::from_int(last_dim)
    // Normalize
    let std = (variance + eps).to_double().sqrt() |> Float::from_double
    for j = 0; j < last_dim; j = j + 1 {
      let normalized = (tc.data[base + j] - mean) / std
      result[base + j] = gc.data[j] * normalized + bc.data[j]
    }
  }
  Ok(tensor_new_fixed(t.shape, result))
}

///|
/// Sum along a dimension
pub fn tensor_sum(t : Tensor, dim : Int) -> Result[Tensor, String] {
  let n = t.ndim()
  let d = if dim < 0 { n + dim } else { dim }
  if d < 0 || d >= n {
    return Err("invalid dimension for sum")
  }
  let new_dims = Array::makei(n - 1, fn(i) {
    if i < d {
      t.shape.dims[i]
    } else {
      t.shape.dims[i + 1]
    }
  })
  let new_shape = if new_dims.length() == 0 {
    Shape::{ dims: [1] }
  } else {
    Shape::{ dims: new_dims }
  }
  let tc = t.contiguous()
  let result = FixedArray::make(new_shape.numel(), Float::from_int(0))
  // Generic sum implementation
  sum_along_dim(tc, result, d)
  Ok(tensor_new_fixed(new_shape, result))
}

///|
fn sum_along_dim(src : Tensor, dst : FixedArray[Float], dim : Int) -> Unit {
  let n = src.ndim()
  if n == 1 {
    let mut sum = Float::from_int(0)
    for i = 0; i < src.dim(0); i = i + 1 {
      sum = sum + src.data[i]
    }
    dst[0] = sum
  } else if n == 2 {
    if dim == 0 {
      // Sum over rows -> [cols]
      for j = 0; j < src.dim(1); j = j + 1 {
        let mut sum = Float::from_int(0)
        for i = 0; i < src.dim(0); i = i + 1 {
          sum = sum + src.at2(i, j)
        }
        dst[j] = sum
      }
    } else {
      // Sum over cols -> [rows]
      for i = 0; i < src.dim(0); i = i + 1 {
        let mut sum = Float::from_int(0)
        for j = 0; j < src.dim(1); j = j + 1 {
          sum = sum + src.at2(i, j)
        }
        dst[i] = sum
      }
    }
  } else if n == 3 {
    let d0 = src.dim(0)
    let d1 = src.dim(1)
    let d2 = src.dim(2)
    if dim == 0 {
      for i = 0; i < d1; i = i + 1 {
        for j = 0; j < d2; j = j + 1 {
          let mut sum = Float::from_int(0)
          for b = 0; b < d0; b = b + 1 {
            sum = sum + src.at3(b, i, j)
          }
          dst[i * d2 + j] = sum
        }
      }
    } else if dim == 1 {
      for b = 0; b < d0; b = b + 1 {
        for j = 0; j < d2; j = j + 1 {
          let mut sum = Float::from_int(0)
          for i = 0; i < d1; i = i + 1 {
            sum = sum + src.at3(b, i, j)
          }
          dst[b * d2 + j] = sum
        }
      }
    } else {
      for b = 0; b < d0; b = b + 1 {
        for i = 0; i < d1; i = i + 1 {
          let mut sum = Float::from_int(0)
          for j = 0; j < d2; j = j + 1 {
            sum = sum + src.at3(b, i, j)
          }
          dst[b * d1 + i] = sum
        }
      }
    }
  }
}

///|
/// Log softmax along last dimension (numerically stable)
pub fn tensor_log_softmax(t : Tensor) -> Result[Tensor, String] {
  if t.ndim() == 0 {
    return Err("log_softmax requires at least 1D tensor")
  }
  let tc = t.contiguous()
  let result = FixedArray::make(tc.numel(), Float::from_int(0))
  let last_dim = t.dim(-1)
  let outer = t.numel() / last_dim
  for i = 0; i < outer; i = i + 1 {
    let base = i * last_dim
    // Find max for numerical stability
    let mut max_val = tc.data[base]
    for j = 1; j < last_dim; j = j + 1 {
      if tc.data[base + j] > max_val {
        max_val = tc.data[base + j]
      }
    }
    // Compute log(sum(exp(x - max)))
    let mut sum = Float::from_int(0)
    for j = 0; j < last_dim; j = j + 1 {
      let e = @math.exp((tc.data[base + j] - max_val).to_double())
        |> Float::from_double
      sum = sum + e
    }
    let log_sum = @math.ln(sum.to_double()) |> Float::from_double
    // log_softmax = (x - max) - log(sum(exp(x - max)))
    for j = 0; j < last_dim; j = j + 1 {
      result[base + j] = tc.data[base + j] - max_val - log_sum
    }
  }
  Ok(tensor_new_fixed(t.shape, result))
}

///|
/// Cross-entropy loss: logits [batch, vocab_size], labels [batch]
/// Returns average negative log-likelihood
pub fn tensor_cross_entropy(
  logits : Tensor,
  labels : Array[Int],
) -> Result[Float, String] {
  if logits.ndim() != 2 {
    return Err("cross_entropy expects 2D logits [batch, vocab_size]")
  }
  let batch = logits.dim(0)
  if labels.length() != batch {
    return Err("labels length must match batch size")
  }
  let log_probs = match tensor_log_softmax(logits) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  let mut loss = Float::from_int(0)
  for i = 0; i < batch; i = i + 1 {
    loss = loss - log_probs.at2(i, labels[i])
  }
  Ok(loss / Float::from_int(batch))
}

///|
/// Argmax of a 1D tensor
pub fn tensor_argmax(t : Tensor) -> Int {
  let tc = t.contiguous()
  let mut best = 0
  let mut best_val = tc.data[0]
  for i = 1; i < tc.numel(); i = i + 1 {
    if tc.data[i] > best_val {
      best_val = tc.data[i]
      best = i
    }
  }
  best
}

///|
/// Mean along a dimension
pub fn tensor_mean(t : Tensor, dim : Int) -> Result[Tensor, String] {
  let n = t.ndim()
  let d = if dim < 0 { n + dim } else { dim }
  if d < 0 || d >= n {
    return Err("invalid dimension for mean")
  }
  match tensor_sum(t, dim) {
    Err(e) => Err(e)
    Ok(sum_t) => {
      let count = Float::from_int(t.shape.dims[d])
      Ok(tensor_scale(sum_t, Float::from_int(1) / count))
    }
  }
}
