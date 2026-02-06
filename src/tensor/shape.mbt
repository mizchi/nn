///|
/// Shape represents tensor dimensions.
/// Examples: [batch, seq, hidden], [rows, cols], [n]
pub struct Shape {
  dims : Array[Int]
}

///|
pub fn shape_new(dims : Array[Int]) -> Result[Shape, String] {
  for i = 0; i < dims.length(); i = i + 1 {
    if dims[i] <= 0 {
      return Err(
        "shape dimensions must be positive, got " + dims[i].to_string(),
      )
    }
  }
  Ok(Shape::{ dims: dims.copy() })
}

///|
/// Create shape without validation (for internal use when dims are known valid)
pub fn shape_new_unchecked(dims : Array[Int]) -> Shape {
  Shape::{ dims: dims.copy() }
}

///|
pub fn shape_from(dims : FixedArray[Int]) -> Result[Shape, String] {
  let arr = Array::makei(dims.length(), fn(i) { dims[i] })
  shape_new(arr)
}

///|
pub fn Shape::ndim(self : Shape) -> Int {
  self.dims.length()
}

///|
pub fn Shape::dim(self : Shape, i : Int) -> Int {
  let idx = if i < 0 { self.dims.length() + i } else { i }
  self.dims[idx]
}

///|
pub fn Shape::numel(self : Shape) -> Int {
  let mut n = 1
  for i = 0; i < self.dims.length(); i = i + 1 {
    n = n * self.dims[i]
  }
  n
}

///|
pub fn Shape::bytes(self : Shape) -> Int {
  self.numel() * 4
}

///|
/// Compute strides for row-major (C-contiguous) layout.
/// For shape [2, 3, 4], strides = [12, 4, 1]
pub fn Shape::strides(self : Shape) -> Array[Int] {
  let n = self.dims.length()
  if n == 0 {
    return []
  }
  let strides = Array::make(n, 1)
  for i = n - 2; i >= 0; i = i - 1 {
    strides[i] = strides[i + 1] * self.dims[i + 1]
  }
  strides
}

///|
pub fn Shape::to_string(self : Shape) -> String {
  let mut s = "Shape["
  for i = 0; i < self.dims.length(); i = i + 1 {
    if i > 0 {
      s = s + ", "
    }
    s = s + self.dims[i].to_string()
  }
  s + "]"
}

///|
pub impl Show for Shape with output(self, logger) {
  logger.write_string(self.to_string())
}

///|
pub fn Shape::eq(self : Shape, other : Shape) -> Bool {
  if self.dims.length() != other.dims.length() {
    return false
  }
  for i = 0; i < self.dims.length(); i = i + 1 {
    if self.dims[i] != other.dims[i] {
      return false
    }
  }
  true
}

///|
pub fn shape_broadcast(a : Shape, b : Shape) -> Result[Shape, String] {
  let na = a.dims.length()
  let nb = b.dims.length()
  let n = if na > nb { na } else { nb }
  let result = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    let da = if i < na { a.dims[na - 1 - i] } else { 1 }
    let db = if i < nb { b.dims[nb - 1 - i] } else { 1 }
    if da == db {
      result[n - 1 - i] = da
    } else if da == 1 {
      result[n - 1 - i] = db
    } else if db == 1 {
      result[n - 1 - i] = da
    } else {
      return Err(
        "cannot broadcast shapes " + a.to_string() + " and " + b.to_string(),
      )
    }
  }
  shape_new(result)
}

///|
/// Compute output shape for matmul: [..., m, k] @ [..., k, n] -> [..., m, n]
pub fn shape_matmul(a : Shape, b : Shape) -> Result[Shape, String] {
  if a.ndim() < 2 || b.ndim() < 2 {
    return Err("matmul requires at least 2D tensors")
  }
  let m = a.dim(-2)
  let k1 = a.dim(-1)
  let k2 = b.dim(-2)
  let n = b.dim(-1)
  if k1 != k2 {
    return Err(
      "matmul inner dimensions mismatch: " +
      k1.to_string() +
      " vs " +
      k2.to_string(),
    )
  }
  // Handle batch dimensions
  let batch_a = Array::makei(a.ndim() - 2, fn(i) { a.dims[i] })
  let batch_b = Array::makei(b.ndim() - 2, fn(i) { b.dims[i] })
  let batch_shape = if batch_a.length() == 0 && batch_b.length() == 0 {
    Ok(Shape::{ dims: [] })
  } else if batch_a.length() == 0 {
    shape_new(batch_b)
  } else if batch_b.length() == 0 {
    shape_new(batch_a)
  } else {
    shape_broadcast(Shape::{ dims: batch_a }, Shape::{ dims: batch_b })
  }
  match batch_shape {
    Ok(batch) => {
      let result = batch.dims.copy()
      result.push(m)
      result.push(n)
      shape_new(result)
    }
    Err(e) => Err(e)
  }
}

///|
pub fn shape_transpose(
  s : Shape,
  dim0 : Int,
  dim1 : Int,
) -> Result[Shape, String] {
  let n = s.ndim()
  let d0 = if dim0 < 0 { n + dim0 } else { dim0 }
  let d1 = if dim1 < 0 { n + dim1 } else { dim1 }
  if d0 < 0 || d0 >= n || d1 < 0 || d1 >= n {
    return Err("transpose dimensions out of range")
  }
  let result = s.dims.copy()
  let tmp = result[d0]
  result[d0] = result[d1]
  result[d1] = tmp
  shape_new(result)
}
