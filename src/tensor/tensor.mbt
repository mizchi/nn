///|
/// Tensor represents an N-dimensional array with shape metadata.
/// CPU-backed with FixedArray[Float] for zero-copy FFI.
pub struct Tensor {
  data : FixedArray[Float]
  shape : Shape
  offset : Int // For views into larger arrays
  strides : Array[Int] // For non-contiguous views
}

///|
pub fn tensor_new(shape : Shape, data : Array[Float]) -> Result[Tensor, String] {
  if data.length() != shape.numel() {
    return Err(
      "data length " +
      data.length().to_string() +
      " does not match shape " +
      shape.to_string() +
      " (numel=" +
      shape.numel().to_string() +
      ")",
    )
  }
  let fa = FixedArray::make(data.length(), Float::from_int(0))
  for i = 0; i < data.length(); i = i + 1 {
    fa[i] = data[i]
  }
  Ok(Tensor::{ data: fa, shape, offset: 0, strides: shape.strides() })
}

///|
/// Internal: create Tensor from FixedArray without copying (no validation)
fn tensor_new_fixed(shape : Shape, data : FixedArray[Float]) -> Tensor {
  Tensor::{ data, shape, offset: 0, strides: shape.strides() }
}

///|
pub fn tensor_zeros(shape : Shape) -> Tensor {
  let data = FixedArray::make(shape.numel(), Float::from_int(0))
  Tensor::{ data, shape, offset: 0, strides: shape.strides() }
}

///|
pub fn tensor_ones(shape : Shape) -> Tensor {
  let data = FixedArray::make(shape.numel(), Float::from_int(1))
  Tensor::{ data, shape, offset: 0, strides: shape.strides() }
}

///|
pub fn tensor_full(shape : Shape, value : Float) -> Tensor {
  let data = FixedArray::make(shape.numel(), value)
  Tensor::{ data, shape, offset: 0, strides: shape.strides() }
}

///|
pub fn tensor_from_array(arr : Array[Float]) -> Tensor {
  let shape = Shape::{ dims: [arr.length()] }
  let fa = FixedArray::make(arr.length(), Float::from_int(0))
  for i = 0; i < arr.length(); i = i + 1 {
    fa[i] = arr[i]
  }
  Tensor::{ data: fa, shape, offset: 0, strides: [1] }
}

///|
pub fn Tensor::ndim(self : Tensor) -> Int {
  self.shape.ndim()
}

///|
pub fn Tensor::dim(self : Tensor, i : Int) -> Int {
  self.shape.dim(i)
}

///|
pub fn Tensor::numel(self : Tensor) -> Int {
  self.shape.numel()
}

///|
pub fn Tensor::is_contiguous(self : Tensor) -> Bool {
  let expected = self.shape.strides()
  if self.strides.length() != expected.length() {
    return false
  }
  for i = 0; i < self.strides.length(); i = i + 1 {
    if self.strides[i] != expected[i] {
      return false
    }
  }
  true
}

///|
/// Compute flat index from N-dimensional indices
fn Tensor::flat_index(self : Tensor, indices : Array[Int]) -> Int {
  let mut idx = self.offset
  for i = 0; i < indices.length(); i = i + 1 {
    idx = idx + indices[i] * self.strides[i]
  }
  idx
}

///|
pub fn Tensor::get(self : Tensor, indices : Array[Int]) -> Float {
  self.data[self.flat_index(indices)]
}

///|
pub fn Tensor::set(self : Tensor, indices : Array[Int], value : Float) -> Unit {
  self.data[self.flat_index(indices)] = value
}

///|
/// Get element for 1D tensor
pub fn Tensor::at1(self : Tensor, i : Int) -> Float {
  self.data[self.offset + i * self.strides[0]]
}

///|
/// Get element for 2D tensor
pub fn Tensor::at2(self : Tensor, i : Int, j : Int) -> Float {
  self.data[self.offset + i * self.strides[0] + j * self.strides[1]]
}

///|
/// Get element for 3D tensor
pub fn Tensor::at3(self : Tensor, i : Int, j : Int, k : Int) -> Float {
  self.data[self.offset +
  i * self.strides[0] +
  j * self.strides[1] +
  k * self.strides[2]]
}

///|
/// Set element for 2D tensor
pub fn Tensor::set2(self : Tensor, i : Int, j : Int, value : Float) -> Unit {
  self.data[self.offset + i * self.strides[0] + j * self.strides[1]] = value
}

///|
/// Create a contiguous copy of this tensor
pub fn Tensor::contiguous(self : Tensor) -> Tensor {
  if self.is_contiguous() && self.offset == 0 {
    return self
  }
  let new_data = FixedArray::make(self.numel(), Float::from_int(0))
  copy_tensor_data(self, new_data)
  Tensor::{
    data: new_data,
    shape: self.shape,
    offset: 0,
    strides: self.shape.strides(),
  }
}

///|
fn copy_tensor_data(src : Tensor, dst : FixedArray[Float]) -> Unit {
  let n = src.ndim()
  if n == 1 {
    for i = 0; i < src.dim(0); i = i + 1 {
      dst[i] = src.at1(i)
    }
  } else if n == 2 {
    let mut idx = 0
    for i = 0; i < src.dim(0); i = i + 1 {
      for j = 0; j < src.dim(1); j = j + 1 {
        dst[idx] = src.at2(i, j)
        idx = idx + 1
      }
    }
  } else if n == 3 {
    let mut idx = 0
    for i = 0; i < src.dim(0); i = i + 1 {
      for j = 0; j < src.dim(1); j = j + 1 {
        for k = 0; k < src.dim(2); k = k + 1 {
          dst[idx] = src.at3(i, j, k)
          idx = idx + 1
        }
      }
    }
  } else {
    // Generic N-D copy using recursive indexing
    let _ = copy_nd(src, dst, [], 0)

  }
}

///|
fn copy_nd(
  src : Tensor,
  dst : FixedArray[Float],
  indices : Array[Int],
  dst_idx : Int,
) -> Int {
  let depth = indices.length()
  if depth == src.ndim() {
    dst[dst_idx] = src.get(indices)
    return dst_idx + 1
  }
  let mut idx = dst_idx
  for i = 0; i < src.dim(depth); i = i + 1 {
    let new_indices = indices.copy()
    new_indices.push(i)
    idx = copy_nd(src, dst, new_indices, idx)
  }
  idx
}

///|
pub fn Tensor::to_string(self : Tensor) -> String {
  "Tensor(" + self.shape.to_string() + ")"
}

///|
pub impl Show for Tensor with output(self, logger) {
  logger.write_string(self.to_string())
}

///|
/// Reshape tensor to new shape (must have same numel)
pub fn Tensor::reshape(
  self : Tensor,
  new_shape : Shape,
) -> Result[Tensor, String] {
  if self.numel() != new_shape.numel() {
    return Err(
      "cannot reshape " +
      self.shape.to_string() +
      " to " +
      new_shape.to_string(),
    )
  }
  let t = self.contiguous()
  Ok(Tensor::{
    data: t.data,
    shape: new_shape,
    offset: 0,
    strides: new_shape.strides(),
  })
}

///|
/// Transpose two dimensions
pub fn Tensor::transpose(
  self : Tensor,
  dim0 : Int,
  dim1 : Int,
) -> Result[Tensor, String] {
  match shape_transpose(self.shape, dim0, dim1) {
    Err(e) => Err(e)
    Ok(new_shape) => {
      let n = self.ndim()
      let d0 = if dim0 < 0 { n + dim0 } else { dim0 }
      let d1 = if dim1 < 0 { n + dim1 } else { dim1 }
      let new_strides = self.strides.copy()
      let tmp = new_strides[d0]
      new_strides[d0] = new_strides[d1]
      new_strides[d1] = tmp
      Ok(Tensor::{
        data: self.data,
        shape: new_shape,
        offset: self.offset,
        strides: new_strides,
      })
    }
  }
}

///|
/// View a slice along the first dimension
pub fn Tensor::slice(
  self : Tensor,
  start : Int,
  end : Int,
) -> Result[Tensor, String] {
  if start < 0 || end > self.dim(0) || start >= end {
    return Err("invalid slice range")
  }
  let new_dims = self.shape.dims.copy()
  new_dims[0] = end - start
  let new_shape = Shape::{ dims: new_dims }
  Ok(Tensor::{
    data: self.data,
    shape: new_shape,
    offset: self.offset + start * self.strides[0],
    strides: self.strides,
  })
}

///|
/// Get raw data as Array[Float] (always copies)
pub fn Tensor::to_array(self : Tensor) -> Array[Float] {
  let n = self.numel()
  let result = Array::make(n, Float::from_int(0))
  if self.is_contiguous() && self.offset == 0 {
    for i = 0; i < n; i = i + 1 {
      result[i] = self.data[i]
    }
  } else {
    let tmp = FixedArray::make(n, Float::from_int(0))
    copy_tensor_data(self, tmp)
    for i = 0; i < n; i = i + 1 {
      result[i] = tmp[i]
    }
  }
  result
}
