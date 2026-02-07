///|
/// A variable in the computation graph, referencing a node in the tape.
pub struct Var {
  id : Int
  tape : Tape
}

///|
/// Access the tensor data of this variable.
pub fn Var::data(self : Var) -> @tensor.Tensor {
  self.tape.nodes[self.id].data
}

///|
/// Access the gradient of this variable (available after backward).
pub fn Var::grad(self : Var) -> @tensor.Tensor? {
  self.tape.nodes[self.id].grad
}

///|
/// Matrix multiplication: a @ b
/// Supports 2D and 3D (batched with broadcast) matmul.
/// Uses BLAS sgemm with trans flags to avoid transpose copies.
pub fn var_matmul(a : Var, b : Var) -> Var {
  let tape = a.tape
  let a_data = a.data().contiguous()
  let b_data = b.data().contiguous()
  let result = @tensor.tensor_matmul(a_data, b_data).unwrap()
  tape.push_node(result, fn(dy) {
    let dyc = dy.contiguous()
    if a_data.ndim() == 2 && b_data.ndim() == 2 {
      let m = a_data.dim(0)
      let k = a_data.dim(1)
      let n = b_data.dim(1)
      let one = Float::from_int(1)
      let zero = Float::from_int(0)
      // dA = dY @ B^T: sgemm(NoTrans, Trans, m, k, n, 1, dy, n, b, n, 0, da, k)
      let da_data = FixedArray::make(m * k, zero)
      @tensor.sgemm(0, 1, m, k, n, one, dyc.data, n, b_data.data, n, zero, da_data, k)
      let da = @tensor.tensor_new_fixed(a_data.shape, da_data)
      accumulate(tape, a.id, da)
      // dB = A^T @ dY: sgemm(Trans, NoTrans, k, n, m, 1, a, k, dy, n, 0, db, n)
      let db_data = FixedArray::make(k * n, zero)
      @tensor.sgemm(1, 0, k, n, m, one, a_data.data, k, dyc.data, n, zero, db_data, n)
      let db = @tensor.tensor_new_fixed(b_data.shape, db_data)
      accumulate(tape, b.id, db)
    } else if a_data.ndim() == 3 && b_data.ndim() == 2 {
      // [batch, m, k] @ [k, n] -> [batch, m, n]
      let batch = a_data.dim(0)
      let m = a_data.dim(1)
      let k = b_data.dim(0)
      let n = b_data.dim(1)
      let n_rows = batch * m
      let one = Float::from_int(1)
      let zero = Float::from_int(0)
      // dA = dY @ B^T: treat dY as [batch*m, n], da as [batch*m, k]
      let da_data = FixedArray::make(n_rows * k, zero)
      @tensor.sgemm(
        0, 1, n_rows, k, n, one, dyc.data, n, b_data.data, n, zero, da_data, k,
      )
      let da = @tensor.tensor_new_fixed(a_data.shape, da_data)
      accumulate(tape, a.id, da)
      // dB = A_flat^T @ dY_flat: treat a as [batch*m, k], dy as [batch*m, n]
      // sgemm(Trans, NoTrans, k, n, batch*m, 1, a, k, dy, n, 0, db, n)
      let db_data = FixedArray::make(k * n, zero)
      @tensor.sgemm(
        1, 0, k, n, n_rows, one, a_data.data, k, dyc.data, n, zero, db_data, n,
      )
      let db = @tensor.tensor_new_fixed(b_data.shape, db_data)
      accumulate(tape, b.id, db)
    } else {
      abort("var_matmul backward: unsupported shape combination")
    }
  })
}

///|
/// Element-wise addition: a + b (same shape).
pub fn var_add(a : Var, b : Var) -> Var {
  let tape = a.tape
  let result = @tensor.tensor_add(a.data(), b.data()).unwrap()
  tape.push_node(result, fn(dy) {
    accumulate(tape, a.id, dy)
    accumulate(tape, b.id, dy)
  })
}

///|
/// Broadcast add bias: x [*, features] + bias [features].
/// Adds bias to the last dimension. Uses C FFI for vectorized ops.
pub fn var_add_bias(x : Var, bias : Var) -> Var {
  let tape = x.tape
  let xd = x.data().contiguous()
  let bd = bias.data().contiguous()
  let last_dim = xd.dim(-1)
  let total = xd.numel()
  let result_data = FixedArray::make(total, Float::from_int(0))
  @tensor.bias_add_forward(xd.data, bd.data, result_data, total, last_dim)
  let result = @tensor.tensor_new_fixed(xd.shape, result_data)
  tape.push_node(result, fn(dy) {
    // dx = dy (same shape)
    accumulate(tape, x.id, dy)
    // dbias = sum over all dims except last
    let dyc = dy.contiguous()
    let db_data = FixedArray::make(last_dim, Float::from_int(0))
    @tensor.bias_add_backward(dyc.data, db_data, dyc.numel(), last_dim)
    let db = @tensor.tensor_new_fixed(bd.shape, db_data)
    accumulate(tape, bias.id, db)
  })
}

///|
/// Scalar multiplication: x * s.
pub fn var_mul_scalar(x : Var, s : Float) -> Var {
  let tape = x.tape
  let result = @tensor.tensor_scale(x.data(), s)
  tape.push_node(result, fn(dy) {
    let dx = @tensor.tensor_scale(dy, s)
    accumulate(tape, x.id, dx)
  })
}

///|
/// Sum all elements to a scalar.
pub fn var_sum(x : Var) -> Var {
  let tape = x.tape
  let xd = x.data().contiguous()
  let mut total = Float::from_int(0)
  for i = 0; i < xd.numel(); i = i + 1 {
    total += xd.data[i]
  }
  let result_shape = @tensor.shape_new([1]).unwrap()
  let result = @tensor.tensor_new_fixed(
    result_shape,
    FixedArray::make(1, total),
  )
  tape.push_node(result, fn(dy) {
    // d(sum)/dx = ones * dy_scalar
    let dy_val = dy.contiguous().data[0]
    let dx = @tensor.tensor_full(xd.shape, dy_val)
    accumulate(tape, x.id, dx)
  })
}
