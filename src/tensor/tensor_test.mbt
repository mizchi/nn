///|
test "shape_new" {
  let s = shape_new([2, 3, 4]).unwrap()
  inspect(s.ndim(), content="3")
  inspect(s.dim(0), content="2")
  inspect(s.dim(1), content="3")
  inspect(s.dim(2), content="4")
  inspect(s.dim(-1), content="4")
  inspect(s.numel(), content="24")
}

///|
test "shape_strides" {
  let s = shape_new([2, 3, 4]).unwrap()
  let strides = s.strides()
  inspect(strides[0], content="12")
  inspect(strides[1], content="4")
  inspect(strides[2], content="1")
}

///|
test "shape_matmul" {
  let a = shape_new([2, 3]).unwrap()
  let b = shape_new([3, 4]).unwrap()
  let c = shape_matmul(a, b).unwrap()
  inspect(c.dim(0), content="2")
  inspect(c.dim(1), content="4")
}

///|
test "tensor_zeros" {
  let s = shape_new([2, 3]).unwrap()
  let t = tensor_zeros(s)
  inspect(t.numel(), content="6")
  inspect(t.at2(0, 0), content="0")
}

///|
test "tensor_ones" {
  let s = shape_new([2, 3]).unwrap()
  let t = tensor_ones(s)
  inspect(t.at2(1, 2), content="1")
}

///|
test "tensor_from_array" {
  let f = Float::from_int
  let t = tensor_from_array([f(1), f(2), f(3)])
  inspect(t.ndim(), content="1")
  inspect(t.dim(0), content="3")
  inspect(t.at1(1), content="2")
}

///|
test "tensor_add" {
  let f = Float::from_int
  let s = shape_new([2, 2]).unwrap()
  let a = tensor_new(s, [f(1), f(2), f(3), f(4)]).unwrap()
  let b = tensor_new(s, [f(10), f(20), f(30), f(40)]).unwrap()
  let c = tensor_add(a, b).unwrap()
  inspect(c.at2(0, 0), content="11")
  inspect(c.at2(1, 1), content="44")
}

///|
test "tensor_scale" {
  let f = Float::from_int
  let s = shape_new([2]).unwrap()
  let t = tensor_new(s, [f(2), f(4)]).unwrap()
  let scaled = tensor_scale(t, f(3))
  inspect(scaled.at1(0), content="6")
  inspect(scaled.at1(1), content="12")
}

///|
test "tensor_matmul_2d" {
  let f = Float::from_int
  // [2, 3] @ [3, 2] -> [2, 2]
  let a_shape = shape_new([2, 3]).unwrap()
  let b_shape = shape_new([3, 2]).unwrap()
  let a = tensor_new(a_shape, [f(1), f(2), f(3), f(4), f(5), f(6)]).unwrap()
  let b = tensor_new(b_shape, [f(7), f(8), f(9), f(10), f(11), f(12)]).unwrap()
  let c = tensor_matmul_2d(a, b).unwrap()
  inspect(c.dim(0), content="2")
  inspect(c.dim(1), content="2")
  // c[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
  inspect(c.at2(0, 0), content="58")
  // c[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
  inspect(c.at2(0, 1), content="64")
}

///|
test "tensor_transpose" {
  let f = Float::from_int
  let s = shape_new([2, 3]).unwrap()
  let t = tensor_new(s, [f(1), f(2), f(3), f(4), f(5), f(6)]).unwrap()
  let transposed = t.transpose(0, 1).unwrap()
  inspect(transposed.dim(0), content="3")
  inspect(transposed.dim(1), content="2")
  // Original: [[1,2,3],[4,5,6]]
  // Transposed: [[1,4],[2,5],[3,6]]
  inspect(transposed.at2(0, 0), content="1")
  inspect(transposed.at2(0, 1), content="4")
  inspect(transposed.at2(1, 0), content="2")
  inspect(transposed.at2(2, 1), content="6")
}

///|
test "tensor_reshape" {
  let f = Float::from_int
  let s = shape_new([2, 3]).unwrap()
  let t = tensor_new(s, [f(1), f(2), f(3), f(4), f(5), f(6)]).unwrap()
  let new_shape = shape_new([3, 2]).unwrap()
  let reshaped = t.reshape(new_shape).unwrap()
  inspect(reshaped.dim(0), content="3")
  inspect(reshaped.dim(1), content="2")
  inspect(reshaped.at2(0, 0), content="1")
  inspect(reshaped.at2(2, 1), content="6")
}

///|
test "tensor_softmax" {
  let f = Float::from_int
  let s = shape_new([2, 3]).unwrap()
  let t = tensor_new(s, [f(1), f(2), f(3), f(1), f(2), f(3)]).unwrap()
  let sm = tensor_softmax(t).unwrap()
  // Each row should sum to 1
  let row0_sum = sm.at2(0, 0) + sm.at2(0, 1) + sm.at2(0, 2)
  let row1_sum = sm.at2(1, 0) + sm.at2(1, 1) + sm.at2(1, 2)
  let one = f(1)
  let ok0 = Float::is_close(row0_sum, one)
  let ok1 = Float::is_close(row1_sum, one)
  inspect(ok0, content="true")
  inspect(ok1, content="true")
}

///|
test "tensor_relu" {
  let f = Float::from_int
  let t = tensor_from_array([f(-2), f(-1), f(0), f(1), f(2)])
  let r = tensor_relu(t)
  inspect(r.at1(0), content="0")
  inspect(r.at1(1), content="0")
  inspect(r.at1(2), content="0")
  inspect(r.at1(3), content="1")
  inspect(r.at1(4), content="2")
}

///|
test "tensor_gelu" {
  let f = Float::from_int
  let t = tensor_from_array([f(0), f(1), f(-1)])
  let g = tensor_gelu(t)
  // GELU(0) = 0
  let ok0 = Float::is_close(g.at1(0), f(0))
  inspect(ok0, content="true")
  // GELU(1) ~= 0.841
  let gelu1 = g.at1(1)
  let ok1 = gelu1 > (0.8 |> Float::from_double) &&
    gelu1 < (0.9 |> Float::from_double)
  inspect(ok1, content="true")
}

///|
test "tensor_layer_norm" {
  let f = Float::from_int
  let s = shape_new([2, 4]).unwrap()
  let t = tensor_new(s, [f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8)]).unwrap()
  let gamma_shape = shape_new([4]).unwrap()
  let gamma = tensor_new(gamma_shape, [f(1), f(1), f(1), f(1)]).unwrap()
  let beta = tensor_new(gamma_shape, [f(0), f(0), f(0), f(0)]).unwrap()
  let eps = 0.00001 |> Float::from_double
  let ln = tensor_layer_norm(t, gamma, beta, eps).unwrap()
  // After layer norm, each row should have mean ~0 and std ~1
  let row0_mean = (ln.at2(0, 0) + ln.at2(0, 1) + ln.at2(0, 2) + ln.at2(0, 3)) /
    f(4)
  let ok_mean = Float::is_close(row0_mean, f(0))
  inspect(ok_mean, content="true")
}

///|
test "tensor_sum" {
  let f = Float::from_int
  let s = shape_new([2, 3]).unwrap()
  let t = tensor_new(s, [f(1), f(2), f(3), f(4), f(5), f(6)]).unwrap()
  // Sum over dim 0: [5, 7, 9]
  let sum0 = tensor_sum(t, 0).unwrap()
  inspect(sum0.dim(0), content="3")
  inspect(sum0.at1(0), content="5")
  inspect(sum0.at1(1), content="7")
  inspect(sum0.at1(2), content="9")
  // Sum over dim 1: [6, 15]
  let sum1 = tensor_sum(t, 1).unwrap()
  inspect(sum1.dim(0), content="2")
  inspect(sum1.at1(0), content="6")
  inspect(sum1.at1(1), content="15")
}

///|
test "tensor_slice" {
  let f = Float::from_int
  let s = shape_new([4, 2]).unwrap()
  let t = tensor_new(s, [f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8)]).unwrap()
  let sliced = t.slice(1, 3).unwrap()
  inspect(sliced.dim(0), content="2")
  inspect(sliced.dim(1), content="2")
  inspect(sliced.at2(0, 0), content="3")
  inspect(sliced.at2(1, 1), content="6")
}
