///|
test "sdot" {
  let f = Float::from_int
  let x = [f(1), f(2), f(3)]
  let y = [f(4), f(5), f(6)]
  let result = sdot(x, y)
  // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
  inspect(Float::is_close(result, f(32)), content="true")
}

///|
test "saxpy" {
  let f = Float::from_int
  let x = [f(1), f(2), f(3)]
  let y = [f(4), f(5), f(6)]
  saxpy(f(2), x, y)
  // y = 2*x + y = [2+4, 4+5, 6+6] = [6, 9, 12]
  inspect(Float::is_close(y[0], f(6)), content="true")
  inspect(Float::is_close(y[1], f(9)), content="true")
  inspect(Float::is_close(y[2], f(12)), content="true")
}

///|
test "snrm2" {
  let f = Float::from_int
  let x = [f(3), f(4)]
  let result = snrm2(x)
  // ||[3,4]||_2 = 5
  inspect(Float::is_close(result, f(5)), content="true")
}

///|
test "sgemv" {
  let f = Float::from_int
  // A = [[1,2], [3,4]], x = [1, 1], y = A @ x = [3, 7]
  let a = [f(1), f(2), f(3), f(4)] // row-major 2x2
  let x = [f(1), f(1)]
  let y = [f(0), f(0)]
  sgemv(a, x, y, 2, 2)
  inspect(Float::is_close(y[0], f(3)), content="true")
  inspect(Float::is_close(y[1], f(7)), content="true")
}

///|
test "sgemm" {
  let f = Float::from_int
  // A = [[1,2], [3,4]], B = [[1,0], [0,1]] (identity)
  // C = A @ B = A
  let a = [f(1), f(2), f(3), f(4)]
  let b = [f(1), f(0), f(0), f(1)]
  let c = [f(0), f(0), f(0), f(0)]
  sgemm(a, b, c, 2, 2, 2)
  inspect(Float::is_close(c[0], f(1)), content="true")
  inspect(Float::is_close(c[1], f(2)), content="true")
  inspect(Float::is_close(c[2], f(3)), content="true")
  inspect(Float::is_close(c[3], f(4)), content="true")
}

///|
test "sgemm_non_square" {
  let f = Float::from_int
  // A = [[1,2,3], [4,5,6]] (2x3)
  // B = [[1], [0], [1]] (3x1)
  // C = A @ B = [[4], [10]] (2x1)
  let a = [f(1), f(2), f(3), f(4), f(5), f(6)]
  let b = [f(1), f(0), f(1)]
  let c = [f(0), f(0)]
  sgemm(a, b, c, 2, 1, 3)
  inspect(Float::is_close(c[0], f(4)), content="true")
  inspect(Float::is_close(c[1], f(10)), content="true")
}

// ============================================================================
// Optimized C buffer tests
// ============================================================================

///|
test "float_buffer_roundtrip" {
  let f = Float::from_int
  let arr = [f(1), f(2), f(3), f(4)]
  let buf = alloc_floats(4)
  copy_to_buffer(buf, arr)
  // Check individual values
  inspect(Float::is_close(get_float(buf, 0), f(1)), content="true")
  inspect(Float::is_close(get_float(buf, 1), f(2)), content="true")
  inspect(Float::is_close(get_float(buf, 2), f(3)), content="true")
  inspect(Float::is_close(get_float(buf, 3), f(4)), content="true")
  // Modify and check
  set_float(buf, 2, f(99))
  inspect(Float::is_close(get_float(buf, 2), f(99)), content="true")
  // Copy back
  let out : Array[Float] = [f(0), f(0), f(0), f(0)]
  copy_from_buffer(out, buf)
  inspect(Float::is_close(out[0], f(1)), content="true")
  inspect(Float::is_close(out[2], f(99)), content="true")
  free_floats(buf)
}

///|
test "sgemm_direct" {
  let f = Float::from_int
  // Same as sgemm test but with C buffers
  let a_arr = [f(1), f(2), f(3), f(4)]
  let b_arr = [f(1), f(0), f(0), f(1)]
  let c_arr : Array[Float] = [f(0), f(0), f(0), f(0)]
  let a = alloc_floats(4)
  let b = alloc_floats(4)
  let c = alloc_floats(4)
  copy_to_buffer(a, a_arr)
  copy_to_buffer(b, b_arr)
  copy_to_buffer(c, c_arr)
  sgemm_direct(a, b, c, 2, 2, 2)
  inspect(Float::is_close(get_float(c, 0), f(1)), content="true")
  inspect(Float::is_close(get_float(c, 1), f(2)), content="true")
  inspect(Float::is_close(get_float(c, 2), f(3)), content="true")
  inspect(Float::is_close(get_float(c, 3), f(4)), content="true")
  free_floats(a)
  free_floats(b)
  free_floats(c)
}

///|
test "layer1_fused" {
  let f = Float::from_int
  // batch=2, in_dim=3, out_dim=2
  // input: [[1,2,3], [4,5,6]] (2x3)
  // weight: [[1,0], [0,1], [1,1]] (3x2)
  // bias: [1, -1]
  // Before ReLU: input @ weight + bias
  //   row0: [1*1+2*0+3*1, 1*0+2*1+3*1] + [1,-1] = [4, 5] + [1,-1] = [5, 4]
  //   row1: [4*1+5*0+6*1, 4*0+5*1+6*1] + [1,-1] = [10, 11] + [1,-1] = [11, 10]
  // After ReLU: [5, 4, 11, 10]
  let input_arr = [f(1), f(2), f(3), f(4), f(5), f(6)]
  let weight_arr = [f(1), f(0), f(0), f(1), f(1), f(1)]
  let bias_arr = [f(1), f(-1)]
  let input = alloc_floats(6)
  let weight = alloc_floats(6)
  let bias = alloc_floats(2)
  let output = alloc_floats(4)
  copy_to_buffer(input, input_arr)
  copy_to_buffer(weight, weight_arr)
  copy_to_buffer(bias, bias_arr)
  layer1_fused(input, weight, bias, output, 2, 3, 2)
  inspect(Float::is_close(get_float(output, 0), f(5)), content="true")
  inspect(Float::is_close(get_float(output, 1), f(4)), content="true")
  inspect(Float::is_close(get_float(output, 2), f(11)), content="true")
  inspect(Float::is_close(get_float(output, 3), f(10)), content="true")
  free_floats(input)
  free_floats(weight)
  free_floats(bias)
  free_floats(output)
}

///|
test "layer2_fused" {
  let f = Float::from_int
  // Same setup as layer1 but no ReLU
  // Test with negative bias to verify no ReLU applied
  let input_arr = [f(1), f(2), f(3), f(4), f(5), f(6)]
  let weight_arr = [f(1), f(0), f(0), f(1), f(1), f(1)]
  let bias_arr = [f(-10), f(-20)] // Large negative to test no ReLU
  let input = alloc_floats(6)
  let weight = alloc_floats(6)
  let bias = alloc_floats(2)
  let output = alloc_floats(4)
  copy_to_buffer(input, input_arr)
  copy_to_buffer(weight, weight_arr)
  copy_to_buffer(bias, bias_arr)
  layer2_fused(input, weight, bias, output, 2, 3, 2)
  // row0: [4, 5] + [-10, -20] = [-6, -15]
  // row1: [10, 11] + [-10, -20] = [0, -9]
  inspect(Float::is_close(get_float(output, 0), f(-6)), content="true")
  inspect(Float::is_close(get_float(output, 1), f(-15)), content="true")
  inspect(Float::is_close(get_float(output, 2), f(0)), content="true")
  inspect(Float::is_close(get_float(output, 3), f(-9)), content="true")
  free_floats(input)
  free_floats(weight)
  free_floats(bias)
  free_floats(output)
}
