///|
pub enum MlpError {
  InvalidSpec(String)
  InvalidShape(String)
} derive(Show, Eq)

///|
pub fn mlp_error_invalid_spec(message : String) -> MlpError {
  InvalidSpec(message)
}

///|
pub fn mlp_error_invalid_shape(message : String) -> MlpError {
  InvalidShape(message)
}

///|
pub struct MlpSpec {
  input_size : Int
  hidden_size : Int
  output_size : Int
  batch_size : Int
} derive(Show, Eq)

///|
pub fn mlp_spec_new(
  input_size : Int,
  hidden_size : Int,
  output_size : Int,
  batch_size : Int,
) -> Result[MlpSpec, MlpError] {
  let spec = { input_size, hidden_size, output_size, batch_size }
  mlp_validate_spec(spec).map(fn(_) { spec })
}

///|
pub fn mlp_validate_spec(spec : MlpSpec) -> Result[Unit, MlpError] {
  if spec.input_size <= 0 {
    Err(InvalidSpec("input_size must be > 0"))
  } else if spec.hidden_size <= 0 {
    Err(InvalidSpec("hidden_size must be > 0"))
  } else if spec.output_size <= 0 {
    Err(InvalidSpec("output_size must be > 0"))
  } else if spec.batch_size <= 0 {
    Err(InvalidSpec("batch_size must be > 0"))
  } else {
    Ok(())
  }
}

///|
pub struct MlpParams {
  weight1 : Array[Float]
  bias1 : Array[Float]
  weight2 : Array[Float]
  bias2 : Array[Float]
} derive(Show, Eq)

///|
pub enum MlpInitPolicy {
  Zeros
  Deterministic
  XavierUniform
  HeUniform
} derive(Show, Eq)

///|
pub let mlp_init_policy_zeros : MlpInitPolicy = Zeros

///|
pub let mlp_init_policy_deterministic : MlpInitPolicy = Deterministic

///|
pub let mlp_init_policy_xavier_uniform : MlpInitPolicy = XavierUniform

///|
pub let mlp_init_policy_he_uniform : MlpInitPolicy = HeUniform

///|
fn expect_length(
  name : String,
  actual : Int,
  expected : Int,
) -> Result[Unit, MlpError] {
  if actual == expected {
    Ok(())
  } else {
    Err(InvalidShape(name + " length mismatch"))
  }
}

///|
pub fn mlp_validate_params(
  spec : MlpSpec,
  params : MlpParams,
) -> Result[Unit, MlpError] {
  let w1 = elements_weight1(spec)
  let b1 = elements_bias1(spec)
  let w2 = elements_weight2(spec)
  let b2 = elements_bias2(spec)
  expect_length("weight1", params.weight1.length(), w1)
  .bind(fn(_) { expect_length("bias1", params.bias1.length(), b1) })
  .bind(fn(_) { expect_length("weight2", params.weight2.length(), w2) })
  .bind(fn(_) { expect_length("bias2", params.bias2.length(), b2) })
}

///|
pub fn mlp_params_new(
  spec : MlpSpec,
  weight1 : Array[Float],
  bias1 : Array[Float],
  weight2 : Array[Float],
  bias2 : Array[Float],
) -> Result[MlpParams, MlpError] {
  let params = { weight1, bias1, weight2, bias2 }
  mlp_validate_spec(spec).bind(fn(_) {
    mlp_validate_params(spec, params).map(fn(_) { params })
  })
}

///|
pub struct MlpBufferPlan {
  input_bytes : Int
  hidden_bytes : Int
  output_bytes : Int
  weight1_bytes : Int
  bias1_bytes : Int
  weight2_bytes : Int
  bias2_bytes : Int
} derive(Show, Eq)

///|
fn bytes_for_f32(count : Int) -> Int {
  count * 4
}

///|
fn bytes_for_u32(count : Int) -> Int {
  count * 4
}

///|
fn elements_input(spec : MlpSpec) -> Int {
  spec.batch_size * spec.input_size
}

///|
fn elements_hidden(spec : MlpSpec) -> Int {
  spec.batch_size * spec.hidden_size
}

///|
fn elements_output(spec : MlpSpec) -> Int {
  spec.batch_size * spec.output_size
}

///|
fn elements_weight1(spec : MlpSpec) -> Int {
  spec.input_size * spec.hidden_size
}

///|
fn elements_bias1(spec : MlpSpec) -> Int {
  spec.hidden_size
}

///|
fn elements_weight2(spec : MlpSpec) -> Int {
  spec.hidden_size * spec.output_size
}

///|
fn elements_bias2(spec : MlpSpec) -> Int {
  spec.output_size
}

///|
pub fn mlp_plan_buffers(spec : MlpSpec) -> Result[MlpBufferPlan, MlpError] {
  mlp_validate_spec(spec).map(fn(_) {
    {
      input_bytes: bytes_for_f32(elements_input(spec)),
      hidden_bytes: bytes_for_f32(elements_hidden(spec)),
      output_bytes: bytes_for_f32(elements_output(spec)),
      weight1_bytes: bytes_for_f32(elements_weight1(spec)),
      bias1_bytes: bytes_for_f32(elements_bias1(spec)),
      weight2_bytes: bytes_for_f32(elements_weight2(spec)),
      bias2_bytes: bytes_for_f32(elements_bias2(spec)),
    }
  })
}

///|
pub struct MlpLossBufferPlan {
  labels_bytes : Int
  loss_bytes : Int
  probs_bytes : Int
  loss_sum_bytes : Int
} derive(Show, Eq)

///|
pub fn mlp_plan_loss_buffers(
  spec : MlpSpec,
) -> Result[MlpLossBufferPlan, MlpError] {
  mlp_validate_spec(spec).map(fn(_) {
    {
      labels_bytes: bytes_for_u32(spec.batch_size),
      loss_bytes: bytes_for_f32(spec.batch_size),
      probs_bytes: bytes_for_f32(elements_output(spec)),
      loss_sum_bytes: bytes_for_f32(1),
    }
  })
}

///|
pub struct MlpTrainBufferPlan {
  grad_weight1_bytes : Int
  grad_bias1_bytes : Int
  grad_weight2_bytes : Int
  grad_bias2_bytes : Int
} derive(Show, Eq)

///|
pub fn mlp_plan_train_buffers(
  spec : MlpSpec,
) -> Result[MlpTrainBufferPlan, MlpError] {
  mlp_validate_spec(spec).map(fn(_) {
    {
      grad_weight1_bytes: bytes_for_f32(elements_weight1(spec)),
      grad_bias1_bytes: bytes_for_f32(elements_bias1(spec)),
      grad_weight2_bytes: bytes_for_f32(elements_weight2(spec)),
      grad_bias2_bytes: bytes_for_f32(elements_bias2(spec)),
    }
  })
}

///|
pub struct MlpResourcePlan {
  input : @wgpu.BufferDescriptor
  hidden : @wgpu.BufferDescriptor
  output : @wgpu.BufferDescriptor
  weight1 : @wgpu.BufferDescriptor
  bias1 : @wgpu.BufferDescriptor
  weight2 : @wgpu.BufferDescriptor
  bias2 : @wgpu.BufferDescriptor
} derive(Show, Eq)

///|
fn buffer_usage_storage_upload() -> @wgpu.BufferUsage {
  @wgpu.buffer_usage_or(@wgpu.buffer_usage_storage, @wgpu.buffer_usage_copy_dst)
}

///|
fn buffer_usage_storage_readback() -> @wgpu.BufferUsage {
  @wgpu.buffer_usage_or(
    buffer_usage_storage_upload(),
    @wgpu.buffer_usage_copy_src,
  )
}

///|
fn buffer_desc(
  size : Int,
  usage : @wgpu.BufferUsage,
  label : String,
) -> @wgpu.BufferDescriptor {
  @wgpu.buffer_descriptor(size, usage, false, Some(label))
}

///|
pub fn mlp_plan_resources(spec : MlpSpec) -> Result[MlpResourcePlan, MlpError] {
  mlp_plan_buffers(spec).map(fn(plan) {
    let upload = buffer_usage_storage_upload()
    let readback = buffer_usage_storage_readback()
    {
      input: buffer_desc(plan.input_bytes, upload, "mlp_input"),
      hidden: buffer_desc(plan.hidden_bytes, upload, "mlp_hidden"),
      output: buffer_desc(plan.output_bytes, readback, "mlp_output"),
      weight1: buffer_desc(plan.weight1_bytes, readback, "mlp_w1"),
      bias1: buffer_desc(plan.bias1_bytes, readback, "mlp_b1"),
      weight2: buffer_desc(plan.weight2_bytes, readback, "mlp_w2"),
      bias2: buffer_desc(plan.bias2_bytes, readback, "mlp_b2"),
    }
  })
}

///|
pub struct MlpLossResourcePlan {
  labels : @wgpu.BufferDescriptor
  loss : @wgpu.BufferDescriptor
  probs : @wgpu.BufferDescriptor
  loss_sum : @wgpu.BufferDescriptor
} derive(Show, Eq)

///|
pub fn mlp_plan_loss_resources(
  spec : MlpSpec,
) -> Result[MlpLossResourcePlan, MlpError] {
  mlp_plan_loss_buffers(spec).map(fn(plan) {
    let upload = buffer_usage_storage_upload()
    let readback = buffer_usage_storage_readback()
    {
      labels: buffer_desc(plan.labels_bytes, upload, "mlp_labels"),
      loss: buffer_desc(plan.loss_bytes, readback, "mlp_loss"),
      probs: buffer_desc(plan.probs_bytes, readback, "mlp_probs"),
      loss_sum: buffer_desc(plan.loss_sum_bytes, readback, "mlp_loss_sum"),
    }
  })
}

///|
pub fn mlp_validate_input(
  spec : MlpSpec,
  input : Array[Float],
) -> Result[Unit, MlpError] {
  expect_length("input", input.length(), elements_input(spec))
}

///|
fn init_value(index : Int, seed : Int) -> Float {
  let raw = (index + seed) % 23
  let centered = raw - 11
  Float::from_int(centered) / Float::from_int(10)
}

///|
fn init_array(len : Int, seed : Int, offset : Int) -> Array[Float] {
  Array::makei(len, fn(i) { init_value(i + offset, seed) })
}

///|
fn init_unit(index : Int, seed : Int) -> Float {
  let raw = (index + seed) % 23
  let centered = raw - 11
  Float::from_int(centered) / Float::from_int(11)
}

///|
fn init_scaled(
  len : Int,
  seed : Int,
  offset : Int,
  scale : Float,
) -> Array[Float] {
  Array::makei(len, fn(i) { init_unit(i + offset, seed) * scale })
}

///|
fn zeros_array(len : Int) -> Array[Float] {
  Array::make(len, Float::from_int(0))
}

///|
fn xavier_scale(fan_in : Int, fan_out : Int) -> Float {
  let denom = Float::from_int(fan_in) + Float::from_int(fan_out)
  Float::sqrt(Float::from_int(6) / denom)
}

///|
fn he_scale(fan_in : Int) -> Float {
  Float::sqrt(Float::from_int(6) / Float::from_int(fan_in))
}

///|
pub fn mlp_init_params(
  spec : MlpSpec,
  seed : Int,
) -> Result[MlpParams, MlpError] {
  mlp_init_params_with_policy(spec, seed, Deterministic)
}

///|
pub fn mlp_init_params_with_policy(
  spec : MlpSpec,
  seed : Int,
  policy : MlpInitPolicy,
) -> Result[MlpParams, MlpError] {
  mlp_validate_spec(spec).bind(fn(_) {
    match policy {
      Zeros => {
        let w1 = zeros_array(elements_weight1(spec))
        let b1 = zeros_array(elements_bias1(spec))
        let w2 = zeros_array(elements_weight2(spec))
        let b2 = zeros_array(elements_bias2(spec))
        mlp_params_new(spec, w1, b1, w2, b2)
      }
      Deterministic => {
        let w1 = init_array(elements_weight1(spec), seed, 0)
        let b1 = init_array(elements_bias1(spec), seed, 1000)
        let w2 = init_array(elements_weight2(spec), seed, 2000)
        let b2 = init_array(elements_bias2(spec), seed, 3000)
        mlp_params_new(spec, w1, b1, w2, b2)
      }
      XavierUniform => {
        let scale1 = xavier_scale(spec.input_size, spec.hidden_size)
        let scale2 = xavier_scale(spec.hidden_size, spec.output_size)
        let w1 = init_scaled(elements_weight1(spec), seed, 0, scale1)
        let w2 = init_scaled(elements_weight2(spec), seed, 2000, scale2)
        let b1 = zeros_array(elements_bias1(spec))
        let b2 = zeros_array(elements_bias2(spec))
        mlp_params_new(spec, w1, b1, w2, b2)
      }
      HeUniform => {
        let scale1 = he_scale(spec.input_size)
        let scale2 = he_scale(spec.hidden_size)
        let w1 = init_scaled(elements_weight1(spec), seed, 0, scale1)
        let w2 = init_scaled(elements_weight2(spec), seed, 2000, scale2)
        let b1 = zeros_array(elements_bias1(spec))
        let b2 = zeros_array(elements_bias2(spec))
        mlp_params_new(spec, w1, b1, w2, b2)
      }
    }
  })
}

///|
pub struct MlpCpuResult {
  hidden : Array[Float]
  output : Array[Float]
} derive(Show, Eq)

///|
fn relu(x : Float) -> Float {
  let zero = Float::from_int(0)
  if x < zero {
    zero
  } else {
    x
  }
}

///|
pub fn mlp_forward(
  spec : MlpSpec,
  params : MlpParams,
  input : Array[Float],
) -> Result[MlpCpuResult, MlpError] {
  mlp_validate_spec(spec)
  .bind(fn(_) { mlp_validate_params(spec, params) })
  .bind(fn(_) { mlp_validate_input(spec, input) })
  .map(fn(_) {
    let zero = Float::from_int(0)
    let hidden = Array::make(elements_hidden(spec), zero)
    let output = Array::make(elements_output(spec), zero)
    for b = 0; b < spec.batch_size; b = b + 1 {
      let input_offset = b * spec.input_size
      let hidden_offset = b * spec.hidden_size
      let output_offset = b * spec.output_size
      for j = 0; j < spec.hidden_size; j = j + 1 {
        let acc = for i = 0, acc = params.bias1[j]
                      i < spec.input_size
                      i = i + 1, acc = acc +
                        input[input_offset + i] *
                        params.weight1[i * spec.hidden_size + j] {

        } else {
          acc
        }
        hidden[hidden_offset + j] = relu(acc)
      }
      for k = 0; k < spec.output_size; k = k + 1 {
        let acc = for j = 0, acc = params.bias2[k]
                      j < spec.hidden_size
                      j = j + 1, acc = acc +
                        hidden[hidden_offset + j] *
                        params.weight2[j * spec.output_size + k] {

        } else {
          acc
        }
        output[output_offset + k] = acc
      }
    }
    { hidden, output }
  })
}

///|
fn expect_bytes_length(
  name : String,
  actual : Int,
  expected : Int,
) -> Result[Unit, MlpError] {
  if actual == expected {
    Ok(())
  } else {
    Err(InvalidShape(name + " bytes length mismatch"))
  }
}

///|
fn read_u32_le(bytes : Bytes, offset : Int) -> UInt {
  for j = 0, acc = UInt::default()
      j < 4
      j = j + 1, acc = acc | (bytes[offset + j].to_uint() << (j * 8)) {

  } else {
    acc
  }
}

///|
fn bytes_to_floats(bytes : Bytes, count : Int) -> Array[Float] {
  Array::makei(count, fn(i) {
    let bits = read_u32_le(bytes, i * 4)
    Float::reinterpret_from_uint(bits)
  })
}

///|
fn bytes_to_floats_from(
  bytes : Bytes,
  offset : Int,
  count : Int,
) -> Array[Float] {
  Array::makei(count, fn(i) {
    let bits = read_u32_le(bytes, offset + i * 4)
    Float::reinterpret_from_uint(bits)
  })
}

///|
fn floats_to_bytes(values : Array[Float]) -> Bytes {
  let total = values.length() * 4
  Bytes::makei(total, fn(idx) {
    let i = idx / 4
    let j = idx % 4
    let bits = Float::reinterpret_as_uint(values[i])
    (bits >> (j * 8)).to_byte()
  })
}

///|
pub fn mlp_input_to_bytes(
  spec : MlpSpec,
  input : Array[Float],
) -> Result[Bytes, MlpError] {
  mlp_validate_input(spec, input).map(fn(_) { floats_to_bytes(input) })
}

///|
pub fn mlp_input_bytes_to_array(
  spec : MlpSpec,
  bytes : Bytes,
) -> Result[Array[Float], MlpError] {
  let expected = elements_input(spec) * 4
  expect_bytes_length("input", bytes.length(), expected).map(fn(_) {
    bytes_to_floats(bytes, elements_input(spec))
  })
}

///|
fn expect_output_length(spec : MlpSpec, actual : Int) -> Result[Unit, MlpError] {
  let expected = elements_output(spec)
  if actual == expected {
    Ok(())
  } else {
    Err(InvalidShape("output length mismatch"))
  }
}

///|

///|
pub fn mlp_output_from_bytes(
  spec : MlpSpec,
  output : Array[Float],
) -> Result[Bytes, MlpError] {
  expect_output_length(spec, output.length()).map(fn(_) {
    floats_to_bytes(output)
  })
}

///|
pub fn mlp_output_bytes_to_array(
  spec : MlpSpec,
  bytes : Bytes,
) -> Result[Array[Float], MlpError] {
  let expected = elements_output(spec) * 4
  expect_bytes_length("output", bytes.length(), expected).map(fn(_) {
    bytes_to_floats(bytes, elements_output(spec))
  })
}

///|
pub fn mlp_params_bytes_length(spec : MlpSpec) -> Result[Int, MlpError] {
  mlp_validate_spec(spec).map(fn(_) {
    let total = elements_weight1(spec) +
      elements_bias1(spec) +
      elements_weight2(spec) +
      elements_bias2(spec)
    total * 4
  })
}

///|
pub fn mlp_params_to_bytes(
  spec : MlpSpec,
  params : MlpParams,
) -> Result[Bytes, MlpError] {
  mlp_validate_spec(spec)
  .bind(fn(_) { mlp_validate_params(spec, params) })
  .map(fn(_) {
    let w1_len = elements_weight1(spec)
    let b1_len = elements_bias1(spec)
    let w2_len = elements_weight2(spec)
    let b2_len = elements_bias2(spec)
    let total = (w1_len + b1_len + w2_len + b2_len) * 4
    Bytes::makei(total, fn(idx) {
      let i = idx / 4
      let j = idx % 4
      let v = if i < w1_len {
        params.weight1[i]
      } else if i < w1_len + b1_len {
        params.bias1[i - w1_len]
      } else if i < w1_len + b1_len + w2_len {
        params.weight2[i - w1_len - b1_len]
      } else {
        params.bias2[i - w1_len - b1_len - w2_len]
      }
      let bits = Float::reinterpret_as_uint(v)
      (bits >> (j * 8)).to_byte()
    })
  })
}

///|
pub fn mlp_params_from_bytes(
  spec : MlpSpec,
  bytes : Bytes,
) -> Result[MlpParams, MlpError] {
  mlp_validate_spec(spec).bind(fn(_) {
    let w1_len = elements_weight1(spec)
    let b1_len = elements_bias1(spec)
    let w2_len = elements_weight2(spec)
    let b2_len = elements_bias2(spec)
    let expected = (w1_len + b1_len + w2_len + b2_len) * 4
    expect_bytes_length("params", bytes.length(), expected).bind(fn(_) {
      let mut offset = 0
      let w1 = bytes_to_floats_from(bytes, offset, w1_len)
      offset = offset + w1_len * 4
      let b1 = bytes_to_floats_from(bytes, offset, b1_len)
      offset = offset + b1_len * 4
      let w2 = bytes_to_floats_from(bytes, offset, w2_len)
      offset = offset + w2_len * 4
      let b2 = bytes_to_floats_from(bytes, offset, b2_len)
      mlp_params_new(spec, w1, b1, w2, b2)
    })
  })
}

///|
pub struct MlpDispatchPlan {
  workgroup_size : Int
  layer1_dispatch_x : Int
  layer2_dispatch_x : Int
} derive(Show, Eq)

///|
fn ceil_div(n : Int, d : Int) -> Int {
  (n + d - 1) / d
}

///|
pub fn mlp_dispatch_plan(
  spec : MlpSpec,
  workgroup_size : Int,
) -> Result[MlpDispatchPlan, MlpError] {
  if workgroup_size <= 0 {
    Err(InvalidSpec("workgroup_size must be > 0"))
  } else {
    mlp_validate_spec(spec).map(fn(_) {
      let layer1_elements = elements_hidden(spec)
      let layer2_elements = elements_output(spec)
      {
        workgroup_size,
        layer1_dispatch_x: ceil_div(layer1_elements, workgroup_size),
        layer2_dispatch_x: ceil_div(layer2_elements, workgroup_size),
      }
    })
  }
}

///|
pub fn mlp_loss_dispatch_x(
  spec : MlpSpec,
  workgroup_size : Int,
) -> Result[Int, MlpError] {
  if workgroup_size <= 0 {
    Err(InvalidSpec("workgroup_size must be > 0"))
  } else {
    mlp_validate_spec(spec).map(fn(_) {
      ceil_div(spec.batch_size, workgroup_size)
    })
  }
}

///|
pub fn mlp_loss_reduce_dispatch_x(
  spec : MlpSpec,
  workgroup_size : Int,
) -> Result[Int, MlpError] {
  if workgroup_size <= 0 {
    Err(InvalidSpec("workgroup_size must be > 0"))
  } else {
    mlp_validate_spec(spec).map(fn(_) { 1 })
  }
}

///|
pub struct MlpTrainDispatchPlan {
  workgroup_size : Int
  grad_w1_dispatch_x : Int
  grad_b1_dispatch_x : Int
  grad_w2_dispatch_x : Int
  grad_b2_dispatch_x : Int
  update_w1_dispatch_x : Int
  update_b1_dispatch_x : Int
  update_w2_dispatch_x : Int
  update_b2_dispatch_x : Int
} derive(Show, Eq)

///|
pub fn mlp_train_dispatch_plan(
  spec : MlpSpec,
  workgroup_size : Int,
) -> Result[MlpTrainDispatchPlan, MlpError] {
  if workgroup_size <= 0 {
    Err(InvalidSpec("workgroup_size must be > 0"))
  } else {
    mlp_validate_spec(spec).map(fn(_) {
      let w1 = elements_weight1(spec)
      let b1 = elements_bias1(spec)
      let w2 = elements_weight2(spec)
      let b2 = elements_bias2(spec)
      {
        workgroup_size,
        grad_w1_dispatch_x: ceil_div(w1, workgroup_size),
        grad_b1_dispatch_x: ceil_div(b1, workgroup_size),
        grad_w2_dispatch_x: ceil_div(w2, workgroup_size),
        grad_b2_dispatch_x: ceil_div(b2, workgroup_size),
        update_w1_dispatch_x: ceil_div(w1, workgroup_size),
        update_b1_dispatch_x: ceil_div(b1, workgroup_size),
        update_w2_dispatch_x: ceil_div(w2, workgroup_size),
        update_b2_dispatch_x: ceil_div(b2, workgroup_size),
      }
    })
  }
}

///|
fn u32_lit(value : Int) -> String {
  value.to_string() + "u"
}

///|
fn wgsl_layer1(spec : MlpSpec, workgroup_size : Int) -> String {
  let input = u32_lit(spec.input_size)
  let hidden = u32_lit(spec.hidden_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP layer1: input -> hidden (ReLU)\n"
  let s1 = s0 + "const INPUT_SIZE : u32 = " + input + ";\n"
  let s2 = s1 + "const HIDDEN_SIZE : u32 = " + hidden + ";\n"
  let s3 = s2 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s4 = s3 + "@group(0) @binding(0) var<storage, read> input : array<f32>;\n"
  let s5 = s4 +
    "@group(0) @binding(1) var<storage, read> weight1 : array<f32>;\n"
  let s6 = s5 + "@group(0) @binding(2) var<storage, read> bias1 : array<f32>;\n"
  let s7 = s6 +
    "@group(0) @binding(3) var<storage, read_write> hidden : array<f32>;\n"
  let s8 = s7 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s9 = s8 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s10 = s9 + "  let index = gid.x;\n"
  let s11 = s10 + "  if (index >= BATCH_SIZE * HIDDEN_SIZE) { return; }\n"
  let s12 = s11 + "  let b = index / HIDDEN_SIZE;\n"
  let s13 = s12 + "  let j = index % HIDDEN_SIZE;\n"
  let s14 = s13 + "  var acc = bias1[j];\n"
  let s15 = s14 + "  for (var i : u32 = 0u; i < INPUT_SIZE; i = i + 1u) {\n"
  let s16 = s15 + "    let input_idx = b * INPUT_SIZE + i;\n"
  let s17 = s16 + "    let w_idx = i * HIDDEN_SIZE + j;\n"
  let s18 = s17 + "    acc = acc + input[input_idx] * weight1[w_idx];\n"
  let s19 = s18 + "  }\n"
  let s20 = s19 + "  if (acc < 0.0) { acc = 0.0; }\n"
  let s21 = s20 + "  hidden[index] = acc;\n"
  s21 + "}\n"
}

///|
fn wgsl_layer2(spec : MlpSpec, workgroup_size : Int) -> String {
  let hidden = u32_lit(spec.hidden_size)
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP layer2: hidden -> output\n"
  let s1 = s0 + "const HIDDEN_SIZE : u32 = " + hidden + ";\n"
  let s2 = s1 + "const OUTPUT_SIZE : u32 = " + output + ";\n"
  let s3 = s2 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s4 = s3 +
    "@group(0) @binding(0) var<storage, read> hidden : array<f32>;\n"
  let s5 = s4 +
    "@group(0) @binding(1) var<storage, read> weight2 : array<f32>;\n"
  let s6 = s5 + "@group(0) @binding(2) var<storage, read> bias2 : array<f32>;\n"
  let s7 = s6 +
    "@group(0) @binding(3) var<storage, read_write> output : array<f32>;\n"
  let s8 = s7 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s9 = s8 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s10 = s9 + "  let index = gid.x;\n"
  let s11 = s10 + "  if (index >= BATCH_SIZE * OUTPUT_SIZE) { return; }\n"
  let s12 = s11 + "  let b = index / OUTPUT_SIZE;\n"
  let s13 = s12 + "  let k = index % OUTPUT_SIZE;\n"
  let s14 = s13 + "  var acc = bias2[k];\n"
  let s15 = s14 + "  for (var j : u32 = 0u; j < HIDDEN_SIZE; j = j + 1u) {\n"
  let s16 = s15 + "    let hidden_idx = b * HIDDEN_SIZE + j;\n"
  let s17 = s16 + "    let w_idx = j * OUTPUT_SIZE + k;\n"
  let s18 = s17 + "    acc = acc + hidden[hidden_idx] * weight2[w_idx];\n"
  let s19 = s18 + "  }\n"
  let s20 = s19 + "  output[index] = acc;\n"
  s20 + "}\n"
}

///|
fn wgsl_loss(spec : MlpSpec, workgroup_size : Int) -> String {
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP loss: softmax + cross-entropy\n"
  let s1 = s0 + "const OUTPUT_SIZE : u32 = " + output + ";\n"
  let s2 = s1 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s3 = s2 + "const EPS : f32 = 0.000001;\n"
  let s4 = s3 +
    "@group(0) @binding(0) var<storage, read> logits : array<f32>;\n"
  let s5 = s4 +
    "@group(0) @binding(1) var<storage, read> labels : array<u32>;\n"
  let s6 = s5 +
    "@group(0) @binding(2) var<storage, read_write> loss : array<f32>;\n"
  let s7 = s6 +
    "@group(0) @binding(3) var<storage, read_write> probs : array<f32>;\n"
  let s8 = s7 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s9 = s8 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s10 = s9 + "  let b = gid.x;\n"
  let s11 = s10 + "  if (b >= BATCH_SIZE) { return; }\n"
  let s12 = s11 + "  let base = b * OUTPUT_SIZE;\n"
  let s13 = s12 + "  var max_val = logits[base];\n"
  let s14 = s13 + "  for (var k : u32 = 1u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s15 = s14 + "    let v = logits[base + k];\n"
  let s16 = s15 + "    if (v > max_val) { max_val = v; }\n"
  let s17 = s16 + "  }\n"
  let s18 = s17 + "  var sum = 0.0;\n"
  let s19 = s18 + "  for (var k : u32 = 0u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s20 = s19 + "    sum = sum + exp(logits[base + k] - max_val);\n"
  let s21 = s20 + "  }\n"
  let s22 = s21 + "  for (var k : u32 = 0u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s23 = s22 +
    "    probs[base + k] = exp(logits[base + k] - max_val) / sum;\n"
  let s24 = s23 + "  }\n"
  let s25 = s24 + "  let label = labels[b];\n"
  let s26 = s25 + "  let prob = probs[base + label];\n"
  let s27 = s26 + "  loss[b] = -log(prob + EPS);\n"
  s27 + "}\n"
}

///|
fn wgsl_loss_reduce(spec : MlpSpec, workgroup_size : Int) -> String {
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP loss reduce: sum over batch\n"
  let s1 = s0 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s2 = s1 + "@group(0) @binding(0) var<storage, read> loss : array<f32>;\n"
  let s3 = s2 +
    "@group(0) @binding(1) var<storage, read_write> loss_sum : array<f32>;\n"
  let s4 = s3 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s5 = s4 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s6 = s5 + "  if (gid.x != 0u) { return; }\n"
  let s7 = s6 + "  var acc = 0.0;\n"
  let s8 = s7 + "  for (var i : u32 = 0u; i < BATCH_SIZE; i = i + 1u) {\n"
  let s9 = s8 + "    acc = acc + loss[i];\n"
  let s10 = s9 + "  }\n"
  let s11 = s10 + "  loss_sum[0] = acc;\n"
  s11 + "}\n"
}

///|
fn wgsl_grad_w2(spec : MlpSpec, workgroup_size : Int) -> String {
  let hidden = u32_lit(spec.hidden_size)
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP grad w2\n"
  let s1 = s0 + "const HIDDEN_SIZE : u32 = " + hidden + ";\n"
  let s2 = s1 + "const OUTPUT_SIZE : u32 = " + output + ";\n"
  let s3 = s2 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s4 = s3 +
    "@group(0) @binding(0) var<storage, read> hidden : array<f32>;\n"
  let s5 = s4 + "@group(0) @binding(1) var<storage, read> probs : array<f32>;\n"
  let s6 = s5 +
    "@group(0) @binding(2) var<storage, read> labels : array<u32>;\n"
  let s7 = s6 +
    "@group(0) @binding(3) var<storage, read_write> grad_w2 : array<f32>;\n"
  let s8 = s7 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s9 = s8 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s10 = s9 + "  let index = gid.x;\n"
  let s11 = s10 + "  if (index >= HIDDEN_SIZE * OUTPUT_SIZE) { return; }\n"
  let s12 = s11 + "  let j = index / OUTPUT_SIZE;\n"
  let s13 = s12 + "  let k = index % OUTPUT_SIZE;\n"
  let s14 = s13 + "  var acc = 0.0;\n"
  let s15 = s14 + "  for (var b : u32 = 0u; b < BATCH_SIZE; b = b + 1u) {\n"
  let s16 = s15 + "    let base = b * OUTPUT_SIZE;\n"
  let s17 = s16 + "    var dlogit = probs[base + k];\n"
  let s18 = s17 + "    let label = labels[b];\n"
  let s19 = s18 + "    if (label == k) { dlogit = dlogit - 1.0; }\n"
  let s20 = s19 + "    let hidden_idx = b * HIDDEN_SIZE + j;\n"
  let s21 = s20 + "    acc = acc + hidden[hidden_idx] * dlogit;\n"
  let s22 = s21 + "  }\n"
  let s23 = s22 + "  grad_w2[index] = acc;\n"
  s23 + "}\n"
}

///|
fn wgsl_grad_b2(spec : MlpSpec, workgroup_size : Int) -> String {
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP grad b2\n"
  let s1 = s0 + "const OUTPUT_SIZE : u32 = " + output + ";\n"
  let s2 = s1 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s3 = s2 + "@group(0) @binding(0) var<storage, read> probs : array<f32>;\n"
  let s4 = s3 +
    "@group(0) @binding(1) var<storage, read> labels : array<u32>;\n"
  let s5 = s4 +
    "@group(0) @binding(2) var<storage, read_write> grad_b2 : array<f32>;\n"
  let s6 = s5 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s7 = s6 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s8 = s7 + "  let k = gid.x;\n"
  let s9 = s8 + "  if (k >= OUTPUT_SIZE) { return; }\n"
  let s10 = s9 + "  var acc = 0.0;\n"
  let s11 = s10 + "  for (var b : u32 = 0u; b < BATCH_SIZE; b = b + 1u) {\n"
  let s12 = s11 + "    let base = b * OUTPUT_SIZE;\n"
  let s13 = s12 + "    var dlogit = probs[base + k];\n"
  let s14 = s13 + "    let label = labels[b];\n"
  let s15 = s14 + "    if (label == k) { dlogit = dlogit - 1.0; }\n"
  let s16 = s15 + "    acc = acc + dlogit;\n"
  let s17 = s16 + "  }\n"
  let s18 = s17 + "  grad_b2[k] = acc;\n"
  s18 + "}\n"
}

///|
fn wgsl_grad_w1(spec : MlpSpec, workgroup_size : Int) -> String {
  let input = u32_lit(spec.input_size)
  let hidden = u32_lit(spec.hidden_size)
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP grad w1\n"
  let s1 = s0 + "const INPUT_SIZE : u32 = " + input + ";\n"
  let s2 = s1 + "const HIDDEN_SIZE : u32 = " + hidden + ";\n"
  let s3 = s2 + "const OUTPUT_SIZE : u32 = " + output + ";\n"
  let s4 = s3 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s5 = s4 + "@group(0) @binding(0) var<storage, read> input : array<f32>;\n"
  let s6 = s5 +
    "@group(0) @binding(1) var<storage, read> hidden : array<f32>;\n"
  let s7 = s6 + "@group(0) @binding(2) var<storage, read> probs : array<f32>;\n"
  let s8 = s7 +
    "@group(0) @binding(3) var<storage, read> labels : array<u32>;\n"
  let s9 = s8 +
    "@group(0) @binding(4) var<storage, read> weight2 : array<f32>;\n"
  let s10 = s9 +
    "@group(0) @binding(5) var<storage, read_write> grad_w1 : array<f32>;\n"
  let s11 = s10 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s12 = s11 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s13 = s12 + "  let index = gid.x;\n"
  let s14 = s13 + "  if (index >= INPUT_SIZE * HIDDEN_SIZE) { return; }\n"
  let s15 = s14 + "  let i = index / HIDDEN_SIZE;\n"
  let s16 = s15 + "  let j = index % HIDDEN_SIZE;\n"
  let s17 = s16 + "  var acc = 0.0;\n"
  let s18 = s17 + "  for (var b : u32 = 0u; b < BATCH_SIZE; b = b + 1u) {\n"
  let s19 = s18 + "    let hidden_idx = b * HIDDEN_SIZE + j;\n"
  let s20 = s19 + "    let h = hidden[hidden_idx];\n"
  let s21 = s20 + "    if (h <= 0.0) { continue; }\n"
  let s22 = s21 + "    var sum = 0.0;\n"
  let s23 = s22 + "    let base = b * OUTPUT_SIZE;\n"
  let s24 = s23 + "    let label = labels[b];\n"
  let s25 = s24 + "    for (var k : u32 = 0u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s26 = s25 + "      var dlogit = probs[base + k];\n"
  let s27 = s26 + "      if (label == k) { dlogit = dlogit - 1.0; }\n"
  let s28 = s27 + "      let w2_idx = j * OUTPUT_SIZE + k;\n"
  let s29 = s28 + "      sum = sum + dlogit * weight2[w2_idx];\n"
  let s30 = s29 + "    }\n"
  let s31 = s30 + "    let input_idx = b * INPUT_SIZE + i;\n"
  let s32 = s31 + "    acc = acc + input[input_idx] * sum;\n"
  let s33 = s32 + "  }\n"
  let s34 = s33 + "  grad_w1[index] = acc;\n"
  s34 + "}\n"
}

///|
fn wgsl_grad_b1(spec : MlpSpec, workgroup_size : Int) -> String {
  let hidden = u32_lit(spec.hidden_size)
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP grad b1\n"
  let s1 = s0 + "const HIDDEN_SIZE : u32 = " + hidden + ";\n"
  let s2 = s1 + "const OUTPUT_SIZE : u32 = " + output + ";\n"
  let s3 = s2 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s4 = s3 +
    "@group(0) @binding(0) var<storage, read> hidden : array<f32>;\n"
  let s5 = s4 + "@group(0) @binding(1) var<storage, read> probs : array<f32>;\n"
  let s6 = s5 +
    "@group(0) @binding(2) var<storage, read> labels : array<u32>;\n"
  let s7 = s6 +
    "@group(0) @binding(3) var<storage, read> weight2 : array<f32>;\n"
  let s8 = s7 +
    "@group(0) @binding(4) var<storage, read_write> grad_b1 : array<f32>;\n"
  let s9 = s8 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s10 = s9 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s11 = s10 + "  let j = gid.x;\n"
  let s12 = s11 + "  if (j >= HIDDEN_SIZE) { return; }\n"
  let s13 = s12 + "  var acc = 0.0;\n"
  let s14 = s13 + "  for (var b : u32 = 0u; b < BATCH_SIZE; b = b + 1u) {\n"
  let s15 = s14 + "    let hidden_idx = b * HIDDEN_SIZE + j;\n"
  let s16 = s15 + "    let h = hidden[hidden_idx];\n"
  let s17 = s16 + "    if (h <= 0.0) { continue; }\n"
  let s18 = s17 + "    var sum = 0.0;\n"
  let s19 = s18 + "    let base = b * OUTPUT_SIZE;\n"
  let s20 = s19 + "    let label = labels[b];\n"
  let s21 = s20 + "    for (var k : u32 = 0u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s22 = s21 + "      var dlogit = probs[base + k];\n"
  let s23 = s22 + "      if (label == k) { dlogit = dlogit - 1.0; }\n"
  let s24 = s23 + "      let w2_idx = j * OUTPUT_SIZE + k;\n"
  let s25 = s24 + "      sum = sum + dlogit * weight2[w2_idx];\n"
  let s26 = s25 + "    }\n"
  let s27 = s26 + "    acc = acc + sum;\n"
  let s28 = s27 + "  }\n"
  let s29 = s28 + "  grad_b1[j] = acc;\n"
  s29 + "}\n"
}

///|
fn wgsl_update_param(
  name : String,
  total : Int,
  spec : MlpSpec,
  workgroup_size : Int,
) -> String {
  let total_lit = u32_lit(total)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP update: " + name + "\n"
  let s1 = s0 + "const TOTAL : u32 = " + total_lit + ";\n"
  let s2 = s1 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s3 = s2 +
    "@group(0) @binding(0) var<storage, read_write> param : array<f32>;\n"
  let s4 = s3 + "@group(0) @binding(1) var<storage, read> grad : array<f32>;\n"
  let s5 = s4 + "@group(0) @binding(2) var<storage, read> lr : array<f32>;\n"
  let s6 = s5 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s7 = s6 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s8 = s7 + "  let index = gid.x;\n"
  let s9 = s8 + "  if (index >= TOTAL) { return; }\n"
  let s10 = s9 + "  let scale = lr[0] / f32(BATCH_SIZE);\n"
  let s11 = s10 + "  param[index] = param[index] - grad[index] * scale;\n"
  s11 + "}\n"
}

///|
pub struct MlpShaderPlan {
  workgroup_size : Int
  layer1_wgsl : String
  layer2_wgsl : String
} derive(Show, Eq)

///|
pub struct MlpLossShaderPlan {
  workgroup_size : Int
  loss_wgsl : String
  loss_reduce_wgsl : String
} derive(Show, Eq)

///|
pub struct MlpTrainShaderPlan {
  workgroup_size : Int
  grad_w1_wgsl : String
  grad_b1_wgsl : String
  grad_w2_wgsl : String
  grad_b2_wgsl : String
  update_w1_wgsl : String
  update_b1_wgsl : String
  update_w2_wgsl : String
  update_b2_wgsl : String
} derive(Show, Eq)

///|
pub fn mlp_shader_plan(
  spec : MlpSpec,
  workgroup_size : Int,
) -> Result[MlpShaderPlan, MlpError] {
  if workgroup_size <= 0 {
    Err(InvalidSpec("workgroup_size must be > 0"))
  } else {
    mlp_validate_spec(spec).map(fn(_) {
      {
        workgroup_size,
        layer1_wgsl: wgsl_layer1(spec, workgroup_size),
        layer2_wgsl: wgsl_layer2(spec, workgroup_size),
      }
    })
  }
}

///|
pub fn mlp_loss_shader_plan(
  spec : MlpSpec,
  workgroup_size : Int,
) -> Result[MlpLossShaderPlan, MlpError] {
  if workgroup_size <= 0 {
    Err(InvalidSpec("workgroup_size must be > 0"))
  } else {
    mlp_validate_spec(spec).map(fn(_) {
      {
        workgroup_size,
        loss_wgsl: wgsl_loss(spec, workgroup_size),
        loss_reduce_wgsl: wgsl_loss_reduce(spec, workgroup_size),
      }
    })
  }
}

///|
pub fn mlp_train_shader_plan(
  spec : MlpSpec,
  workgroup_size : Int,
) -> Result[MlpTrainShaderPlan, MlpError] {
  if workgroup_size <= 0 {
    Err(InvalidSpec("workgroup_size must be > 0"))
  } else {
    mlp_validate_spec(spec).map(fn(_) {
      {
        workgroup_size,
        grad_w1_wgsl: wgsl_grad_w1(spec, workgroup_size),
        grad_b1_wgsl: wgsl_grad_b1(spec, workgroup_size),
        grad_w2_wgsl: wgsl_grad_w2(spec, workgroup_size),
        grad_b2_wgsl: wgsl_grad_b2(spec, workgroup_size),
        update_w1_wgsl: wgsl_update_param(
          "w1",
          elements_weight1(spec),
          spec,
          workgroup_size,
        ),
        update_b1_wgsl: wgsl_update_param(
          "b1",
          elements_bias1(spec),
          spec,
          workgroup_size,
        ),
        update_w2_wgsl: wgsl_update_param(
          "w2",
          elements_weight2(spec),
          spec,
          workgroup_size,
        ),
        update_b2_wgsl: wgsl_update_param(
          "b2",
          elements_bias2(spec),
          spec,
          workgroup_size,
        ),
      }
    })
  }
}

///|
pub fn mlp_shader_plan_default() -> MlpShaderPlan {
  let spec = { input_size: 1, hidden_size: 1, output_size: 1, batch_size: 1 }
  {
    workgroup_size: 64,
    layer1_wgsl: wgsl_layer1(spec, 64),
    layer2_wgsl: wgsl_layer2(spec, 64),
  }
}

///|
pub fn mlp_loss_shader_plan_default() -> MlpLossShaderPlan {
  let spec = { input_size: 1, hidden_size: 1, output_size: 1, batch_size: 1 }
  {
    workgroup_size: 64,
    loss_wgsl: wgsl_loss(spec, 64),
    loss_reduce_wgsl: wgsl_loss_reduce(spec, 64),
  }
}
