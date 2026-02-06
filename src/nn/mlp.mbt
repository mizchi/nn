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
  correct_bytes : Int
  correct_sum_bytes : Int
  epoch_loss_bytes : Int
  epoch_correct_bytes : Int
  epoch_seen_bytes : Int
} derive(Show, Eq)

///|
pub struct MlpDatasetBufferPlan {
  dataset_input_bytes : Int
  dataset_label_bytes : Int
  indices_bytes : Int
  shuffle_params_bytes : Int
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
      correct_bytes: bytes_for_f32(spec.batch_size),
      correct_sum_bytes: bytes_for_f32(1),
      epoch_loss_bytes: bytes_for_f32(1),
      epoch_correct_bytes: bytes_for_f32(1),
      epoch_seen_bytes: bytes_for_f32(1),
    }
  })
}

///|
pub fn mlp_plan_loss_buffers_for_dataset(
  spec : MlpSpec,
  dataset_count : Int,
) -> Result[MlpLossBufferPlan, MlpError] {
  if dataset_count <= 0 {
    return Err(InvalidSpec("dataset_count must be > 0"))
  }
  mlp_validate_spec(spec).map(fn(_) {
    {
      labels_bytes: bytes_for_u32(dataset_count),
      loss_bytes: bytes_for_f32(spec.batch_size),
      probs_bytes: bytes_for_f32(elements_output(spec)),
      loss_sum_bytes: bytes_for_f32(1),
      correct_bytes: bytes_for_f32(spec.batch_size),
      correct_sum_bytes: bytes_for_f32(1),
      epoch_loss_bytes: bytes_for_f32(1),
      epoch_correct_bytes: bytes_for_f32(1),
      epoch_seen_bytes: bytes_for_f32(1),
    }
  })
}

///|
pub fn mlp_plan_dataset_buffers(
  spec : MlpSpec,
  dataset_count : Int,
) -> Result[MlpDatasetBufferPlan, MlpError] {
  if dataset_count <= 0 {
    return Err(InvalidSpec("dataset_count must be > 0"))
  }
  mlp_validate_spec(spec).map(fn(_) {
    {
      dataset_input_bytes: bytes_for_f32(dataset_count * spec.input_size),
      dataset_label_bytes: bytes_for_u32(dataset_count),
      indices_bytes: bytes_for_u32(dataset_count + 1),
      shuffle_params_bytes: bytes_for_u32(2),
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
  correct : @wgpu.BufferDescriptor
  correct_sum : @wgpu.BufferDescriptor
  epoch_loss : @wgpu.BufferDescriptor
  epoch_correct : @wgpu.BufferDescriptor
  epoch_seen : @wgpu.BufferDescriptor
} derive(Show, Eq)

///|
pub fn mlp_plan_loss_resources(
  spec : MlpSpec,
) -> Result[MlpLossResourcePlan, MlpError] {
  mlp_plan_loss_buffers(spec).map(fn(plan) {
    let upload = buffer_usage_storage_upload()
    let readback = buffer_usage_storage_readback()
    let storage = @wgpu.buffer_usage_storage
    {
      labels: buffer_desc(plan.labels_bytes, upload, "mlp_labels"),
      loss: buffer_desc(plan.loss_bytes, readback, "mlp_loss"),
      probs: buffer_desc(plan.probs_bytes, readback, "mlp_probs"),
      loss_sum: buffer_desc(plan.loss_sum_bytes, readback, "mlp_loss_sum"),
      correct: buffer_desc(plan.correct_bytes, storage, "mlp_correct"),
      correct_sum: buffer_desc(
        plan.correct_sum_bytes,
        readback,
        "mlp_correct_sum",
      ),
      epoch_loss: buffer_desc(plan.epoch_loss_bytes, readback, "mlp_epoch_loss"),
      epoch_correct: buffer_desc(
        plan.epoch_correct_bytes,
        readback,
        "mlp_epoch_correct",
      ),
      epoch_seen: buffer_desc(plan.epoch_seen_bytes, readback, "mlp_epoch_seen"),
    }
  })
}

///|
pub fn mlp_plan_loss_resources_for_dataset(
  spec : MlpSpec,
  dataset_count : Int,
) -> Result[MlpLossResourcePlan, MlpError] {
  mlp_plan_loss_buffers_for_dataset(spec, dataset_count).map(fn(plan) {
    let upload = buffer_usage_storage_upload()
    let readback = buffer_usage_storage_readback()
    let storage = @wgpu.buffer_usage_storage
    {
      labels: buffer_desc(plan.labels_bytes, upload, "mlp_labels"),
      loss: buffer_desc(plan.loss_bytes, readback, "mlp_loss"),
      probs: buffer_desc(plan.probs_bytes, readback, "mlp_probs"),
      loss_sum: buffer_desc(plan.loss_sum_bytes, readback, "mlp_loss_sum"),
      correct: buffer_desc(plan.correct_bytes, storage, "mlp_correct"),
      correct_sum: buffer_desc(
        plan.correct_sum_bytes,
        readback,
        "mlp_correct_sum",
      ),
      epoch_loss: buffer_desc(plan.epoch_loss_bytes, readback, "mlp_epoch_loss"),
      epoch_correct: buffer_desc(
        plan.epoch_correct_bytes,
        readback,
        "mlp_epoch_correct",
      ),
      epoch_seen: buffer_desc(plan.epoch_seen_bytes, readback, "mlp_epoch_seen"),
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
fn workgroup_tile_dim(workgroup_size : Int) -> Result[Int, MlpError] {
  if workgroup_size <= 0 {
    return Err(InvalidSpec("workgroup_size must be > 0"))
  }
  let mut dim = 1
  while dim * dim < workgroup_size {
    dim = dim + 1
  }
  if dim * dim != workgroup_size {
    Err(InvalidSpec("workgroup_size must be a perfect square for tiled matmul"))
  } else {
    Ok(dim)
  }
}

///|
fn tiles_for(rows : Int, cols : Int, tile_dim : Int) -> Int {
  if tile_dim <= 0 {
    0
  } else {
    ceil_div(rows, tile_dim) * ceil_div(cols, tile_dim)
  }
}

///|
pub fn mlp_dispatch_plan(
  spec : MlpSpec,
  workgroup_size : Int,
) -> Result[MlpDispatchPlan, MlpError] {
  if workgroup_size <= 0 {
    Err(InvalidSpec("workgroup_size must be > 0"))
  } else {
    mlp_validate_spec(spec).bind(fn(_) {
      workgroup_tile_dim(workgroup_size).map(fn(tile_dim) {
        {
          workgroup_size,
          layer1_dispatch_x: tiles_for(
            spec.batch_size,
            spec.hidden_size,
            tile_dim,
          ),
          layer2_dispatch_x: tiles_for(
            spec.batch_size,
            spec.output_size,
            tile_dim,
          ),
        }
      })
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
  grad_update_w1_dispatch_x : Int
  grad_update_b1_dispatch_x : Int
  grad_update_w2_dispatch_x : Int
  grad_update_b2_dispatch_x : Int
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
        grad_update_w1_dispatch_x: ceil_div(w1, workgroup_size),
        grad_update_b1_dispatch_x: ceil_div(b1, workgroup_size),
        grad_update_w2_dispatch_x: ceil_div(w2, workgroup_size),
        grad_update_b2_dispatch_x: ceil_div(b2, workgroup_size),
      }
    })
  }
}

///|
fn u32_lit(value : Int) -> String {
  value.to_string() + "u"
}

///|
fn wgsl_layer1(spec : MlpSpec, workgroup_size : Int, tile_dim : Int) -> String {
  let input = u32_lit(spec.input_size)
  let hidden = u32_lit(spec.hidden_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let workgroup_u = u32_lit(workgroup_size)
  let tile = u32_lit(tile_dim)
  let s0 = "// MLP layer1: input -> hidden (ReLU), tiled\n"
  let s1 = s0 + "const INPUT_SIZE : u32 = " + input + ";\n"
  let s2 = s1 + "const HIDDEN_SIZE : u32 = " + hidden + ";\n"
  let s3 = s2 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s4 = s3 + "const TILE_DIM : u32 = " + tile + ";\n"
  let s5 = s4 + "const WORKGROUP_SIZE : u32 = " + workgroup_u + ";\n"
  let s6 = s5 + "@group(0) @binding(0) var<storage, read> input : array<f32>;\n"
  let s7 = s6 +
    "@group(0) @binding(1) var<storage, read> indices : array<u32>;\n"
  let s8 = s7 +
    "@group(0) @binding(2) var<storage, read> weight1 : array<f32>;\n"
  let s9 = s8 + "@group(0) @binding(3) var<storage, read> bias1 : array<f32>;\n"
  let s10 = s9 +
    "@group(0) @binding(4) var<storage, read_write> hidden : array<f32>;\n"
  let s11 = s10 + "var<workgroup> input_tile : array<f32, WORKGROUP_SIZE>;\n"
  let s12 = s11 + "var<workgroup> weight_tile : array<f32, WORKGROUP_SIZE>;\n"
  let s13 = s12 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s14 = s13 +
    "fn main(@builtin(workgroup_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {\n"
  let s15 = s14 + "  let tile_id = gid.x;\n"
  let s16 = s15 + "  let local = lid.x;\n"
  let s17 = s16 +
    "  let tiles_per_row = (HIDDEN_SIZE + TILE_DIM - 1u) / TILE_DIM;\n"
  let s18 = s17 + "  let tile_row = tile_id / tiles_per_row;\n"
  let s19 = s18 + "  let tile_col = tile_id % tiles_per_row;\n"
  let s20 = s19 + "  let row_in_tile = local / TILE_DIM;\n"
  let s21 = s20 + "  let col_in_tile = local % TILE_DIM;\n"
  let s22 = s21 + "  let b = tile_row * TILE_DIM + row_in_tile;\n"
  let s23 = s22 + "  let j = tile_col * TILE_DIM + col_in_tile;\n"
  let s24 = s23 + "  let in_bounds = (b < BATCH_SIZE) && (j < HIDDEN_SIZE);\n"
  let s25 = s24 + "  var acc = 0.0;\n"
  let s26 = s25 + "  if (in_bounds) { acc = bias1[j]; }\n"
  let s27 = s26 + "  var kk : u32 = 0u;\n"
  let s28 = s27 + "  loop {\n"
  let s29 = s28 + "    if (kk >= INPUT_SIZE) { break; }\n"
  let s30 = s29 + "    let input_row = tile_row * TILE_DIM + row_in_tile;\n"
  let s31 = s30 + "    let input_col = kk + col_in_tile;\n"
  let s32 = s31 +
    "    if (input_row < BATCH_SIZE && input_col < INPUT_SIZE) {\n"
  let s33 = s32 + "      let sample = indices[1u + indices[0] + input_row];\n"
  let s34 = s33 +
    "      input_tile[local] = input[sample * INPUT_SIZE + input_col];\n"
  let s35 = s34 + "    } else {\n"
  let s36 = s35 + "      input_tile[local] = 0.0;\n"
  let s37 = s36 + "    }\n"
  let s38 = s37 + "    let weight_row = kk + row_in_tile;\n"
  let s39 = s38 + "    let weight_col = tile_col * TILE_DIM + col_in_tile;\n"
  let s40 = s39 +
    "    if (weight_row < INPUT_SIZE && weight_col < HIDDEN_SIZE) {\n"
  let s41 = s40 +
    "      weight_tile[local] = weight1[weight_row * HIDDEN_SIZE + weight_col];\n"
  let s42 = s41 + "    } else {\n"
  let s43 = s42 + "      weight_tile[local] = 0.0;\n"
  let s44 = s43 + "    }\n"
  let s45 = s44 + "    workgroupBarrier();\n"
  let s46 = s45 + "    for (var k : u32 = 0u; k < TILE_DIM; k = k + 1u) {\n"
  let s47 = s46 + "      let a = input_tile[row_in_tile * TILE_DIM + k];\n"
  let s48 = s47 + "      let bval = weight_tile[k * TILE_DIM + col_in_tile];\n"
  let s49 = s48 + "      acc = acc + a * bval;\n"
  let s50 = s49 + "    }\n"
  let s51 = s50 + "    workgroupBarrier();\n"
  let s52 = s51 + "    kk = kk + TILE_DIM;\n"
  let s53 = s52 + "  }\n"
  let s54 = s53 + "  if (in_bounds) {\n"
  let s55 = s54 + "    if (acc < 0.0) { acc = 0.0; }\n"
  let s56 = s55 + "    hidden[b * HIDDEN_SIZE + j] = acc;\n"
  let s57 = s56 + "  }\n"
  s57 + "}\n"
}

///|
fn wgsl_layer2(spec : MlpSpec, workgroup_size : Int, tile_dim : Int) -> String {
  let hidden = u32_lit(spec.hidden_size)
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let workgroup_u = u32_lit(workgroup_size)
  let tile = u32_lit(tile_dim)
  let s0 = "// MLP layer2: hidden -> output, tiled\n"
  let s1 = s0 + "const HIDDEN_SIZE : u32 = " + hidden + ";\n"
  let s2 = s1 + "const OUTPUT_SIZE : u32 = " + output + ";\n"
  let s3 = s2 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s4 = s3 + "const TILE_DIM : u32 = " + tile + ";\n"
  let s5 = s4 + "const WORKGROUP_SIZE : u32 = " + workgroup_u + ";\n"
  let s6 = s5 +
    "@group(0) @binding(0) var<storage, read> hidden : array<f32>;\n"
  let s7 = s6 +
    "@group(0) @binding(1) var<storage, read> weight2 : array<f32>;\n"
  let s8 = s7 + "@group(0) @binding(2) var<storage, read> bias2 : array<f32>;\n"
  let s9 = s8 +
    "@group(0) @binding(3) var<storage, read_write> output : array<f32>;\n"
  let s10 = s9 + "var<workgroup> input_tile : array<f32, WORKGROUP_SIZE>;\n"
  let s11 = s10 + "var<workgroup> weight_tile : array<f32, WORKGROUP_SIZE>;\n"
  let s12 = s11 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s13 = s12 +
    "fn main(@builtin(workgroup_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {\n"
  let s14 = s13 + "  let tile_id = gid.x;\n"
  let s15 = s14 + "  let local = lid.x;\n"
  let s16 = s15 +
    "  let tiles_per_row = (OUTPUT_SIZE + TILE_DIM - 1u) / TILE_DIM;\n"
  let s17 = s16 + "  let tile_row = tile_id / tiles_per_row;\n"
  let s18 = s17 + "  let tile_col = tile_id % tiles_per_row;\n"
  let s19 = s18 + "  let row_in_tile = local / TILE_DIM;\n"
  let s20 = s19 + "  let col_in_tile = local % TILE_DIM;\n"
  let s21 = s20 + "  let b = tile_row * TILE_DIM + row_in_tile;\n"
  let s22 = s21 + "  let k = tile_col * TILE_DIM + col_in_tile;\n"
  let s23 = s22 + "  let in_bounds = (b < BATCH_SIZE) && (k < OUTPUT_SIZE);\n"
  let s24 = s23 + "  var acc = 0.0;\n"
  let s25 = s24 + "  if (in_bounds) { acc = bias2[k]; }\n"
  let s26 = s25 + "  var kk : u32 = 0u;\n"
  let s27 = s26 + "  loop {\n"
  let s28 = s27 + "    if (kk >= HIDDEN_SIZE) { break; }\n"
  let s29 = s28 + "    let input_row = tile_row * TILE_DIM + row_in_tile;\n"
  let s30 = s29 + "    let input_col = kk + col_in_tile;\n"
  let s31 = s30 +
    "    if (input_row < BATCH_SIZE && input_col < HIDDEN_SIZE) {\n"
  let s32 = s31 +
    "      input_tile[local] = hidden[input_row * HIDDEN_SIZE + input_col];\n"
  let s33 = s32 + "    } else {\n"
  let s34 = s33 + "      input_tile[local] = 0.0;\n"
  let s35 = s34 + "    }\n"
  let s36 = s35 + "    let weight_row = kk + row_in_tile;\n"
  let s37 = s36 + "    let weight_col = tile_col * TILE_DIM + col_in_tile;\n"
  let s38 = s37 +
    "    if (weight_row < HIDDEN_SIZE && weight_col < OUTPUT_SIZE) {\n"
  let s39 = s38 +
    "      weight_tile[local] = weight2[weight_row * OUTPUT_SIZE + weight_col];\n"
  let s40 = s39 + "    } else {\n"
  let s41 = s40 + "      weight_tile[local] = 0.0;\n"
  let s42 = s41 + "    }\n"
  let s43 = s42 + "    workgroupBarrier();\n"
  let s44 = s43 + "    for (var j : u32 = 0u; j < TILE_DIM; j = j + 1u) {\n"
  let s45 = s44 + "      let a = input_tile[row_in_tile * TILE_DIM + j];\n"
  let s46 = s45 + "      let bval = weight_tile[j * TILE_DIM + col_in_tile];\n"
  let s47 = s46 + "      acc = acc + a * bval;\n"
  let s48 = s47 + "    }\n"
  let s49 = s48 + "    workgroupBarrier();\n"
  let s50 = s49 + "    kk = kk + TILE_DIM;\n"
  let s51 = s50 + "  }\n"
  let s52 = s51 + "  if (in_bounds) {\n"
  let s53 = s52 + "    output[b * OUTPUT_SIZE + k] = acc;\n"
  let s54 = s53 + "  }\n"
  s54 + "}\n"
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
    "@group(0) @binding(2) var<storage, read> indices : array<u32>;\n"
  let s7 = s6 +
    "@group(0) @binding(3) var<storage, read_write> loss : array<f32>;\n"
  let s8 = s7 +
    "@group(0) @binding(4) var<storage, read_write> probs : array<f32>;\n"
  let s9 = s8 +
    "@group(0) @binding(5) var<storage, read_write> correct : array<f32>;\n"
  let s10 = s9 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s11 = s10 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s12 = s11 + "  let b = gid.x;\n"
  let s13 = s12 + "  if (b >= BATCH_SIZE) { return; }\n"
  let s14 = s13 + "  let base = b * OUTPUT_SIZE;\n"
  let s15 = s14 + "  var max_val = logits[base];\n"
  let s16 = s15 + "  var best_k : u32 = 0u;\n"
  let s17 = s16 + "  for (var k : u32 = 1u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s18 = s17 + "    let v = logits[base + k];\n"
  let s19 = s18 + "    if (v > max_val) { max_val = v; best_k = k; }\n"
  let s20 = s19 + "  }\n"
  let s21 = s20 + "  var sum = 0.0;\n"
  let s22 = s21 + "  for (var k : u32 = 0u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s23 = s22 + "    sum = sum + exp(logits[base + k] - max_val);\n"
  let s24 = s23 + "  }\n"
  let s25 = s24 + "  for (var k : u32 = 0u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s26 = s25 +
    "    probs[base + k] = exp(logits[base + k] - max_val) / sum;\n"
  let s27 = s26 + "  }\n"
  let s28 = s27 + "  let sample = indices[1u + indices[0] + b];\n"
  let s29 = s28 + "  let label = labels[sample];\n"
  let s30 = s29 + "  let prob = probs[base + label];\n"
  let s31 = s30 + "  loss[b] = -log(prob + EPS);\n"
  let s32 = s31 + "  correct[b] = select(0.0, 1.0, best_k == label);\n"
  s32 + "}\n"
}

///|
fn wgsl_layer2_loss(spec : MlpSpec, workgroup_size : Int) -> String {
  let hidden = u32_lit(spec.hidden_size)
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP layer2 + loss (fused)\n"
  let s1 = s0 + "const HIDDEN_SIZE : u32 = " + hidden + ";\n"
  let s2 = s1 + "const OUTPUT_SIZE : u32 = " + output + ";\n"
  let s3 = s2 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s4 = s3 + "const EPS : f32 = 0.000001;\n"
  let s5 = s4 +
    "@group(0) @binding(0) var<storage, read> hidden : array<f32>;\n"
  let s6 = s5 +
    "@group(0) @binding(1) var<storage, read> weight2 : array<f32>;\n"
  let s7 = s6 + "@group(0) @binding(2) var<storage, read> bias2 : array<f32>;\n"
  let s8 = s7 +
    "@group(0) @binding(3) var<storage, read> labels : array<u32>;\n"
  let s9 = s8 +
    "@group(0) @binding(4) var<storage, read> indices : array<u32>;\n"
  let s10 = s9 +
    "@group(0) @binding(5) var<storage, read_write> loss : array<f32>;\n"
  let s11 = s10 +
    "@group(0) @binding(6) var<storage, read_write> probs : array<f32>;\n"
  let s12 = s11 +
    "@group(0) @binding(7) var<storage, read_write> correct : array<f32>;\n"
  let s13 = s12 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s14 = s13 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s15 = s14 + "  let b = gid.x;\n"
  let s16 = s15 + "  if (b >= BATCH_SIZE) { return; }\n"
  let s17 = s16 + "  var logits : array<f32, OUTPUT_SIZE>;\n"
  let s18 = s17 + "  var max_val = -1e30;\n"
  let s19 = s18 + "  var best_k : u32 = 0u;\n"
  let s20 = s19 + "  for (var k : u32 = 0u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s21 = s20 + "    var acc = bias2[k];\n"
  let s22 = s21 + "    for (var j : u32 = 0u; j < HIDDEN_SIZE; j = j + 1u) {\n"
  let s23 = s22 + "      let h = hidden[b * HIDDEN_SIZE + j];\n"
  let s24 = s23 + "      let w = weight2[j * OUTPUT_SIZE + k];\n"
  let s25 = s24 + "      acc = acc + h * w;\n"
  let s26 = s25 + "    }\n"
  let s27 = s26 + "    logits[k] = acc;\n"
  let s28 = s27 + "    if (acc > max_val) { max_val = acc; best_k = k; }\n"
  let s29 = s28 + "  }\n"
  let s30 = s29 + "  var sum = 0.0;\n"
  let s31 = s30 + "  for (var k : u32 = 0u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s32 = s31 + "    sum = sum + exp(logits[k] - max_val);\n"
  let s33 = s32 + "  }\n"
  let s34 = s33 + "  let base = b * OUTPUT_SIZE;\n"
  let s35 = s34 + "  for (var k : u32 = 0u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s36 = s35 + "    probs[base + k] = exp(logits[k] - max_val) / sum;\n"
  let s37 = s36 + "  }\n"
  let s38 = s37 + "  let sample = indices[1u + indices[0] + b];\n"
  let s39 = s38 + "  let label = labels[sample];\n"
  let s40 = s39 + "  let prob = probs[base + label];\n"
  let s41 = s40 + "  loss[b] = -log(prob + EPS);\n"
  let s42 = s41 + "  correct[b] = select(0.0, 1.0, best_k == label);\n"
  s42 + "}\n"
}

///|
fn wgsl_loss_reduce(spec : MlpSpec, workgroup_size : Int) -> String {
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let workgroup_u = u32_lit(workgroup_size)
  let s0 = "// MLP loss reduce: sum over batch\n"
  let s1 = s0 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s2 = s1 + "const WORKGROUP_SIZE : u32 = " + workgroup_u + ";\n"
  let s3 = s2 + "@group(0) @binding(0) var<storage, read> loss : array<f32>;\n"
  let s4 = s3 +
    "@group(0) @binding(1) var<storage, read_write> loss_sum : array<f32>;\n"
  let s5 = s4 + "var<workgroup> scratch : array<f32, WORKGROUP_SIZE>;\n"
  let s6 = s5 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s7 = s6 + "fn main(@builtin(local_invocation_id) lid : vec3<u32>) {\n"
  let s8 = s7 + "  let tid = lid.x;\n"
  let s9 = s8 + "  var acc = 0.0;\n"
  let s10 = s9 +
    "  for (var i : u32 = tid; i < BATCH_SIZE; i = i + WORKGROUP_SIZE) {\n"
  let s11 = s10 + "    acc = acc + loss[i];\n"
  let s12 = s11 + "  }\n"
  let s13 = s12 + "  scratch[tid] = acc;\n"
  let s14 = s13 + "  workgroupBarrier();\n"
  let s15 = s14 + "  var active_count = WORKGROUP_SIZE;\n"
  let s16 = s15 + "  loop {\n"
  let s17 = s16 + "    if (active_count <= 1u) { break; }\n"
  let s18 = s17 + "    let stride = (active_count + 1u) / 2u;\n"
  let s19 = s18 + "    if (tid < stride) {\n"
  let s20 = s19 + "      let other = tid + stride;\n"
  let s21 = s20 + "      if (other < active_count) {\n"
  let s22 = s21 + "        scratch[tid] = scratch[tid] + scratch[other];\n"
  let s23 = s22 + "      }\n"
  let s24 = s23 + "    }\n"
  let s25 = s24 + "    workgroupBarrier();\n"
  let s26 = s25 + "    active_count = stride;\n"
  let s27 = s26 + "  }\n"
  let s28 = s27 + "  if (tid == 0u) { loss_sum[0] = scratch[0]; }\n"
  s28 + "}\n"
}

///|
fn wgsl_loss_epoch_reduce(spec : MlpSpec, workgroup_size : Int) -> String {
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let workgroup_u = u32_lit(workgroup_size)
  let s0 = "// MLP loss + correct reduce + epoch accumulate\n"
  let s1 = s0 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s2 = s1 + "const WORKGROUP_SIZE : u32 = " + workgroup_u + ";\n"
  let s3 = s2 + "@group(0) @binding(0) var<storage, read> loss : array<f32>;\n"
  let s4 = s3 +
    "@group(0) @binding(1) var<storage, read> correct : array<f32>;\n"
  let s5 = s4 +
    "@group(0) @binding(2) var<storage, read_write> epoch_loss : array<f32>;\n"
  let s6 = s5 +
    "@group(0) @binding(3) var<storage, read_write> epoch_correct : array<f32>;\n"
  let s7 = s6 +
    "@group(0) @binding(4) var<storage, read_write> epoch_seen : array<f32>;\n"
  let s8 = s7 + "var<workgroup> loss_scratch : array<f32, WORKGROUP_SIZE>;\n"
  let s9 = s8 + "var<workgroup> correct_scratch : array<f32, WORKGROUP_SIZE>;\n"
  let s10 = s9 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s11 = s10 + "fn main(@builtin(local_invocation_id) lid : vec3<u32>) {\n"
  let s12 = s11 + "  let tid = lid.x;\n"
  let s13 = s12 + "  var loss_acc = 0.0;\n"
  let s14 = s13 + "  var correct_acc = 0.0;\n"
  let s15 = s14 +
    "  for (var i : u32 = tid; i < BATCH_SIZE; i = i + WORKGROUP_SIZE) {\n"
  let s16 = s15 + "    loss_acc = loss_acc + loss[i];\n"
  let s17 = s16 + "    correct_acc = correct_acc + correct[i];\n"
  let s18 = s17 + "  }\n"
  let s19 = s18 + "  loss_scratch[tid] = loss_acc;\n"
  let s20 = s19 + "  correct_scratch[tid] = correct_acc;\n"
  let s21 = s20 + "  workgroupBarrier();\n"
  let s22 = s21 + "  var active_count = WORKGROUP_SIZE;\n"
  let s23 = s22 + "  loop {\n"
  let s24 = s23 + "    if (active_count <= 1u) { break; }\n"
  let s25 = s24 + "    let stride = (active_count + 1u) / 2u;\n"
  let s26 = s25 + "    if (tid < stride) {\n"
  let s27 = s26 + "      let other = tid + stride;\n"
  let s28 = s27 + "      if (other < active_count) {\n"
  let s29 = s28 +
    "        loss_scratch[tid] = loss_scratch[tid] + loss_scratch[other];\n"
  let s30 = s29 +
    "        correct_scratch[tid] = correct_scratch[tid] + correct_scratch[other];\n"
  let s31 = s30 + "      }\n"
  let s32 = s31 + "    }\n"
  let s33 = s32 + "    workgroupBarrier();\n"
  let s34 = s33 + "    active_count = stride;\n"
  let s35 = s34 + "  }\n"
  let s36 = s35 + "  if (tid == 0u) {\n"
  let s37 = s36 + "    epoch_loss[0] = epoch_loss[0] + loss_scratch[0];\n"
  let s38 = s37 +
    "    epoch_correct[0] = epoch_correct[0] + correct_scratch[0];\n"
  let s39 = s38 + "    epoch_seen[0] = epoch_seen[0] + f32(BATCH_SIZE);\n"
  let s40 = s39 + "  }\n"
  s40 + "}\n"
}

///|
fn wgsl_grad_update_w2(spec : MlpSpec, workgroup_size : Int) -> String {
  let hidden = u32_lit(spec.hidden_size)
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP grad+update w2 (fused)\n"
  let s1 = s0 + "const HIDDEN_SIZE : u32 = " + hidden + ";\n"
  let s2 = s1 + "const OUTPUT_SIZE : u32 = " + output + ";\n"
  let s3 = s2 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s4 = s3 +
    "@group(0) @binding(0) var<storage, read> hidden : array<f32>;\n"
  let s5 = s4 + "@group(0) @binding(1) var<storage, read> probs : array<f32>;\n"
  let s6 = s5 +
    "@group(0) @binding(2) var<storage, read> labels : array<u32>;\n"
  let s7 = s6 +
    "@group(0) @binding(3) var<storage, read> indices : array<u32>;\n"
  let s8 = s7 +
    "@group(0) @binding(4) var<storage, read_write> weight2 : array<f32>;\n"
  let s9 = s8 + "@group(0) @binding(5) var<storage, read> lr : array<f32>;\n"
  let s10 = s9 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s11 = s10 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s12 = s11 + "  let index = gid.x;\n"
  let s13 = s12 + "  if (index >= HIDDEN_SIZE * OUTPUT_SIZE) { return; }\n"
  let s14 = s13 + "  let j = index / OUTPUT_SIZE;\n"
  let s15 = s14 + "  let k = index % OUTPUT_SIZE;\n"
  let s16 = s15 + "  var acc = 0.0;\n"
  let s17 = s16 + "  for (var b : u32 = 0u; b < BATCH_SIZE; b = b + 1u) {\n"
  let s18 = s17 + "    let base = b * OUTPUT_SIZE;\n"
  let s19 = s18 + "    var dlogit = probs[base + k];\n"
  let s20 = s19 + "    let sample = indices[1u + indices[0] + b];\n"
  let s21 = s20 + "    let label = labels[sample];\n"
  let s22 = s21 + "    if (label == k) { dlogit = dlogit - 1.0; }\n"
  let s23 = s22 + "    let hidden_idx = b * HIDDEN_SIZE + j;\n"
  let s24 = s23 + "    acc = acc + hidden[hidden_idx] * dlogit;\n"
  let s25 = s24 + "  }\n"
  let s26 = s25 + "  let scale = lr[0] / f32(BATCH_SIZE);\n"
  let s27 = s26 + "  weight2[index] = weight2[index] - acc * scale;\n"
  s27 + "}\n"
}

///|
fn wgsl_grad_update_b2(spec : MlpSpec, workgroup_size : Int) -> String {
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP grad+update b2 (fused)\n"
  let s1 = s0 + "const OUTPUT_SIZE : u32 = " + output + ";\n"
  let s2 = s1 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s3 = s2 + "@group(0) @binding(0) var<storage, read> probs : array<f32>;\n"
  let s4 = s3 +
    "@group(0) @binding(1) var<storage, read> labels : array<u32>;\n"
  let s5 = s4 +
    "@group(0) @binding(2) var<storage, read> indices : array<u32>;\n"
  let s6 = s5 +
    "@group(0) @binding(3) var<storage, read_write> bias2 : array<f32>;\n"
  let s7 = s6 + "@group(0) @binding(4) var<storage, read> lr : array<f32>;\n"
  let s8 = s7 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s9 = s8 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s10 = s9 + "  let k = gid.x;\n"
  let s11 = s10 + "  if (k >= OUTPUT_SIZE) { return; }\n"
  let s12 = s11 + "  var acc = 0.0;\n"
  let s13 = s12 + "  for (var b : u32 = 0u; b < BATCH_SIZE; b = b + 1u) {\n"
  let s14 = s13 + "    let base = b * OUTPUT_SIZE;\n"
  let s15 = s14 + "    var dlogit = probs[base + k];\n"
  let s16 = s15 + "    let sample = indices[1u + indices[0] + b];\n"
  let s17 = s16 + "    let label = labels[sample];\n"
  let s18 = s17 + "    if (label == k) { dlogit = dlogit - 1.0; }\n"
  let s19 = s18 + "    acc = acc + dlogit;\n"
  let s20 = s19 + "  }\n"
  let s21 = s20 + "  let scale = lr[0] / f32(BATCH_SIZE);\n"
  let s22 = s21 + "  bias2[k] = bias2[k] - acc * scale;\n"
  s22 + "}\n"
}

///|
fn wgsl_grad_update_w1(spec : MlpSpec, workgroup_size : Int) -> String {
  let input = u32_lit(spec.input_size)
  let hidden = u32_lit(spec.hidden_size)
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP grad+update w1 (fused)\n"
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
    "@group(0) @binding(4) var<storage, read> indices : array<u32>;\n"
  let s10 = s9 +
    "@group(0) @binding(5) var<storage, read> weight2 : array<f32>;\n"
  let s11 = s10 +
    "@group(0) @binding(6) var<storage, read_write> weight1 : array<f32>;\n"
  let s12 = s11 + "@group(0) @binding(7) var<storage, read> lr : array<f32>;\n"
  let s13 = s12 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s14 = s13 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s15 = s14 + "  let index = gid.x;\n"
  let s16 = s15 + "  if (index >= INPUT_SIZE * HIDDEN_SIZE) { return; }\n"
  let s17 = s16 + "  let i = index / HIDDEN_SIZE;\n"
  let s18 = s17 + "  let j = index % HIDDEN_SIZE;\n"
  let s19 = s18 + "  var acc = 0.0;\n"
  let s20 = s19 + "  for (var b : u32 = 0u; b < BATCH_SIZE; b = b + 1u) {\n"
  let s21 = s20 + "    let hidden_idx = b * HIDDEN_SIZE + j;\n"
  let s22 = s21 + "    let h = hidden[hidden_idx];\n"
  let s23 = s22 + "    if (h <= 0.0) { continue; }\n"
  let s24 = s23 + "    var sum = 0.0;\n"
  let s25 = s24 + "    let base = b * OUTPUT_SIZE;\n"
  let s26 = s25 + "    let sample = indices[1u + indices[0] + b];\n"
  let s27 = s26 + "    let label = labels[sample];\n"
  let s28 = s27 + "    for (var k : u32 = 0u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s29 = s28 + "      var dlogit = probs[base + k];\n"
  let s30 = s29 + "      if (label == k) { dlogit = dlogit - 1.0; }\n"
  let s31 = s30 + "      let w2_idx = j * OUTPUT_SIZE + k;\n"
  let s32 = s31 + "      sum = sum + dlogit * weight2[w2_idx];\n"
  let s33 = s32 + "    }\n"
  let s34 = s33 + "    let input_idx = sample * INPUT_SIZE + i;\n"
  let s35 = s34 + "    acc = acc + input[input_idx] * sum;\n"
  let s36 = s35 + "  }\n"
  let s37 = s36 + "  let scale = lr[0] / f32(BATCH_SIZE);\n"
  let s38 = s37 + "  weight1[index] = weight1[index] - acc * scale;\n"
  s38 + "}\n"
}

///|
fn wgsl_grad_update_b1(spec : MlpSpec, workgroup_size : Int) -> String {
  let hidden = u32_lit(spec.hidden_size)
  let output = u32_lit(spec.output_size)
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP grad+update b1 (fused)\n"
  let s1 = s0 + "const HIDDEN_SIZE : u32 = " + hidden + ";\n"
  let s2 = s1 + "const OUTPUT_SIZE : u32 = " + output + ";\n"
  let s3 = s2 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s4 = s3 +
    "@group(0) @binding(0) var<storage, read> hidden : array<f32>;\n"
  let s5 = s4 + "@group(0) @binding(1) var<storage, read> probs : array<f32>;\n"
  let s6 = s5 +
    "@group(0) @binding(2) var<storage, read> labels : array<u32>;\n"
  let s7 = s6 +
    "@group(0) @binding(3) var<storage, read> indices : array<u32>;\n"
  let s8 = s7 +
    "@group(0) @binding(4) var<storage, read> weight2 : array<f32>;\n"
  let s9 = s8 +
    "@group(0) @binding(5) var<storage, read_write> bias1 : array<f32>;\n"
  let s10 = s9 + "@group(0) @binding(6) var<storage, read> lr : array<f32>;\n"
  let s11 = s10 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s12 = s11 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s13 = s12 + "  let j = gid.x;\n"
  let s14 = s13 + "  if (j >= HIDDEN_SIZE) { return; }\n"
  let s15 = s14 + "  var acc = 0.0;\n"
  let s16 = s15 + "  for (var b : u32 = 0u; b < BATCH_SIZE; b = b + 1u) {\n"
  let s17 = s16 + "    let hidden_idx = b * HIDDEN_SIZE + j;\n"
  let s18 = s17 + "    let h = hidden[hidden_idx];\n"
  let s19 = s18 + "    if (h <= 0.0) { continue; }\n"
  let s20 = s19 + "    var sum = 0.0;\n"
  let s21 = s20 + "    let base = b * OUTPUT_SIZE;\n"
  let s22 = s21 + "    let sample = indices[1u + indices[0] + b];\n"
  let s23 = s22 + "    let label = labels[sample];\n"
  let s24 = s23 + "    for (var k : u32 = 0u; k < OUTPUT_SIZE; k = k + 1u) {\n"
  let s25 = s24 + "      var dlogit = probs[base + k];\n"
  let s26 = s25 + "      if (label == k) { dlogit = dlogit - 1.0; }\n"
  let s27 = s26 + "      let w2_idx = j * OUTPUT_SIZE + k;\n"
  let s28 = s27 + "      sum = sum + dlogit * weight2[w2_idx];\n"
  let s29 = s28 + "    }\n"
  let s30 = s29 + "    acc = acc + sum;\n"
  let s31 = s30 + "  }\n"
  let s32 = s31 + "  let scale = lr[0] / f32(BATCH_SIZE);\n"
  let s33 = s32 + "  bias1[j] = bias1[j] - acc * scale;\n"
  s33 + "}\n"
}

///|
fn wgsl_indices(
  spec : MlpSpec,
  dataset_count : Int,
  workgroup_size : Int,
) -> String {
  let dataset = u32_lit(dataset_count)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP indices init\n"
  let s1 = s0 + "const DATASET_COUNT : u32 = " + dataset + ";\n"
  let s2 = s1 +
    "@group(0) @binding(0) var<storage, read_write> indices : array<u32>;\n"
  let s3 = s2 +
    "@group(0) @binding(1) var<storage, read> shuffle : array<u32>;\n"
  let s4 = s3 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s5 = s4 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s6 = s5 + "  let i = gid.x;\n"
  let s7 = s6 + "  if (i >= DATASET_COUNT) { return; }\n"
  let s8 = s7 + "  let seed = shuffle[0];\n"
  let s9 = s8 + "  let stride = shuffle[1];\n"
  let s10 = s9 + "  indices[i + 1u] = (i * stride + seed) % DATASET_COUNT;\n"
  s10 + "}\n"
}

///|
fn wgsl_epoch_accum(spec : MlpSpec, workgroup_size : Int) -> String {
  let batch = u32_lit(spec.batch_size)
  let workgroup = workgroup_size.to_string()
  let s0 = "// MLP epoch accumulate\n"
  let s1 = s0 + "const BATCH_SIZE : u32 = " + batch + ";\n"
  let s2 = s1 +
    "@group(0) @binding(0) var<storage, read> loss_sum : array<f32>;\n"
  let s3 = s2 +
    "@group(0) @binding(1) var<storage, read> correct_sum : array<f32>;\n"
  let s4 = s3 +
    "@group(0) @binding(2) var<storage, read_write> epoch_loss : array<f32>;\n"
  let s5 = s4 +
    "@group(0) @binding(3) var<storage, read_write> epoch_correct : array<f32>;\n"
  let s6 = s5 +
    "@group(0) @binding(4) var<storage, read_write> epoch_seen : array<f32>;\n"
  let s7 = s6 + "@compute @workgroup_size(" + workgroup + ")\n"
  let s8 = s7 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s9 = s8 + "  if (gid.x != 0u) { return; }\n"
  let s10 = s9 + "  epoch_loss[0] = epoch_loss[0] + loss_sum[0];\n"
  let s11 = s10 + "  epoch_correct[0] = epoch_correct[0] + correct_sum[0];\n"
  let s12 = s11 + "  epoch_seen[0] = epoch_seen[0] + f32(BATCH_SIZE);\n"
  s12 + "}\n"
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
  layer2_loss_wgsl : String
  loss_reduce_wgsl : String
  loss_epoch_reduce_wgsl : String
} derive(Show, Eq)

///|
pub struct MlpDatasetShaderPlan {
  workgroup_size : Int
  indices_wgsl : String
} derive(Show, Eq)

///|
pub struct MlpTrainShaderPlan {
  workgroup_size : Int
  grad_update_w1_wgsl : String
  grad_update_b1_wgsl : String
  grad_update_w2_wgsl : String
  grad_update_b2_wgsl : String
  epoch_accum_wgsl : String
} derive(Show, Eq)

///|
pub fn mlp_shader_plan(
  spec : MlpSpec,
  workgroup_size : Int,
) -> Result[MlpShaderPlan, MlpError] {
  if workgroup_size <= 0 {
    Err(InvalidSpec("workgroup_size must be > 0"))
  } else {
    mlp_validate_spec(spec).bind(fn(_) {
      workgroup_tile_dim(workgroup_size).map(fn(tile_dim) {
        {
          workgroup_size,
          layer1_wgsl: wgsl_layer1(spec, workgroup_size, tile_dim),
          layer2_wgsl: wgsl_layer2(spec, workgroup_size, tile_dim),
        }
      })
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
        layer2_loss_wgsl: wgsl_layer2_loss(spec, workgroup_size),
        loss_reduce_wgsl: wgsl_loss_reduce(spec, workgroup_size),
        loss_epoch_reduce_wgsl: wgsl_loss_epoch_reduce(spec, workgroup_size),
      }
    })
  }
}

///|
pub fn mlp_dataset_shader_plan(
  spec : MlpSpec,
  dataset_count : Int,
  workgroup_size : Int,
) -> Result[MlpDatasetShaderPlan, MlpError] {
  if workgroup_size <= 0 {
    return Err(InvalidSpec("workgroup_size must be > 0"))
  }
  if dataset_count <= 0 {
    return Err(InvalidSpec("dataset_count must be > 0"))
  }
  mlp_validate_spec(spec).map(fn(_) {
    {
      workgroup_size,
      indices_wgsl: wgsl_indices(spec, dataset_count, workgroup_size),
    }
  })
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
        grad_update_w1_wgsl: wgsl_grad_update_w1(spec, workgroup_size),
        grad_update_b1_wgsl: wgsl_grad_update_b1(spec, workgroup_size),
        grad_update_w2_wgsl: wgsl_grad_update_w2(spec, workgroup_size),
        grad_update_b2_wgsl: wgsl_grad_update_b2(spec, workgroup_size),
        epoch_accum_wgsl: wgsl_epoch_accum(spec, workgroup_size),
      }
    })
  }
}

///|
pub fn mlp_shader_plan_default() -> MlpShaderPlan {
  let spec = { input_size: 1, hidden_size: 1, output_size: 1, batch_size: 1 }
  {
    workgroup_size: 64,
    layer1_wgsl: wgsl_layer1(spec, 64, 8),
    layer2_wgsl: wgsl_layer2(spec, 64, 8),
  }
}

///|
pub fn mlp_loss_shader_plan_default() -> MlpLossShaderPlan {
  let spec = { input_size: 1, hidden_size: 1, output_size: 1, batch_size: 1 }
  {
    workgroup_size: 64,
    loss_wgsl: wgsl_loss(spec, 64),
    layer2_loss_wgsl: wgsl_layer2_loss(spec, 64),
    loss_reduce_wgsl: wgsl_loss_reduce(spec, 64),
    loss_epoch_reduce_wgsl: wgsl_loss_epoch_reduce(spec, 64),
  }
}
