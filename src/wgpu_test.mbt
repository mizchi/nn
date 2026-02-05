///|
test "default_adapter_options" {
  let opts = @wgpu.request_adapter_options_default()
  inspect(opts.power_preference, content="Default")
  inspect(opts.force_fallback_adapter, content="false")
}

///|
test "default_device_descriptor" {
  let desc = @wgpu.device_descriptor_default()
  inspect(desc.required_features.length(), content="0")
  inspect(desc.required_limits.max_bind_groups, content="0")
}

///|
test "buffer_usage_flags" {
  let usage = @wgpu.buffer_usage_or(
    @wgpu.buffer_usage_copy_dst,
    @wgpu.buffer_usage_storage,
  )
  inspect(usage, content="136")
}

///|
test "shader_stage_flags" {
  let stages = @wgpu.shader_stage_or(
    @wgpu.shader_stage_vertex,
    @wgpu.shader_stage_fragment,
  )
  inspect(stages, content="3")
}

///|
test "enum_helpers" {
  inspect(@wgpu.power_preference_low_power, content="LowPower")
  inspect(@wgpu.texture_dimension_2d, content="D2")
  inspect(@wgpu.filter_mode_linear, content="Linear")
  inspect(@wgpu.binding_type_uniform_buffer, content="UniformBuffer")
  inspect(@wgpu.primitive_topology_triangle_list, content="TriangleList")
  inspect(@wgpu.feature_other("x"), content="Other(\"x\")")
  inspect(@wgpu.error_validation("bad"), content="Validation(\"bad\")")
}
