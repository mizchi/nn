///|
test "surface_configuration_default" {
  let config = surface_configuration_default(@wgpu.texture_format_rgba8_unorm, 640, 480)
  inspect(config.width, content="640")
  inspect(config.height, content="480")
  inspect(config.usage, content="16")
}
