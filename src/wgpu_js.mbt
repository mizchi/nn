///|
// JS backend (stub)
pub fn is_supported() -> Bool {
  false
}

///|
pub fn device_poll(_device : Device, _wait : Bool) -> Unit {
  ()
}

///|
pub fn dispatch_batch_begin() -> Unit {
  ()
}

///|
pub fn dispatch_batch_add(
  _pipeline : ComputePipeline,
  _bind_group : BindGroup,
  _dispatch_x : Int,
) -> Unit {
  ()
}

///|
pub fn dispatch_batch_submit(_device : Device) -> Unit {
  ()
}

///|
pub async fn request_adapter(
  _options : RequestAdapterOptions,
) -> Result[Adapter, WgpuError] {
  Err(NotImplemented)
}

///|
pub async fn request_device(
  _adapter : Adapter,
  _desc : DeviceDescriptor,
) -> Result[Device, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_get_queue(_device : Device) -> Result[Queue, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_create_buffer(
  _device : Device,
  _desc : BufferDescriptor,
) -> Result[Buffer, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_create_texture(
  _device : Device,
  _desc : TextureDescriptor,
) -> Result[Texture, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn texture_create_view(
  _texture : Texture,
) -> Result[TextureView, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_create_sampler(
  _device : Device,
  _desc : SamplerDescriptor,
) -> Result[Sampler, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_create_shader_module(
  _device : Device,
  _desc : ShaderModuleDescriptor,
) -> Result[ShaderModule, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_create_bind_group_layout(
  _device : Device,
  _desc : BindGroupLayoutDescriptor,
) -> Result[BindGroupLayout, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_create_bind_group(
  _device : Device,
  _desc : BindGroupDescriptor,
) -> Result[BindGroup, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_create_pipeline_layout(
  _device : Device,
  _desc : PipelineLayoutDescriptor,
) -> Result[PipelineLayout, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_create_render_pipeline(
  _device : Device,
  _desc : RenderPipelineDescriptor,
) -> Result[RenderPipeline, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_create_compute_pipeline(
  _device : Device,
  _desc : ComputePipelineDescriptor,
) -> Result[ComputePipeline, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_dispatch_compute(
  _device : Device,
  _pipeline : ComputePipeline,
  _bind_group : BindGroup,
  _dispatch_x : Int,
) -> Result[Unit, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_create_command_encoder(
  _device : Device,
  _desc : CommandEncoderDescriptor,
) -> Result[CommandEncoder, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn command_encoder_finish(
  _encoder : CommandEncoder,
) -> Result[CommandBuffer, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn queue_submit(
  _queue : Queue,
  _commands : Array[CommandBuffer],
) -> Result[Unit, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn queue_write_buffer(
  _queue : Queue,
  _buffer : Buffer,
  _offset : Int,
  _data : Array[Byte],
) -> Result[Unit, WgpuError] {
  Err(NotImplemented)
}

///|
pub fn device_read_buffer_bytes(
  _device : Device,
  _buffer : Buffer,
  _size : Int,
) -> Result[Bytes, WgpuError] {
  Err(NotImplemented)
}
