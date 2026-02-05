///|
///|
pub fn queue_write_bytes(
  queue : @wgpu.Queue,
  buffer : @wgpu.Buffer,
  offset : Int,
  data : Bytes
) -> Result[Unit, @wgpu.WgpuError] {
  @wgpu.queue_write_buffer(queue, buffer, offset, data.to_array())
}

///|
extern "js" fn read_buffer_bytes_js(
  device : @wgpu.Device,
  buffer : @wgpu.Buffer,
  size : Int
) -> @js_async.Promise[Bytes] =
  #| async (device, buffer, size) => {
  #|   if (size <= 0) return new Uint8Array(0)
  #|   const readBuffer = device.createBuffer({
  #|     size,
  #|     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  #|   })
  #|   const encoder = device.createCommandEncoder()
  #|   encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size)
  #|   device.queue.submit([encoder.finish()])
  #|   await readBuffer.mapAsync(GPUMapMode.READ)
  #|   const mapped = readBuffer.getMappedRange()
  #|   const copy = new Uint8Array(mapped).slice()
  #|   readBuffer.unmap()
  #|   if (readBuffer.destroy) readBuffer.destroy()
  #|   return copy
  #| }

///|
pub async fn device_read_buffer_bytes(
  device : @wgpu.Device,
  buffer : @wgpu.Buffer,
  size : Int
) -> Result[Bytes, @wgpu.WgpuError] {
  let bytes = read_buffer_bytes_js(device, buffer, size).wait()
  Ok(bytes)
}
