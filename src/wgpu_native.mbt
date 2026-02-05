///|
fn power_preference_to_native(pref : PowerPreference) -> Int {
  match pref {
    Default => 0
    LowPower => 1
    HighPerformance => 2
  }
}

///|
fn binding_type_to_native(ty : BindingType) -> Int {
  match ty {
    UniformBuffer => 2
    StorageBuffer => 3
    ReadOnlyStorageBuffer => 4
    _ => 0
  }
}

///|
fn is_buffer_binding(ty : BindingType) -> Bool {
  match ty {
    UniformBuffer => true
    StorageBuffer => true
    ReadOnlyStorageBuffer => true
    _ => false
  }
}

///|
priv struct NativeLayoutEntry {
  binding : Int
  visibility : Int
  binding_type : Int
} derive(Show, Eq)

///|
fn to_native_layout_entry(
  entry : BindGroupLayoutEntry,
) -> Result[NativeLayoutEntry, WgpuError] {
  if not(is_buffer_binding(entry.ty)) {
    log_native_error(
      "bind group layout entry not supported (binding=" +
      entry.binding.to_string() +
      " type=" +
      entry.ty.to_string() +
      ")",
    )
    return Err(NotSupported)
  }
  Ok({
    binding: entry.binding,
    visibility: entry.visibility,
    binding_type: binding_type_to_native(entry.ty),
  })
}

///|
priv struct NativeBindEntry {
  binding : Int
  buffer : Buffer
  offset : Int
  size : Int
}

///|
fn to_native_bind_entry(
  entry : BindGroupEntry,
) -> Result[NativeBindEntry, WgpuError] {
  match entry.resource {
    Buffer(buf, offset, size) =>
      Ok({ binding: entry.binding, buffer: buf, offset, size })
    _ => {
      log_native_error(
        "bind group entry not supported (binding=" +
        entry.binding.to_string() +
        ")",
      )
      Err(NotSupported)
    }
  }
}

///|
fn limits_is_default(limits : Limits) -> Bool {
  limits.max_texture_dimension_2d == 0 &&
  limits.max_storage_buffer_binding_size == 0 &&
  limits.max_uniform_buffer_binding_size == 0 &&
  limits.max_bind_groups == 0 &&
  limits.max_compute_workgroup_size_x == 0 &&
  limits.max_compute_workgroup_size_y == 0 &&
  limits.max_compute_workgroup_size_z == 0 &&
  limits.max_compute_workgroups_per_dimension == 0
}

///|
#borrow(instance)
extern "C" fn native_ptr_is_null_instance(instance : Instance) -> Bool = "wgpu_native_ptr_is_null"

///|
#borrow(adapter)
extern "C" fn native_ptr_is_null_adapter(adapter : Adapter) -> Bool = "wgpu_native_ptr_is_null"

///|
#borrow(device)
extern "C" fn native_ptr_is_null_device(device : Device) -> Bool = "wgpu_native_ptr_is_null"

///|
#borrow(queue)
extern "C" fn native_ptr_is_null_queue(queue : Queue) -> Bool = "wgpu_native_ptr_is_null"

///|
#borrow(buffer)
extern "C" fn native_ptr_is_null_buffer(buffer : Buffer) -> Bool = "wgpu_native_ptr_is_null"

///|
#borrow(shader_module)
extern "C" fn native_ptr_is_null_shader_module(
  shader_module : ShaderModule,
) -> Bool = "wgpu_native_ptr_is_null"

///|
#borrow(layout)
extern "C" fn native_ptr_is_null_bind_group_layout(
  layout : BindGroupLayout,
) -> Bool = "wgpu_native_ptr_is_null"

///|
#borrow(group)
extern "C" fn native_ptr_is_null_bind_group(group : BindGroup) -> Bool = "wgpu_native_ptr_is_null"

///|
#borrow(layout)
extern "C" fn native_ptr_is_null_pipeline_layout(
  layout : PipelineLayout,
) -> Bool = "wgpu_native_ptr_is_null"

///|
#borrow(pipeline)
extern "C" fn native_ptr_is_null_compute_pipeline(
  pipeline : ComputePipeline,
) -> Bool = "wgpu_native_ptr_is_null"

///|
extern "C" fn native_instance_new() -> Instance = "wgpu_native_instance_new"

///|
extern "C" fn native_request_adapter(
  instance : Instance,
  power_preference : Int,
  force_fallback : Bool,
) -> Adapter = "wgpu_native_request_adapter"

///|
extern "C" fn native_request_device(adapter : Adapter) -> Device = "wgpu_native_request_device"

///|
extern "C" fn native_device_get_queue(device : Device) -> Queue = "wgpu_native_device_get_queue"

///|
extern "C" fn native_device_create_buffer(
  device : Device,
  size : Int,
  usage : Int,
  mapped_at_creation : Bool,
) -> Buffer = "wgpu_native_device_create_buffer"

///|
#borrow(code)
extern "C" fn native_device_create_shader_module(
  device : Device,
  code : Bytes,
  len : Int,
) -> ShaderModule = "wgpu_native_device_create_shader_module"

///|
extern "C" fn native_device_create_bind_group_layout_1(
  device : Device,
  binding0 : Int,
  visibility0 : Int,
  type0 : Int,
) -> BindGroupLayout = "wgpu_native_device_create_bind_group_layout_1"

///|
extern "C" fn native_device_create_bind_group_layout_2(
  device : Device,
  binding0 : Int,
  visibility0 : Int,
  type0 : Int,
  binding1 : Int,
  visibility1 : Int,
  type1 : Int,
) -> BindGroupLayout = "wgpu_native_device_create_bind_group_layout_2"

///|
extern "C" fn native_device_create_bind_group_layout_3(
  device : Device,
  binding0 : Int,
  visibility0 : Int,
  type0 : Int,
  binding1 : Int,
  visibility1 : Int,
  type1 : Int,
  binding2 : Int,
  visibility2 : Int,
  type2 : Int,
) -> BindGroupLayout = "wgpu_native_device_create_bind_group_layout_3"

///|
extern "C" fn native_device_create_bind_group_layout_4(
  device : Device,
  binding0 : Int,
  visibility0 : Int,
  type0 : Int,
  binding1 : Int,
  visibility1 : Int,
  type1 : Int,
  binding2 : Int,
  visibility2 : Int,
  type2 : Int,
  binding3 : Int,
  visibility3 : Int,
  type3 : Int,
) -> BindGroupLayout = "wgpu_native_device_create_bind_group_layout_4"

///|
extern "C" fn native_device_create_bind_group_layout_5(
  device : Device,
  binding0 : Int,
  visibility0 : Int,
  type0 : Int,
  binding1 : Int,
  visibility1 : Int,
  type1 : Int,
  binding2 : Int,
  visibility2 : Int,
  type2 : Int,
  binding3 : Int,
  visibility3 : Int,
  type3 : Int,
  binding4 : Int,
  visibility4 : Int,
  type4 : Int,
) -> BindGroupLayout = "wgpu_native_device_create_bind_group_layout_5"

///|
extern "C" fn native_device_create_bind_group_layout_6(
  device : Device,
  binding0 : Int,
  visibility0 : Int,
  type0 : Int,
  binding1 : Int,
  visibility1 : Int,
  type1 : Int,
  binding2 : Int,
  visibility2 : Int,
  type2 : Int,
  binding3 : Int,
  visibility3 : Int,
  type3 : Int,
  binding4 : Int,
  visibility4 : Int,
  type4 : Int,
  binding5 : Int,
  visibility5 : Int,
  type5 : Int,
) -> BindGroupLayout = "wgpu_native_device_create_bind_group_layout_6"

///|
extern "C" fn native_device_create_bind_group_layout_7(
  device : Device,
  binding0 : Int,
  visibility0 : Int,
  type0 : Int,
  binding1 : Int,
  visibility1 : Int,
  type1 : Int,
  binding2 : Int,
  visibility2 : Int,
  type2 : Int,
  binding3 : Int,
  visibility3 : Int,
  type3 : Int,
  binding4 : Int,
  visibility4 : Int,
  type4 : Int,
  binding5 : Int,
  visibility5 : Int,
  type5 : Int,
  binding6 : Int,
  visibility6 : Int,
  type6 : Int,
) -> BindGroupLayout = "wgpu_native_device_create_bind_group_layout_7"

///|
extern "C" fn native_device_create_bind_group_layout_8(
  device : Device,
  binding0 : Int,
  visibility0 : Int,
  type0 : Int,
  binding1 : Int,
  visibility1 : Int,
  type1 : Int,
  binding2 : Int,
  visibility2 : Int,
  type2 : Int,
  binding3 : Int,
  visibility3 : Int,
  type3 : Int,
  binding4 : Int,
  visibility4 : Int,
  type4 : Int,
  binding5 : Int,
  visibility5 : Int,
  type5 : Int,
  binding6 : Int,
  visibility6 : Int,
  type6 : Int,
  binding7 : Int,
  visibility7 : Int,
  type7 : Int,
) -> BindGroupLayout = "wgpu_native_device_create_bind_group_layout_8"

///|
extern "C" fn native_device_create_bind_group_1(
  device : Device,
  layout : BindGroupLayout,
  binding0 : Int,
  buffer0 : Buffer,
  offset0 : Int,
  size0 : Int,
) -> BindGroup = "wgpu_native_device_create_bind_group_1"

///|
extern "C" fn native_device_create_bind_group_2(
  device : Device,
  layout : BindGroupLayout,
  binding0 : Int,
  buffer0 : Buffer,
  offset0 : Int,
  size0 : Int,
  binding1 : Int,
  buffer1 : Buffer,
  offset1 : Int,
  size1 : Int,
) -> BindGroup = "wgpu_native_device_create_bind_group_2"

///|
extern "C" fn native_device_create_bind_group_3(
  device : Device,
  layout : BindGroupLayout,
  binding0 : Int,
  buffer0 : Buffer,
  offset0 : Int,
  size0 : Int,
  binding1 : Int,
  buffer1 : Buffer,
  offset1 : Int,
  size1 : Int,
  binding2 : Int,
  buffer2 : Buffer,
  offset2 : Int,
  size2 : Int,
) -> BindGroup = "wgpu_native_device_create_bind_group_3"

///|
extern "C" fn native_device_create_bind_group_4(
  device : Device,
  layout : BindGroupLayout,
  binding0 : Int,
  buffer0 : Buffer,
  offset0 : Int,
  size0 : Int,
  binding1 : Int,
  buffer1 : Buffer,
  offset1 : Int,
  size1 : Int,
  binding2 : Int,
  buffer2 : Buffer,
  offset2 : Int,
  size2 : Int,
  binding3 : Int,
  buffer3 : Buffer,
  offset3 : Int,
  size3 : Int,
) -> BindGroup = "wgpu_native_device_create_bind_group_4"

///|
extern "C" fn native_device_create_bind_group_5(
  device : Device,
  layout : BindGroupLayout,
  binding0 : Int,
  buffer0 : Buffer,
  offset0 : Int,
  size0 : Int,
  binding1 : Int,
  buffer1 : Buffer,
  offset1 : Int,
  size1 : Int,
  binding2 : Int,
  buffer2 : Buffer,
  offset2 : Int,
  size2 : Int,
  binding3 : Int,
  buffer3 : Buffer,
  offset3 : Int,
  size3 : Int,
  binding4 : Int,
  buffer4 : Buffer,
  offset4 : Int,
  size4 : Int,
) -> BindGroup = "wgpu_native_device_create_bind_group_5"

///|
extern "C" fn native_device_create_bind_group_6(
  device : Device,
  layout : BindGroupLayout,
  binding0 : Int,
  buffer0 : Buffer,
  offset0 : Int,
  size0 : Int,
  binding1 : Int,
  buffer1 : Buffer,
  offset1 : Int,
  size1 : Int,
  binding2 : Int,
  buffer2 : Buffer,
  offset2 : Int,
  size2 : Int,
  binding3 : Int,
  buffer3 : Buffer,
  offset3 : Int,
  size3 : Int,
  binding4 : Int,
  buffer4 : Buffer,
  offset4 : Int,
  size4 : Int,
  binding5 : Int,
  buffer5 : Buffer,
  offset5 : Int,
  size5 : Int,
) -> BindGroup = "wgpu_native_device_create_bind_group_6"

///|
extern "C" fn native_device_create_bind_group_7(
  device : Device,
  layout : BindGroupLayout,
  binding0 : Int,
  buffer0 : Buffer,
  offset0 : Int,
  size0 : Int,
  binding1 : Int,
  buffer1 : Buffer,
  offset1 : Int,
  size1 : Int,
  binding2 : Int,
  buffer2 : Buffer,
  offset2 : Int,
  size2 : Int,
  binding3 : Int,
  buffer3 : Buffer,
  offset3 : Int,
  size3 : Int,
  binding4 : Int,
  buffer4 : Buffer,
  offset4 : Int,
  size4 : Int,
  binding5 : Int,
  buffer5 : Buffer,
  offset5 : Int,
  size5 : Int,
  binding6 : Int,
  buffer6 : Buffer,
  offset6 : Int,
  size6 : Int,
) -> BindGroup = "wgpu_native_device_create_bind_group_7"

///|
extern "C" fn native_device_create_bind_group_8(
  device : Device,
  layout : BindGroupLayout,
  binding0 : Int,
  buffer0 : Buffer,
  offset0 : Int,
  size0 : Int,
  binding1 : Int,
  buffer1 : Buffer,
  offset1 : Int,
  size1 : Int,
  binding2 : Int,
  buffer2 : Buffer,
  offset2 : Int,
  size2 : Int,
  binding3 : Int,
  buffer3 : Buffer,
  offset3 : Int,
  size3 : Int,
  binding4 : Int,
  buffer4 : Buffer,
  offset4 : Int,
  size4 : Int,
  binding5 : Int,
  buffer5 : Buffer,
  offset5 : Int,
  size5 : Int,
  binding6 : Int,
  buffer6 : Buffer,
  offset6 : Int,
  size6 : Int,
  binding7 : Int,
  buffer7 : Buffer,
  offset7 : Int,
  size7 : Int,
) -> BindGroup = "wgpu_native_device_create_bind_group_8"

///|
extern "C" fn native_device_create_pipeline_layout(
  device : Device,
  layout : BindGroupLayout,
) -> PipelineLayout = "wgpu_native_device_create_pipeline_layout"

///|
#borrow(entry_point)
extern "C" fn native_device_create_compute_pipeline(
  device : Device,
  layout : PipelineLayout,
  shader_module : ShaderModule,
  entry_point : Bytes,
  len : Int,
) -> ComputePipeline = "wgpu_native_device_create_compute_pipeline"

///|
extern "C" fn native_device_dispatch_compute(
  device : Device,
  pipeline : ComputePipeline,
  bind_group : BindGroup,
  dispatch_x : Int,
) -> Unit = "wgpu_native_device_dispatch_compute"

///|
#borrow(data)
extern "C" fn native_queue_write_buffer(
  queue : Queue,
  buffer : Buffer,
  offset : Int,
  data : Bytes,
  len : Int,
) -> Unit = "wgpu_native_queue_write_buffer"

///|
extern "C" fn native_device_read_buffer_bytes(
  device : Device,
  buffer : Buffer,
  size : Int,
) -> Bytes = "wgpu_native_device_read_buffer_bytes"

///|
pub fn is_supported() -> Bool {
  true
}

///|
fn log_native_error(message : String) -> Unit {
  println("wgpu_native: " + message)
}

///|
pub async fn request_adapter(
  options : RequestAdapterOptions,
) -> Result[Adapter, WgpuError] {
  let instance = native_instance_new()
  if native_ptr_is_null_instance(instance) {
    log_native_error("native_instance_new returned null")
    return Err(NotSupported)
  }
  let adapter = native_request_adapter(
    instance,
    power_preference_to_native(options.power_preference),
    options.force_fallback_adapter,
  )
  if native_ptr_is_null_adapter(adapter) {
    log_native_error(
      "native_request_adapter returned null (power_preference=" +
      options.power_preference.to_string() +
      " fallback=" +
      options.force_fallback_adapter.to_string() +
      ")",
    )
    Err(NotSupported)
  } else {
    Ok(adapter)
  }
}

///|
pub async fn request_device(
  adapter : Adapter,
  desc : DeviceDescriptor,
) -> Result[Device, WgpuError] {
  if desc.required_features.length() > 0 ||
    not(limits_is_default(desc.required_limits)) {
    log_native_error("request_device rejected non-default features/limits")
    return Err(NotSupported)
  }
  let device = native_request_device(adapter)
  if native_ptr_is_null_device(device) {
    log_native_error("native_request_device returned null")
    Err(NotSupported)
  } else {
    Ok(device)
  }
}

///|
pub fn device_get_queue(device : Device) -> Result[Queue, WgpuError] {
  let queue = native_device_get_queue(device)
  if native_ptr_is_null_queue(queue) {
    log_native_error("native_device_get_queue returned null")
    Err(NotSupported)
  } else {
    Ok(queue)
  }
}

///|
pub fn device_create_buffer(
  device : Device,
  desc : BufferDescriptor,
) -> Result[Buffer, WgpuError] {
  let buffer = native_device_create_buffer(
    device,
    desc.size,
    desc.usage,
    desc.mapped_at_creation,
  )
  if native_ptr_is_null_buffer(buffer) {
    Err(Validation("buffer creation failed"))
  } else {
    Ok(buffer)
  }
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
  device : Device,
  desc : ShaderModuleDescriptor,
) -> Result[ShaderModule, WgpuError] {
  let code_bytes = @utf8.encode(desc.code)
  let shader = native_device_create_shader_module(
    device,
    code_bytes,
    code_bytes.length(),
  )
  if native_ptr_is_null_shader_module(shader) {
    Err(Validation("shader module creation failed"))
  } else {
    Ok(shader)
  }
}

///|
pub fn device_create_bind_group_layout(
  device : Device,
  desc : BindGroupLayoutDescriptor,
) -> Result[BindGroupLayout, WgpuError] {
  let entries = desc.entries
  let count = entries.length()
  if count <= 0 || count > 8 {
    log_native_error(
      "bind group layout entry count not supported: " + count.to_string(),
    )
    return Err(NotSupported)
  }
  to_native_layout_entry(entries[0]).bind(fn(e0) {
    let layout_result : Result[BindGroupLayout, WgpuError] = match count {
      1 =>
        Ok(
          native_device_create_bind_group_layout_1(
            device,
            e0.binding,
            e0.visibility,
            e0.binding_type,
          ),
        )
      2 =>
        to_native_layout_entry(entries[1]).map(fn(e1) {
          native_device_create_bind_group_layout_2(
            device,
            e0.binding,
            e0.visibility,
            e0.binding_type,
            e1.binding,
            e1.visibility,
            e1.binding_type,
          )
        })
      3 =>
        to_native_layout_entry(entries[1]).bind(fn(e1) {
          to_native_layout_entry(entries[2]).map(fn(e2) {
            native_device_create_bind_group_layout_3(
              device,
              e0.binding,
              e0.visibility,
              e0.binding_type,
              e1.binding,
              e1.visibility,
              e1.binding_type,
              e2.binding,
              e2.visibility,
              e2.binding_type,
            )
          })
        })
      4 =>
        to_native_layout_entry(entries[1]).bind(fn(e1) {
          to_native_layout_entry(entries[2]).bind(fn(e2) {
            to_native_layout_entry(entries[3]).map(fn(e3) {
              native_device_create_bind_group_layout_4(
                device,
                e0.binding,
                e0.visibility,
                e0.binding_type,
                e1.binding,
                e1.visibility,
                e1.binding_type,
                e2.binding,
                e2.visibility,
                e2.binding_type,
                e3.binding,
                e3.visibility,
                e3.binding_type,
              )
            })
          })
        })
      5 =>
        to_native_layout_entry(entries[1]).bind(fn(e1) {
          to_native_layout_entry(entries[2]).bind(fn(e2) {
            to_native_layout_entry(entries[3]).bind(fn(e3) {
              to_native_layout_entry(entries[4]).map(fn(e4) {
                native_device_create_bind_group_layout_5(
                  device,
                  e0.binding,
                  e0.visibility,
                  e0.binding_type,
                  e1.binding,
                  e1.visibility,
                  e1.binding_type,
                  e2.binding,
                  e2.visibility,
                  e2.binding_type,
                  e3.binding,
                  e3.visibility,
                  e3.binding_type,
                  e4.binding,
                  e4.visibility,
                  e4.binding_type,
                )
              })
            })
          })
        })
      6 =>
        to_native_layout_entry(entries[1]).bind(fn(e1) {
          to_native_layout_entry(entries[2]).bind(fn(e2) {
            to_native_layout_entry(entries[3]).bind(fn(e3) {
              to_native_layout_entry(entries[4]).bind(fn(e4) {
                to_native_layout_entry(entries[5]).map(fn(e5) {
                  native_device_create_bind_group_layout_6(
                    device,
                    e0.binding,
                    e0.visibility,
                    e0.binding_type,
                    e1.binding,
                    e1.visibility,
                    e1.binding_type,
                    e2.binding,
                    e2.visibility,
                    e2.binding_type,
                    e3.binding,
                    e3.visibility,
                    e3.binding_type,
                    e4.binding,
                    e4.visibility,
                    e4.binding_type,
                    e5.binding,
                    e5.visibility,
                    e5.binding_type,
                  )
                })
              })
            })
          })
        })
      7 =>
        to_native_layout_entry(entries[1]).bind(fn(e1) {
          to_native_layout_entry(entries[2]).bind(fn(e2) {
            to_native_layout_entry(entries[3]).bind(fn(e3) {
              to_native_layout_entry(entries[4]).bind(fn(e4) {
                to_native_layout_entry(entries[5]).bind(fn(e5) {
                  to_native_layout_entry(entries[6]).map(fn(e6) {
                    native_device_create_bind_group_layout_7(
                      device,
                      e0.binding,
                      e0.visibility,
                      e0.binding_type,
                      e1.binding,
                      e1.visibility,
                      e1.binding_type,
                      e2.binding,
                      e2.visibility,
                      e2.binding_type,
                      e3.binding,
                      e3.visibility,
                      e3.binding_type,
                      e4.binding,
                      e4.visibility,
                      e4.binding_type,
                      e5.binding,
                      e5.visibility,
                      e5.binding_type,
                      e6.binding,
                      e6.visibility,
                      e6.binding_type,
                    )
                  })
                })
              })
            })
          })
        })
      _ =>
        to_native_layout_entry(entries[1]).bind(fn(e1) {
          to_native_layout_entry(entries[2]).bind(fn(e2) {
            to_native_layout_entry(entries[3]).bind(fn(e3) {
              to_native_layout_entry(entries[4]).bind(fn(e4) {
                to_native_layout_entry(entries[5]).bind(fn(e5) {
                  to_native_layout_entry(entries[6]).bind(fn(e6) {
                    to_native_layout_entry(entries[7]).map(fn(e7) {
                      native_device_create_bind_group_layout_8(
                        device,
                        e0.binding,
                        e0.visibility,
                        e0.binding_type,
                        e1.binding,
                        e1.visibility,
                        e1.binding_type,
                        e2.binding,
                        e2.visibility,
                        e2.binding_type,
                        e3.binding,
                        e3.visibility,
                        e3.binding_type,
                        e4.binding,
                        e4.visibility,
                        e4.binding_type,
                        e5.binding,
                        e5.visibility,
                        e5.binding_type,
                        e6.binding,
                        e6.visibility,
                        e6.binding_type,
                        e7.binding,
                        e7.visibility,
                        e7.binding_type,
                      )
                    })
                  })
                })
              })
            })
          })
        })
    }
    layout_result.bind(fn(layout) {
      if native_ptr_is_null_bind_group_layout(layout) {
        Err(Validation("bind group layout creation failed"))
      } else {
        Ok(layout)
      }
    })
  })
}

///|
pub fn device_create_bind_group(
  device : Device,
  desc : BindGroupDescriptor,
) -> Result[BindGroup, WgpuError] {
  let entries = desc.entries
  let count = entries.length()
  if count <= 0 || count > 8 {
    log_native_error(
      "bind group entry count not supported: " + count.to_string(),
    )
    return Err(NotSupported)
  }
  to_native_bind_entry(entries[0]).bind(fn(e0) {
    let group_result : Result[BindGroup, WgpuError] = match count {
      1 =>
        Ok(
          native_device_create_bind_group_1(
            device,
            desc.layout,
            e0.binding,
            e0.buffer,
            e0.offset,
            e0.size,
          ),
        )
      2 =>
        to_native_bind_entry(entries[1]).map(fn(e1) {
          native_device_create_bind_group_2(
            device,
            desc.layout,
            e0.binding,
            e0.buffer,
            e0.offset,
            e0.size,
            e1.binding,
            e1.buffer,
            e1.offset,
            e1.size,
          )
        })
      3 =>
        to_native_bind_entry(entries[1]).bind(fn(e1) {
          to_native_bind_entry(entries[2]).map(fn(e2) {
            native_device_create_bind_group_3(
              device,
              desc.layout,
              e0.binding,
              e0.buffer,
              e0.offset,
              e0.size,
              e1.binding,
              e1.buffer,
              e1.offset,
              e1.size,
              e2.binding,
              e2.buffer,
              e2.offset,
              e2.size,
            )
          })
        })
      4 =>
        to_native_bind_entry(entries[1]).bind(fn(e1) {
          to_native_bind_entry(entries[2]).bind(fn(e2) {
            to_native_bind_entry(entries[3]).map(fn(e3) {
              native_device_create_bind_group_4(
                device,
                desc.layout,
                e0.binding,
                e0.buffer,
                e0.offset,
                e0.size,
                e1.binding,
                e1.buffer,
                e1.offset,
                e1.size,
                e2.binding,
                e2.buffer,
                e2.offset,
                e2.size,
                e3.binding,
                e3.buffer,
                e3.offset,
                e3.size,
              )
            })
          })
        })
      5 =>
        to_native_bind_entry(entries[1]).bind(fn(e1) {
          to_native_bind_entry(entries[2]).bind(fn(e2) {
            to_native_bind_entry(entries[3]).bind(fn(e3) {
              to_native_bind_entry(entries[4]).map(fn(e4) {
                native_device_create_bind_group_5(
                  device,
                  desc.layout,
                  e0.binding,
                  e0.buffer,
                  e0.offset,
                  e0.size,
                  e1.binding,
                  e1.buffer,
                  e1.offset,
                  e1.size,
                  e2.binding,
                  e2.buffer,
                  e2.offset,
                  e2.size,
                  e3.binding,
                  e3.buffer,
                  e3.offset,
                  e3.size,
                  e4.binding,
                  e4.buffer,
                  e4.offset,
                  e4.size,
                )
              })
            })
          })
        })
      6 =>
        to_native_bind_entry(entries[1]).bind(fn(e1) {
          to_native_bind_entry(entries[2]).bind(fn(e2) {
            to_native_bind_entry(entries[3]).bind(fn(e3) {
              to_native_bind_entry(entries[4]).bind(fn(e4) {
                to_native_bind_entry(entries[5]).map(fn(e5) {
                  native_device_create_bind_group_6(
                    device,
                    desc.layout,
                    e0.binding,
                    e0.buffer,
                    e0.offset,
                    e0.size,
                    e1.binding,
                    e1.buffer,
                    e1.offset,
                    e1.size,
                    e2.binding,
                    e2.buffer,
                    e2.offset,
                    e2.size,
                    e3.binding,
                    e3.buffer,
                    e3.offset,
                    e3.size,
                    e4.binding,
                    e4.buffer,
                    e4.offset,
                    e4.size,
                    e5.binding,
                    e5.buffer,
                    e5.offset,
                    e5.size,
                  )
                })
              })
            })
          })
        })
      7 =>
        to_native_bind_entry(entries[1]).bind(fn(e1) {
          to_native_bind_entry(entries[2]).bind(fn(e2) {
            to_native_bind_entry(entries[3]).bind(fn(e3) {
              to_native_bind_entry(entries[4]).bind(fn(e4) {
                to_native_bind_entry(entries[5]).bind(fn(e5) {
                  to_native_bind_entry(entries[6]).map(fn(e6) {
                    native_device_create_bind_group_7(
                      device,
                      desc.layout,
                      e0.binding,
                      e0.buffer,
                      e0.offset,
                      e0.size,
                      e1.binding,
                      e1.buffer,
                      e1.offset,
                      e1.size,
                      e2.binding,
                      e2.buffer,
                      e2.offset,
                      e2.size,
                      e3.binding,
                      e3.buffer,
                      e3.offset,
                      e3.size,
                      e4.binding,
                      e4.buffer,
                      e4.offset,
                      e4.size,
                      e5.binding,
                      e5.buffer,
                      e5.offset,
                      e5.size,
                      e6.binding,
                      e6.buffer,
                      e6.offset,
                      e6.size,
                    )
                  })
                })
              })
            })
          })
        })
      _ =>
        to_native_bind_entry(entries[1]).bind(fn(e1) {
          to_native_bind_entry(entries[2]).bind(fn(e2) {
            to_native_bind_entry(entries[3]).bind(fn(e3) {
              to_native_bind_entry(entries[4]).bind(fn(e4) {
                to_native_bind_entry(entries[5]).bind(fn(e5) {
                  to_native_bind_entry(entries[6]).bind(fn(e6) {
                    to_native_bind_entry(entries[7]).map(fn(e7) {
                      native_device_create_bind_group_8(
                        device,
                        desc.layout,
                        e0.binding,
                        e0.buffer,
                        e0.offset,
                        e0.size,
                        e1.binding,
                        e1.buffer,
                        e1.offset,
                        e1.size,
                        e2.binding,
                        e2.buffer,
                        e2.offset,
                        e2.size,
                        e3.binding,
                        e3.buffer,
                        e3.offset,
                        e3.size,
                        e4.binding,
                        e4.buffer,
                        e4.offset,
                        e4.size,
                        e5.binding,
                        e5.buffer,
                        e5.offset,
                        e5.size,
                        e6.binding,
                        e6.buffer,
                        e6.offset,
                        e6.size,
                        e7.binding,
                        e7.buffer,
                        e7.offset,
                        e7.size,
                      )
                    })
                  })
                })
              })
            })
          })
        })
    }
    group_result.bind(fn(group) {
      if native_ptr_is_null_bind_group(group) {
        Err(Validation("bind group creation failed"))
      } else {
        Ok(group)
      }
    })
  })
}

///|
pub fn device_create_pipeline_layout(
  device : Device,
  desc : PipelineLayoutDescriptor,
) -> Result[PipelineLayout, WgpuError] {
  if desc.bind_group_layouts.length() != 1 {
    return Err(NotSupported)
  }
  let layout = native_device_create_pipeline_layout(
    device,
    desc.bind_group_layouts[0],
  )
  if native_ptr_is_null_pipeline_layout(layout) {
    Err(Validation("pipeline layout creation failed"))
  } else {
    Ok(layout)
  }
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
  device : Device,
  desc : ComputePipelineDescriptor,
) -> Result[ComputePipeline, WgpuError] {
  let entry_bytes = @utf8.encode(desc.compute.entry_point)
  let pipeline = native_device_create_compute_pipeline(
    device,
    desc.layout,
    desc.compute.shader_module,
    entry_bytes,
    entry_bytes.length(),
  )
  if native_ptr_is_null_compute_pipeline(pipeline) {
    Err(Validation("compute pipeline creation failed"))
  } else {
    Ok(pipeline)
  }
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
  queue : Queue,
  buffer : Buffer,
  offset : Int,
  data : Array[Byte],
) -> Result[Unit, WgpuError] {
  let bytes = Bytes::from_array(data)
  native_queue_write_buffer(queue, buffer, offset, bytes, bytes.length())
  Ok(())
}

///|
pub fn device_dispatch_compute(
  device : Device,
  pipeline : ComputePipeline,
  bind_group : BindGroup,
  dispatch_x : Int,
) -> Result[Unit, WgpuError] {
  if dispatch_x <= 0 {
    return Err(Validation("dispatch_x must be > 0"))
  }
  native_device_dispatch_compute(device, pipeline, bind_group, dispatch_x)
  Ok(())
}

///|
pub fn device_read_buffer_bytes(
  device : Device,
  buffer : Buffer,
  size : Int,
) -> Result[Bytes, WgpuError] {
  if size < 0 {
    return Err(Validation("size must be >= 0"))
  }
  let bytes = native_device_read_buffer_bytes(device, buffer, size)
  if bytes.length() != size {
    Err(Validation("readback size mismatch"))
  } else {
    Ok(bytes)
  }
}
