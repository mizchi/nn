///|
// Core opaque handles
#external
pub type Instance

///|
#external
pub type Adapter

///|
#external
pub type Device

///|
#external
pub type Queue

///|
#external
pub type Buffer

///|
#external
pub type Texture

///|
#external
pub type TextureView

///|
#external
pub type Sampler

///|
#external
pub type BindGroupLayout

///|
#external
pub type BindGroup

///|
#external
pub type PipelineLayout

///|
#external
pub type ShaderModule

///|
#external
pub type RenderPipeline

///|
#external
pub type ComputePipeline

///|
#external
pub type CommandEncoder

///|
#external
pub type CommandBuffer

///|
// Errors
pub enum WgpuError {
  NotImplemented
  NotSupported
  Validation(String)
} derive(Show, Eq)

///|
pub fn error_not_implemented() -> WgpuError {
  NotImplemented
}

///|
pub fn error_not_supported() -> WgpuError {
  NotSupported
}

///|
pub fn error_validation(message : String) -> WgpuError {
  Validation(message)
}

///|
// Adapter selection
pub enum PowerPreference {
  Default
  LowPower
  HighPerformance
} derive(Show, Eq)

///|
pub let power_preference_default : PowerPreference = Default

///|
pub let power_preference_low_power : PowerPreference = LowPower

///|
pub let power_preference_high_performance : PowerPreference = HighPerformance

///|
pub struct RequestAdapterOptions {
  power_preference : PowerPreference
  force_fallback_adapter : Bool
} derive(Show, Eq)

///|
pub fn request_adapter_options_default() -> RequestAdapterOptions {
  { power_preference: Default, force_fallback_adapter: false }
}

///|
// Feature set (minimal subset)
pub enum Feature {
  ShaderF16
  TimestampQuery
  TextureCompressionBC
  TextureCompressionETC2
  TextureCompressionASTC
  Float32Filterable
  Other(String)
} derive(Show, Eq)

///|

///|
pub fn feature_other(name : String) -> Feature {
  Other(name)
}

///|
// Limits (minimal subset)
pub struct Limits {
  max_texture_dimension_2d : Int
  max_storage_buffer_binding_size : Int
  max_uniform_buffer_binding_size : Int
  max_bind_groups : Int
  max_compute_workgroup_size_x : Int
  max_compute_workgroup_size_y : Int
  max_compute_workgroup_size_z : Int
  max_compute_workgroups_per_dimension : Int
} derive(Show, Eq)

///|
pub fn limits_default() -> Limits {
  {
    max_texture_dimension_2d: 0,
    max_storage_buffer_binding_size: 0,
    max_uniform_buffer_binding_size: 0,
    max_bind_groups: 0,
    max_compute_workgroup_size_x: 0,
    max_compute_workgroup_size_y: 0,
    max_compute_workgroup_size_z: 0,
    max_compute_workgroups_per_dimension: 0,
  }
}

///|
pub struct DeviceDescriptor {
  required_features : Array[Feature]
  required_limits : Limits
  label : String?
} derive(Show, Eq)

///|
pub fn device_descriptor_default() -> DeviceDescriptor {
  { required_features: [], required_limits: limits_default(), label: None }
}

///|
// Buffer usage flags
pub type BufferUsage = Int

///|
pub let buffer_usage_map_read : BufferUsage = 0x0001

///|
pub let buffer_usage_map_write : BufferUsage = 0x0002

///|
pub let buffer_usage_copy_src : BufferUsage = 0x0004

///|
pub let buffer_usage_copy_dst : BufferUsage = 0x0008

///|
pub let buffer_usage_index : BufferUsage = 0x0010

///|
pub let buffer_usage_vertex : BufferUsage = 0x0020

///|
pub let buffer_usage_uniform : BufferUsage = 0x0040

///|
pub let buffer_usage_storage : BufferUsage = 0x0080

///|
pub let buffer_usage_indirect : BufferUsage = 0x0100

///|
pub let buffer_usage_query_resolve : BufferUsage = 0x0200

///|
pub fn buffer_usage_or(a : BufferUsage, b : BufferUsage) -> BufferUsage {
  a | b
}

///|
// Shader stage flags
pub type ShaderStageFlags = Int

///|
pub let shader_stage_vertex : ShaderStageFlags = 0x0001

///|
pub let shader_stage_fragment : ShaderStageFlags = 0x0002

///|
pub let shader_stage_compute : ShaderStageFlags = 0x0004

///|
pub fn shader_stage_or(
  a : ShaderStageFlags,
  b : ShaderStageFlags,
) -> ShaderStageFlags {
  a | b
}

///|
pub struct BufferDescriptor {
  size : Int
  usage : BufferUsage
  mapped_at_creation : Bool
  label : String?
} derive(Show, Eq)

///|
pub fn buffer_descriptor(
  size : Int,
  usage : BufferUsage,
  mapped_at_creation : Bool,
  label : String?,
) -> BufferDescriptor {
  { size, usage, mapped_at_creation, label }
}

///|
// Texture usage flags
pub type TextureUsage = Int

///|
pub let texture_usage_copy_src : TextureUsage = 0x0001

///|
pub let texture_usage_copy_dst : TextureUsage = 0x0002

///|
pub let texture_usage_texture_binding : TextureUsage = 0x0004

///|
pub let texture_usage_storage_binding : TextureUsage = 0x0008

///|
pub let texture_usage_render_attachment : TextureUsage = 0x0010

///|
pub fn texture_usage_or(a : TextureUsage, b : TextureUsage) -> TextureUsage {
  a | b
}

///|
pub enum TextureDimension {
  D1
  D2
  D3
} derive(Show, Eq)

///|
pub let texture_dimension_2d : TextureDimension = D2

///|
pub enum TextureFormat {
  Rgba8Unorm
  Bgra8Unorm
  Rgba16Float
  R32Float
  R32Uint
} derive(Show, Eq)

///|
pub let texture_format_rgba8_unorm : TextureFormat = Rgba8Unorm

///|
pub let texture_format_bgra8_unorm : TextureFormat = Bgra8Unorm

///|
pub struct Extent3d {
  width : Int
  height : Int
  depth_or_array_layers : Int
} derive(Show, Eq)

///|
pub fn extent_2d(width : Int, height : Int) -> Extent3d {
  { width, height, depth_or_array_layers: 1 }
}

///|
pub struct TextureDescriptor {
  size : Extent3d
  mip_level_count : Int
  sample_count : Int
  dimension : TextureDimension
  format : TextureFormat
  usage : TextureUsage
  label : String?
} derive(Show, Eq)

///|
pub enum FilterMode {
  Nearest
  Linear
} derive(Show, Eq)

///|
pub let filter_mode_nearest : FilterMode = Nearest

///|
pub let filter_mode_linear : FilterMode = Linear

///|
pub struct SamplerDescriptor {
  mag_filter : FilterMode
  min_filter : FilterMode
  mipmap_filter : FilterMode
  label : String?
} derive(Show, Eq)

///|
pub struct ShaderModuleDescriptor {
  code : String
  label : String?
} derive(Show, Eq)

///|
pub fn shader_module_descriptor(
  code : String,
  label : String?,
) -> ShaderModuleDescriptor {
  { code, label }
}

///|
// Bind group layouts
pub enum BindingType {
  UniformBuffer
  StorageBuffer
  ReadOnlyStorageBuffer
  Sampler
  Texture
  StorageTexture
} derive(Show, Eq)

///|
pub let binding_type_uniform_buffer : BindingType = UniformBuffer

///|
pub let binding_type_storage_buffer : BindingType = StorageBuffer

///|
pub let binding_type_read_only_storage_buffer : BindingType = ReadOnlyStorageBuffer

///|
pub let binding_type_sampler : BindingType = Sampler

///|
pub let binding_type_texture : BindingType = Texture

///|
pub struct BindGroupLayoutEntry {
  binding : Int
  visibility : ShaderStageFlags
  ty : BindingType
} derive(Show, Eq)

///|
pub fn bind_group_layout_entry(
  binding : Int,
  visibility : ShaderStageFlags,
  ty : BindingType,
) -> BindGroupLayoutEntry {
  { binding, visibility, ty }
}

///|
pub struct BindGroupLayoutDescriptor {
  entries : Array[BindGroupLayoutEntry]
  label : String?
} derive(Show, Eq)

///|
pub fn bind_group_layout_descriptor(
  entries : Array[BindGroupLayoutEntry],
  label : String?,
) -> BindGroupLayoutDescriptor {
  { entries, label }
}

///|
pub enum BindingResource {
  Buffer(Buffer, Int, Int)
  Sampler(Sampler)
  TextureView(TextureView)
}

///|
pub fn binding_resource_buffer(
  buffer : Buffer,
  offset : Int,
  size : Int,
) -> BindingResource {
  Buffer(buffer, offset, size)
}

///|
pub fn binding_resource_sampler(sampler : Sampler) -> BindingResource {
  Sampler(sampler)
}

///|
pub fn binding_resource_texture_view(view : TextureView) -> BindingResource {
  TextureView(view)
}

///|
pub struct BindGroupEntry {
  binding : Int
  resource : BindingResource
}

///|
pub fn bind_group_entry(
  binding : Int,
  resource : BindingResource,
) -> BindGroupEntry {
  { binding, resource }
}

///|
pub struct BindGroupDescriptor {
  layout : BindGroupLayout
  entries : Array[BindGroupEntry]
  label : String?
}

///|
pub fn bind_group_descriptor(
  layout : BindGroupLayout,
  entries : Array[BindGroupEntry],
  label : String?,
) -> BindGroupDescriptor {
  { layout, entries, label }
}

///|
// Pipeline layouts
pub struct PipelineLayoutDescriptor {
  bind_group_layouts : Array[BindGroupLayout]
  label : String?
}

///|
pub fn pipeline_layout_descriptor(
  bind_group_layouts : Array[BindGroupLayout],
  label : String?,
) -> PipelineLayoutDescriptor {
  { bind_group_layouts, label }
}

///|
// Render pipeline (minimal subset)
pub enum PrimitiveTopology {
  PointList
  LineList
  LineStrip
  TriangleList
  TriangleStrip
} derive(Show, Eq)

///|
pub let primitive_topology_triangle_list : PrimitiveTopology = TriangleList

///|
pub struct VertexState {
  shader_module : ShaderModule
  entry_point : String
}

///|
pub struct ColorTargetState {
  format : TextureFormat
} derive(Show, Eq)

///|
pub struct FragmentState {
  shader_module : ShaderModule
  entry_point : String
  targets : Array[ColorTargetState]
}

///|
pub struct PrimitiveState {
  topology : PrimitiveTopology
} derive(Show, Eq)

///|
pub struct RenderPipelineDescriptor {
  layout : PipelineLayout
  vertex : VertexState
  fragment : FragmentState
  primitive : PrimitiveState
  label : String?
}

///|
// Compute pipeline
pub struct ProgrammableStage {
  shader_module : ShaderModule
  entry_point : String
}

///|
pub fn programmable_stage(
  shader_module : ShaderModule,
  entry_point : String,
) -> ProgrammableStage {
  { shader_module, entry_point }
}

///|
pub struct ComputePipelineDescriptor {
  layout : PipelineLayout
  compute : ProgrammableStage
  label : String?
}

///|
pub fn compute_pipeline_descriptor(
  layout : PipelineLayout,
  compute : ProgrammableStage,
  label : String?,
) -> ComputePipelineDescriptor {
  { layout, compute, label }
}

///|
// Command encoding
pub struct CommandEncoderDescriptor {
  label : String?
} derive(Show, Eq)

///|
// Backend entry points
