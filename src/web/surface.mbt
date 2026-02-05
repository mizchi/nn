///|
using @wgpu {
  type Device,
  type Texture,
  type TextureFormat,
  type TextureUsage,
  type WgpuError,
}

///|
#external
pub type Surface

///|
pub struct SurfaceConfiguration {
  format : TextureFormat
  width : Int
  height : Int
  usage : TextureUsage
  label : Option[String]
} derive(Show, Eq)

///|
pub fn surface_configuration_default(
  format : TextureFormat,
  width : Int,
  height : Int
) -> SurfaceConfiguration {
  {
    format: format,
    width: width,
    height: height,
    usage: @wgpu.texture_usage_render_attachment,
    label: None,
  }
}

///|
pub fn create_surface_from_canvas_id(
  _canvas_id : String
) -> Result[Surface, WgpuError] {
  Err(@wgpu.error_not_implemented())
}

///|
pub fn configure_surface(
  _surface : Surface,
  _device : Device,
  _config : SurfaceConfiguration
) -> Result[Unit, WgpuError] {
  Err(@wgpu.error_not_implemented())
}

///|
pub fn get_current_texture(_surface : Surface) -> Result[Texture, WgpuError] {
  Err(@wgpu.error_not_implemented())
}

///|
pub fn present(_surface : Surface) -> Result[Unit, WgpuError] {
  Ok(())
}
