///|
#external
pub type Window

///|
pub struct WindowSize {
  width : Int
  height : Int
} derive(Show, Eq)

///|
pub enum GlfwError {
  NotImplemented
  Invalid(String)
} derive(Show, Eq)

///|
pub fn glfw_error_not_implemented() -> GlfwError {
  NotImplemented
}

///|
pub fn glfw_error_invalid(message : String) -> GlfwError {
  Invalid(message)
}

///|
pub fn glfw_init() -> Result[Unit, GlfwError] {
  Err(NotImplemented)
}

///|
pub fn terminate() -> Unit {
  ()
}

///|
pub fn create_window(
  _width : Int,
  _height : Int,
  _title : String,
) -> Result[Window, GlfwError] {
  Err(NotImplemented)
}

///|
pub fn destroy_window(_window : Window) -> Unit {
  ()
}

///|
pub fn poll_events() -> Unit {
  ()
}

///|
pub fn window_should_close(_window : Window) -> Bool {
  true
}

///|
pub fn set_window_should_close(_window : Window, _value : Bool) -> Unit {
  ()
}

///|
pub fn swap_buffers(_window : Window) -> Unit {
  ()
}

///|
pub fn get_window_size(_window : Window) -> WindowSize {
  { width: 0, height: 0 }
}

///|
pub fn get_window_width(_window : Window) -> Int {
  0
}

///|
pub fn get_window_height(_window : Window) -> Int {
  0
}
