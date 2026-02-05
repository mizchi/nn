///|
using @nn { type MnistError }

///|
pub struct MnistDataset {
  images : @nn.MnistImages
  labels : @nn.MnistLabels
} derive(Show, Eq)

fn not_supported() -> MnistError {
  @nn.mnist_error_io("mnist loader is native-only")
}

async fn noop_async() -> Unit {
  ignore(@async.with_timeout_opt(0, fn() { () }))
}

///|
pub async fn mnist_load_images(_path : String) -> Result[@nn.MnistImages, MnistError] {
  noop_async()
  Err(not_supported())
}

///|
pub async fn mnist_load_labels(_path : String) -> Result[@nn.MnistLabels, MnistError] {
  noop_async()
  Err(not_supported())
}

///|
pub async fn mnist_load_dataset(
  _images_path : String,
  _labels_path : String
) -> Result[MnistDataset, MnistError] {
  noop_async()
  Err(not_supported())
}

///|
pub async fn mnist_load_mlp_dataset(
  _images_path : String,
  _labels_path : String,
  _limit : Int?
) -> Result[@nn.MlpDataset, MnistError] {
  noop_async()
  Err(not_supported())
}
