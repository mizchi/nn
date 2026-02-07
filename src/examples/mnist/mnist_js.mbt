///|
using @nn {type MnistError}

///|
pub struct MnistDataset {
  images : @nn.MnistImages
  labels : @nn.MnistLabels
} derive(Show, Eq)

///|
extern "js" fn fetch_bytes_js(url : String) -> @js_async.Promise[Bytes] =
  #| async (url) => {
  #|   const res = await fetch(url)
  #|   if (!res.ok) {
  #|     throw new Error(`fetch failed: ${res.status}`)
  #|   }
  #|   const buf = await res.arrayBuffer()
  #|   return new Uint8Array(buf)
  #| }

///|
pub async fn mnist_load_images(
  _path : String,
) -> Result[@nn.MnistImages, MnistError] {
  let data_result = try? fetch_bytes_js(_path).wait()
  match data_result {
    Ok(bytes) => @nn.mnist_read_images(bytes)
    Err(err) => Err(@nn.mnist_error_io(err.to_string()))
  }
}

///|
pub async fn mnist_load_labels(
  _path : String,
) -> Result[@nn.MnistLabels, MnistError] {
  let data_result = try? fetch_bytes_js(_path).wait()
  match data_result {
    Ok(bytes) => @nn.mnist_read_labels(bytes)
    Err(err) => Err(@nn.mnist_error_io(err.to_string()))
  }
}

///|
pub async fn mnist_load_dataset(
  _images_path : String,
  _labels_path : String,
) -> Result[MnistDataset, MnistError] {
  let images = mnist_load_images(_images_path)
  match images {
    Ok(img) =>
      match mnist_load_labels(_labels_path) {
        Ok(lbl) => Ok({ images: img, labels: lbl })
        Err(err) => Err(err)
      }
    Err(err) => Err(err)
  }
}

///|
pub async fn mnist_load_mlp_dataset(
  _images_path : String,
  _labels_path : String,
  _limit : Int?,
) -> Result[@nn.MlpDataset, MnistError] {
  let dataset = mnist_load_dataset(_images_path, _labels_path)
  match dataset {
    Ok(ds) => @nn.mnist_to_mlp_dataset(ds.images, ds.labels, _limit)
    Err(err) => Err(err)
  }
}
