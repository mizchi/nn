///|
using @nn { type MnistError }

///|
pub struct MnistDataset {
  images : @nn.MnistImages
  labels : @nn.MnistLabels
} derive(Show, Eq)

///|
pub async fn mnist_load_images(path : String) -> Result[@nn.MnistImages, MnistError] {
  let data_result = try? @afs.read_file(path)
  match data_result {
    Ok(data) => {
      let bytes = data.binary()
      @nn.mnist_read_images(bytes)
    }
    Err(err) => Err(@nn.mnist_error_io(err.to_string()))
  }
}

///|
pub async fn mnist_load_labels(path : String) -> Result[@nn.MnistLabels, MnistError] {
  let data_result = try? @afs.read_file(path)
  match data_result {
    Ok(data) => {
      let bytes = data.binary()
      @nn.mnist_read_labels(bytes)
    }
    Err(err) => Err(@nn.mnist_error_io(err.to_string()))
  }
}

///|
pub async fn mnist_load_dataset(
  images_path : String,
  labels_path : String
) -> Result[MnistDataset, MnistError] {
  let images = mnist_load_images(images_path)
  match images {
    Ok(img) =>
      match mnist_load_labels(labels_path) {
        Ok(lbl) => Ok({ images: img, labels: lbl })
        Err(err) => Err(err)
      }
    Err(err) => Err(err)
  }
}

///|
pub async fn mnist_load_mlp_dataset(
  images_path : String,
  labels_path : String,
  limit : Int?
) -> Result[@nn.MlpDataset, MnistError] {
  let dataset = mnist_load_dataset(images_path, labels_path)
  match dataset {
    Ok(ds) => @nn.mnist_to_mlp_dataset(ds.images, ds.labels, limit)
    Err(err) => Err(err)
  }
}
