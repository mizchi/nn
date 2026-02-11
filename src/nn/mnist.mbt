///|
pub enum MnistError {
  InvalidMagic(String)
  InvalidLength(String)
  InvalidLabel(String)
  Io(String)
  InvalidDataset(String)
} derive(Show, Eq)

///|
pub fn mnist_error_invalid_magic(message : String) -> MnistError {
  InvalidMagic(message)
}

///|
pub fn mnist_error_invalid_length(message : String) -> MnistError {
  InvalidLength(message)
}

///|
pub fn mnist_error_invalid_label(message : String) -> MnistError {
  InvalidLabel(message)
}

///|
pub fn mnist_error_io(message : String) -> MnistError {
  Io(message)
}

///|
pub fn mnist_error_invalid_dataset(message : String) -> MnistError {
  InvalidDataset(message)
}

///|
pub struct MnistImages {
  count : Int
  rows : Int
  cols : Int
  data : Bytes
} derive(Show, Eq)

///|
pub struct MnistLabels {
  count : Int
  data : Bytes
} derive(Show, Eq)

///|
fn read_u32_be(bytes : Bytes, offset : Int) -> UInt {
  for j = 0, acc = UInt::default()
      j < 4
      j = j + 1, acc = acc | (bytes[offset + j].to_uint() << ((3 - j) * 8)) {

  } nobreak {
    acc
  }
}

///|
fn expect_min_length(
  name : String,
  actual : Int,
  expected : Int,
) -> Result[Unit, MnistError] {
  if actual >= expected {
    Ok(())
  } else {
    Err(InvalidLength(name + " length too small"))
  }
}

///|
fn expect_exact_length(
  name : String,
  actual : Int,
  expected : Int,
) -> Result[Unit, MnistError] {
  if actual == expected {
    Ok(())
  } else {
    Err(InvalidLength(name + " length mismatch"))
  }
}

///|
pub fn mnist_read_images(bytes : Bytes) -> Result[MnistImages, MnistError] {
  expect_min_length("images", bytes.length(), 16)
  .bind(fn(_) {
    let magic = read_u32_be(bytes, 0).reinterpret_as_int()
    if magic != 2051 {
      Err(InvalidMagic("images magic mismatch"))
    } else {
      Ok(())
    }
  })
  .bind(fn(_) {
    let count = read_u32_be(bytes, 4).reinterpret_as_int()
    let rows = read_u32_be(bytes, 8).reinterpret_as_int()
    let cols = read_u32_be(bytes, 12).reinterpret_as_int()
    if count <= 0 || rows <= 0 || cols <= 0 {
      Err(InvalidLength("images header invalid"))
    } else {
      let total = count * rows * cols
      let expected = 16 + total
      expect_exact_length("images", bytes.length(), expected).map(fn(_) {
        let data = bytes[16:expected].to_bytes()
        { count, rows, cols, data }
      })
    }
  })
}

///|
pub fn mnist_read_labels(bytes : Bytes) -> Result[MnistLabels, MnistError] {
  expect_min_length("labels", bytes.length(), 8)
  .bind(fn(_) {
    let magic = read_u32_be(bytes, 0).reinterpret_as_int()
    if magic != 2049 {
      Err(InvalidMagic("labels magic mismatch"))
    } else {
      Ok(())
    }
  })
  .bind(fn(_) {
    let count = read_u32_be(bytes, 4).reinterpret_as_int()
    if count <= 0 {
      Err(InvalidLength("labels header invalid"))
    } else {
      let expected = 8 + count
      expect_exact_length("labels", bytes.length(), expected).map(fn(_) {
        let data = bytes[8:expected].to_bytes()
        { count, data }
      })
    }
  })
}

///|
pub fn mnist_image_size(images : MnistImages) -> Int {
  images.rows * images.cols
}

///|
pub fn mnist_images_to_floats(
  images : MnistImages,
  limit : Int?,
) -> Array[Float] {
  let count = match limit {
    Some(v) => if v <= images.count { v } else { images.count }
    None => images.count
  }
  let image_size = mnist_image_size(images)
  let total = count * image_size
  let denom = Float::from_int(255)
  Array::makei(total, fn(i) {
    let v = images.data[i].to_int()
    Float::from_int(v) / denom
  })
}

///|
pub fn mnist_labels_to_array(
  labels : MnistLabels,
  limit : Int?,
) -> Result[Array[Int], MnistError] {
  let count = match limit {
    Some(v) => if v <= labels.count { v } else { labels.count }
    None => labels.count
  }
  let out : Array[Int] = Array::new(capacity=count)
  for i = 0; i < count; i = i + 1 {
    let v = labels.data[i].to_int()
    if v < 0 || v > 9 {
      return Err(InvalidLabel("label out of range"))
    }
    out.push(v)
  }
  Ok(out)
}

///|
pub fn mnist_to_mlp_dataset(
  images : MnistImages,
  labels : MnistLabels,
  limit : Int?,
) -> Result[MlpDataset, MnistError] {
  let inputs = mnist_images_to_floats(images, limit)
  match mnist_labels_to_array(labels, limit) {
    Ok(labels_arr) => {
      let input_size = mnist_image_size(images)
      match mlp_dataset_new(input_size, inputs, labels_arr) {
        Ok(ds) => Ok(ds)
        Err(err) => Err(InvalidDataset(err.to_string()))
      }
    }
    Err(err) => Err(err)
  }
}
