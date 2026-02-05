///|
fn push_u32_be(buf : Array[Byte], value : Int) -> Unit {
  buf.push(((value >> 24) & 0xff).to_byte())
  buf.push(((value >> 16) & 0xff).to_byte())
  buf.push(((value >> 8) & 0xff).to_byte())
  buf.push((value & 0xff).to_byte())
}

///|
test "mnist_parse_images" {
  let buf : Array[Byte] = []
  push_u32_be(buf, 2051)
  push_u32_be(buf, 2)
  push_u32_be(buf, 2)
  push_u32_be(buf, 1)
  buf.push(Int::to_byte(1))
  buf.push(Int::to_byte(2))
  buf.push(Int::to_byte(3))
  buf.push(Int::to_byte(4))
  let bytes = Bytes::from_array(buf)
  match mnist_read_images(bytes) {
    Ok(images) => {
      inspect(images.count, content="2")
      inspect(images.rows, content="2")
      inspect(images.cols, content="1")
      inspect(images.data.length(), content="4")
      inspect(images.data[3].to_int(), content="4")
    }
    Err(_) => inspect(false, content="true")
  }
}

///|
test "mnist_parse_labels" {
  let buf : Array[Byte] = []
  push_u32_be(buf, 2049)
  push_u32_be(buf, 2)
  buf.push(Int::to_byte(3))
  buf.push(Int::to_byte(7))
  let bytes = Bytes::from_array(buf)
  match mnist_read_labels(bytes) {
    Ok(labels) => {
      inspect(labels.count, content="2")
      inspect(labels.data.length(), content="2")
      let arr = mnist_labels_to_array(labels, None).unwrap()
      inspect(arr[0], content="3")
      inspect(arr[1], content="7")
    }
    Err(_) => inspect(false, content="true")
  }
}
