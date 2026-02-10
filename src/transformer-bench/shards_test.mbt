///|
fn u16_bytes(values : Array[Int]) -> Bytes {
  let out : Array[Byte] = []
  for i = 0; i < values.length(); i = i + 1 {
    let v = values[i]
    out.push((v % 256).to_byte())
    out.push((v / 256).to_byte())
  }
  Bytes::from_array(out)
}

///|
fn ascii_bytes(text : String) -> Bytes {
  let out : Array[Byte] = []
  for c in text {
    out.push(c.to_int().to_byte())
  }
  Bytes::from_array(out)
}

///|
test "load_token_shards_meta_and_ids_uint16" {
  let dir = "/tmp/nn_openwebtext_shards_" + @env.now().to_string()
  let meta_path = dir + "/meta.json"
  let mut err = ""
  let mut ids : Array[Int] = []
  @async.run_async_main(() => {
    let mkdir_result = try? @afs.mkdir(dir, permission=0o755, recursive=true)
    match mkdir_result {
      Ok(_) => ()
      Err(e) => {
        err = "mkdir: " + e.to_string()
        return
      }
    }
    let w0 = try? @afs.write_file(
      dir + "/shard-000000.bin",
      u16_bytes([1, 2, 3]),
      create=0o644,
      truncate=true,
    )
    match w0 {
      Ok(_) => ()
      Err(e) => {
        err = "write shard0: " + e.to_string()
        return
      }
    }
    let w1 = try? @afs.write_file(
      dir + "/shard-000001.bin",
      u16_bytes([4, 5]),
      create=0o644,
      truncate=true,
    )
    match w1 {
      Ok(_) => ()
      Err(e) => {
        err = "write shard1: " + e.to_string()
        return
      }
    }
    let meta = "{\"dtype\":\"uint16\",\"n_vocab\":50257,\"total_tokens\":5,\"shards\":[{\"file\":\"shard-000000.bin\",\"tokens\":3},{\"file\":\"shard-000001.bin\",\"tokens\":2}]}"
    let wm = try? @afs.write_file(
      meta_path,
      ascii_bytes(meta),
      create=0o644,
      truncate=true,
    )
    match wm {
      Ok(_) => ()
      Err(e) => {
        err = "write meta: " + e.to_string()
        return
      }
    }
    let meta_loaded = match load_token_shards_meta(meta_path) {
      Ok(v) => v
      Err(msg) => {
        err = "load meta: " + msg
        return
      }
    }
    match load_token_ids_from_shards(meta_loaded, None, None) {
      Ok(v) => ids = v
      Err(msg) => err = "load ids: " + msg
    }
  })
  inspect(err, content="")
  inspect(ids, content="[1, 2, 3, 4, 5]")
}

///|
test "load_token_ids_from_shards_honors_limits" {
  let dir = "/tmp/nn_openwebtext_shards_limit_" + @env.now().to_string()
  let meta_path = dir + "/meta.json"
  let mut err = ""
  let mut ids : Array[Int] = []
  @async.run_async_main(() => {
    let _ = try? @afs.mkdir(dir, permission=0o755, recursive=true)
    let _ = try? @afs.write_file(
      dir + "/shard-000000.bin",
      u16_bytes([10, 11, 12]),
      create=0o644,
      truncate=true,
    )
    let _ = try? @afs.write_file(
      dir + "/shard-000001.bin",
      u16_bytes([13, 14, 15]),
      create=0o644,
      truncate=true,
    )
    let meta = "{\"dtype\":\"uint16\",\"n_vocab\":50257,\"total_tokens\":6,\"shards\":[{\"file\":\"shard-000000.bin\",\"tokens\":3},{\"file\":\"shard-000001.bin\",\"tokens\":3}]}"
    let _ = try? @afs.write_file(
      meta_path,
      ascii_bytes(meta),
      create=0o644,
      truncate=true,
    )
    let meta_loaded = match load_token_shards_meta(meta_path) {
      Ok(v) => v
      Err(msg) => {
        err = "load meta: " + msg
        return
      }
    }
    match load_token_ids_from_shards(meta_loaded, Some(1), Some(2)) {
      Ok(v) => ids = v
      Err(msg) => err = "load ids: " + msg
    }
  })
  inspect(err, content="")
  inspect(ids, content="[10, 11]")
}
