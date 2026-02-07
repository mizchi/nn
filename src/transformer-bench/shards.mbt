///|
pub struct LmTokenShard {
  file : String
  tokens : Int
} derive(Show, Eq)

///|
pub struct LmTokenShardsMeta {
  meta_path : String
  base_dir : String
  dtype : String
  n_vocab : Int
  total_tokens : Int
  shards : Array[LmTokenShard]
} derive(Show, Eq)

///|
fn dirname(path : String) -> String {
  match path.rev_find("/") {
    Some(i) => if i == 0 { "/" } else { path.unsafe_substring(start=0, end=i) }
    None => "."
  }
}

///|
fn path_join(base : String, name : String) -> String {
  if base.has_suffix("/") {
    base + name
  } else {
    base + "/" + name
  }
}

///|
fn json_get(obj : Map[String, Json], key : String) -> Result[Json, String] {
  match obj.get(key) {
    Some(v) => Ok(v)
    None => Err("missing key: " + key)
  }
}

///|
fn json_get_string(
  obj : Map[String, Json],
  key : String,
) -> Result[String, String] {
  match json_get(obj, key) {
    Ok(v) => {
      let r = try? @json.from_json(v)
      match r {
        Ok(s) => Ok(s)
        Err(err) => Err("invalid string at key=" + key + ": " + err.to_string())
      }
    }
    Err(msg) => Err(msg)
  }
}

///|
fn json_get_int(obj : Map[String, Json], key : String) -> Result[Int, String] {
  match json_get(obj, key) {
    Ok(v) => {
      let r = try? @json.from_json(v)
      match r {
        Ok(n) => Ok(n)
        Err(err) => Err("invalid int at key=" + key + ": " + err.to_string())
      }
    }
    Err(msg) => Err(msg)
  }
}

///|
fn json_get_array(
  obj : Map[String, Json],
  key : String,
) -> Result[Array[Json], String] {
  match json_get(obj, key) {
    Ok(v) => {
      let r = try? @json.from_json(v)
      match r {
        Ok(arr) => Ok(arr)
        Err(err) => Err("invalid array at key=" + key + ": " + err.to_string())
      }
    }
    Err(msg) => Err(msg)
  }
}

///|
fn bytes_ascii_to_string(bytes : Bytes) -> String {
  let chars : Array[Char] = []
  for i = 0; i < bytes.length(); i = i + 1 {
    chars.push(bytes[i].to_int().unsafe_to_char())
  }
  String::from_array(chars)
}

///|
async fn load_json_file(path : String) -> Result[Json, String] {
  let read_result = try? @afs.read_file(path)
  let data = match read_result {
    Ok(v) => v
    Err(err) => return Err("failed to read " + path + ": " + err.to_string())
  }
  let text = bytes_ascii_to_string(data.binary())
  let parsed = try? @json.parse(text.to_string_view())
  match parsed {
    Ok(v) => Ok(v)
    Err(err) => Err("failed to parse json " + path + ": " + err.to_string())
  }
}

///|
pub async fn load_token_shards_meta(
  meta_path : String,
) -> Result[LmTokenShardsMeta, String] {
  let json = match load_json_file(meta_path) {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let obj_result = try? @json.from_json(json)
  let obj : Map[String, Json] = match obj_result {
    Ok(v) => v
    Err(err) =>
      return Err("invalid meta object at " + meta_path + ": " + err.to_string())
  }
  let dtype = match json_get_string(obj, "dtype") {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let n_vocab = match json_get_int(obj, "n_vocab") {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let total_tokens = match json_get_int(obj, "total_tokens") {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let shards_json = match json_get_array(obj, "shards") {
    Ok(v) => v
    Err(msg) => return Err(msg)
  }
  let shards : Array[LmTokenShard] = []
  for i = 0; i < shards_json.length(); i = i + 1 {
    let shard_obj_result = try? @json.from_json(shards_json[i])
    let shard_obj : Map[String, Json] = match shard_obj_result {
      Ok(v) => v
      Err(err) =>
        return Err(
          "invalid shard object at index " +
          i.to_string() +
          ": " +
          err.to_string(),
        )
    }
    let file = match json_get_string(shard_obj, "file") {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    let tokens = match json_get_int(shard_obj, "tokens") {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    shards.push({ file, tokens })
  }
  Ok({
    meta_path,
    base_dir: dirname(meta_path),
    dtype,
    n_vocab,
    total_tokens,
    shards,
  })
}

///|
fn append_ids_u16(
  bytes : Bytes,
  out : Array[Int],
  remaining : Int?,
) -> Result[Int, String] {
  let nbytes = bytes.length()
  if nbytes % 2 != 0 {
    return Err("invalid uint16 shard bytes length: " + nbytes.to_string())
  }
  let total = nbytes / 2
  let take = match remaining {
    Some(v) => if v < total { v } else { total }
    None => total
  }
  for i = 0; i < take; i = i + 1 {
    let lo = bytes[i * 2].to_int()
    let hi = bytes[i * 2 + 1].to_int()
    out.push(lo + hi * 256)
  }
  Ok(take)
}

///|
fn append_ids_u32(
  bytes : Bytes,
  out : Array[Int],
  remaining : Int?,
) -> Result[Int, String] {
  let nbytes = bytes.length()
  if nbytes % 4 != 0 {
    return Err("invalid uint32 shard bytes length: " + nbytes.to_string())
  }
  let total = nbytes / 4
  let take = match remaining {
    Some(v) => if v < total { v } else { total }
    None => total
  }
  for i = 0; i < take; i = i + 1 {
    let base = i * 4
    let b0 = bytes[base].to_int()
    let b1 = bytes[base + 1].to_int()
    let b2 = bytes[base + 2].to_int()
    let b3 = bytes[base + 3].to_int()
    out.push(b0 + b1 * 256 + b2 * 65536 + b3 * 16777216)
  }
  Ok(take)
}

///|
pub async fn load_token_ids_from_shards(
  meta : LmTokenShardsMeta,
  max_shards : Int?,
  max_tokens : Int?,
) -> Result[Array[Int], String] {
  let shard_limit = match max_shards {
    Some(v) =>
      if v <= 0 {
        0
      } else if v > meta.shards.length() {
        meta.shards.length()
      } else {
        v
      }
    None => meta.shards.length()
  }
  let mut remaining = match max_tokens {
    Some(v) => if v <= 0 { Some(0) } else { Some(v) }
    None => None
  }
  let ids : Array[Int] = []
  for i = 0; i < shard_limit; i = i + 1 {
    match remaining {
      Some(0) => break
      Some(_) => ()
      None => ()
    }
    let shard = meta.shards[i]
    let path = path_join(meta.base_dir, shard.file)
    let read_result = try? @afs.read_file(path)
    let data = match read_result {
      Ok(v) => v
      Err(err) =>
        return Err("failed to read shard " + path + ": " + err.to_string())
    }
    let bytes = data.binary()
    let added = if meta.dtype == "uint16" {
      append_ids_u16(bytes, ids, remaining)
    } else if meta.dtype == "uint32" {
      append_ids_u32(bytes, ids, remaining)
    } else {
      Err("unsupported shard dtype: " + meta.dtype)
    }
    let n_added = match added {
      Ok(v) => v
      Err(msg) => return Err(msg)
    }
    remaining = match remaining {
      Some(v) => Some(v - n_added)
      None => None
    }
  }
  Ok(ids)
}
