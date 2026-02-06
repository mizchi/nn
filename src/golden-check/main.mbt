///|
fn read_u32_le(bytes : Bytes, offset : Int) -> UInt {
  for j = 0, acc = UInt::default()
      j < 4
      j = j + 1, acc = acc | (bytes[offset + j].to_uint() << (j * 8)) {

  } else {
    acc
  }
}

///|
fn read_f32_le(bytes : Bytes, offset : Int) -> Float {
  let bits = read_u32_le(bytes, offset)
  Float::reinterpret_from_uint(bits)
}

///|
fn read_i32_le(bytes : Bytes, offset : Int) -> Int {
  let bits = read_u32_le(bytes, offset)
  bits.reinterpret_as_int()
}

///|
fn bytes_to_f32_array(bytes : Bytes) -> Array[Float] {
  let count = bytes.length() / 4
  let out = Array::make(count, Float::from_int(0))
  for i = 0; i < count; i = i + 1 {
    out[i] = read_f32_le(bytes, i * 4)
  }
  out
}

///|
fn bytes_to_i32_array(bytes : Bytes) -> Array[Int] {
  let count = bytes.length() / 4
  let out = Array::make(count, 0)
  for i = 0; i < count; i = i + 1 {
    out[i] = read_i32_le(bytes, i * 4)
  }
  out
}

///|
fn slice_f32(data : Array[Float], offset : Int, length : Int) -> Array[Float] {
  Array::makei(length, fn(i) { data[offset + i] })
}

///|
fn make_tensor(
  data : Array[Float],
  offset : Int,
  shape : Array[Int],
) -> @tensor.Tensor {
  let s = @tensor.shape_new_unchecked(shape)
  let length = s.numel()
  let slice = slice_f32(data, offset, length)
  match @tensor.tensor_new(s, slice) {
    Ok(t) => t
    Err(e) => {
      println("tensor_new error: " + e)
      abort("")
    }
  }
}

///|
async fn golden_check_main() -> Unit {
  // Model config
  let vocab_size = 32
  let d_model = 16
  let num_heads = 2
  let num_layers = 2
  let d_ff = 64
  let max_seq_len = 32
  let batch_size = 2
  let seq_len = 8

  // Load golden data files
  let weights_data = try? @afs.read_file("testdata/golden/weights.bin")
  let weights_bytes : Bytes = match weights_data {
    Ok(data) => data.binary()
    Err(e) => {
      println("Failed to read weights.bin: " + e.to_string())
      return
    }
  }
  let tokens_data = try? @afs.read_file("testdata/golden/tokens.bin")
  let tokens_bytes : Bytes = match tokens_data {
    Ok(data) => data.binary()
    Err(e) => {
      println("Failed to read tokens.bin: " + e.to_string())
      return
    }
  }
  let logits_data = try? @afs.read_file("testdata/golden/logits.bin")
  let logits_bytes : Bytes = match logits_data {
    Ok(data) => data.binary()
    Err(e) => {
      println("Failed to read logits.bin: " + e.to_string())
      return
    }
  }
  println(
    "Loaded: weights=" +
    weights_bytes.length().to_string() +
    " tokens=" +
    tokens_bytes.length().to_string() +
    " logits=" +
    logits_bytes.length().to_string(),
  )

  // Decode weights
  let weights = bytes_to_f32_array(weights_bytes)
  let token_ints = bytes_to_i32_array(tokens_bytes)
  let expected_logits = bytes_to_f32_array(logits_bytes)
  println("Weights count: " + weights.length().to_string())
  println("Tokens count: " + token_ints.length().to_string())
  println("Expected logits count: " + expected_logits.length().to_string())

  // Build transformer params from weights
  let mut offset = 0

  // token_embedding [vocab_size, d_model]
  let token_embedding = make_tensor(weights, offset, [vocab_size, d_model])
  offset = offset + vocab_size * d_model

  // pos_embedding [max_seq_len, d_model]
  let pos_embedding = make_tensor(weights, offset, [max_seq_len, d_model])
  offset = offset + max_seq_len * d_model

  // Per-layer block params
  let blocks : Array[@tensor.TransformerBlockParams] = []
  for _layer = 0; _layer < num_layers; _layer = _layer + 1 {
    let ln1_gamma = make_tensor(weights, offset, [d_model])
    offset = offset + d_model
    let ln1_beta = make_tensor(weights, offset, [d_model])
    offset = offset + d_model
    let w_q = make_tensor(weights, offset, [d_model, d_model])
    offset = offset + d_model * d_model
    let w_k = make_tensor(weights, offset, [d_model, d_model])
    offset = offset + d_model * d_model
    let w_v = make_tensor(weights, offset, [d_model, d_model])
    offset = offset + d_model * d_model
    let w_o = make_tensor(weights, offset, [d_model, d_model])
    offset = offset + d_model * d_model
    let ln2_gamma = make_tensor(weights, offset, [d_model])
    offset = offset + d_model
    let ln2_beta = make_tensor(weights, offset, [d_model])
    offset = offset + d_model
    let ff_w1 = make_tensor(weights, offset, [d_model, d_ff])
    offset = offset + d_model * d_ff
    let ff_b1 = make_tensor(weights, offset, [d_ff])
    offset = offset + d_ff
    let ff_w2 = make_tensor(weights, offset, [d_ff, d_model])
    offset = offset + d_ff * d_model
    let ff_b2 = make_tensor(weights, offset, [d_model])
    offset = offset + d_model
    blocks.push(@tensor.TransformerBlockParams::{
      ln1_gamma,
      ln1_beta,
      w_q,
      w_k,
      w_v,
      w_o,
      ln2_gamma,
      ln2_beta,
      ff_w1,
      ff_b1,
      ff_w2,
      ff_b2,
    })
  }

  // ln_final
  let ln_final_gamma = make_tensor(weights, offset, [d_model])
  offset = offset + d_model
  let ln_final_beta = make_tensor(weights, offset, [d_model])
  offset = offset + d_model

  // lm_head [d_model, vocab_size]
  let lm_head = make_tensor(weights, offset, [d_model, vocab_size])
  offset = offset + d_model * vocab_size
  println(
    "Weight offset: " +
    offset.to_string() +
    " / " +
    weights.length().to_string(),
  )
  let params = @tensor.TransformerParams::{
    token_embedding,
    pos_embedding,
    blocks,
    ln_final_gamma,
    ln_final_beta,
    lm_head,
  }

  // Build config
  let config = match
    @tensor.transformer_config(
      vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len,
    ) {
    Ok(c) => c
    Err(e) => {
      println("Config error: " + e)
      return
    }
  }

  // Build tokens as Array[Array[Int]]
  let tokens : Array[Array[Int]] = Array::make(batch_size, [])
  for b = 0; b < batch_size; b = b + 1 {
    let row : Array[Int] = Array::make(seq_len, 0)
    for s = 0; s < seq_len; s = s + 1 {
      row[s] = token_ints[b * seq_len + s]
    }
    tokens[b] = row
  }

  // Build causal mask
  let mask = match @tensor.causal_mask(seq_len) {
    Ok(m) => m
    Err(e) => {
      println("Mask error: " + e)
      return
    }
  }

  // Run forward pass
  let logits = match
    @tensor.transformer_forward(tokens, params, config, Some(mask)) {
    Ok(l) => l
    Err(e) => {
      println("Forward error: " + e)
      return
    }
  }

  // Compare with expected logits
  let actual = logits.to_array()
  let total = expected_logits.length()
  if actual.length() != total {
    println(
      "FAIL: logits length mismatch: actual=" +
      actual.length().to_string() +
      " expected=" +
      total.to_string(),
    )
    return
  }
  let mut max_abs_diff = Float::from_int(0)
  let mut max_diff_idx = 0
  for i = 0; i < total; i = i + 1 {
    let diff = actual[i] - expected_logits[i]
    let abs_diff = if diff < Float::from_int(0) {
      Float::from_int(0) - diff
    } else {
      diff
    }
    if abs_diff > max_abs_diff {
      max_abs_diff = abs_diff
      max_diff_idx = i
    }
  }
  println("Logits count: " + total.to_string())
  println(
    "Max abs diff: " +
    max_abs_diff.to_string() +
    " at index " +
    max_diff_idx.to_string(),
  )
  println(
    "  actual=" +
    actual[max_diff_idx].to_string() +
    " expected=" +
    expected_logits[max_diff_idx].to_string(),
  )

  // First 5 logits comparison
  println("First 5 logits comparison:")
  for i = 0; i < 5; i = i + 1 {
    println(
      "  [" +
      i.to_string() +
      "] actual=" +
      actual[i].to_string() +
      " expected=" +
      expected_logits[i].to_string(),
    )
  }
  let tolerance = Float::from_double(0.0001)
  if max_abs_diff < tolerance {
    println("PASS: max_abs_error < 1e-4")
  } else {
    let tolerance_2 = Float::from_double(0.01)
    if max_abs_diff < tolerance_2 {
      println("WARN: max_abs_error < 1e-2 (check op ordering)")
    } else {
      println("FAIL: max_abs_error >= 1e-2 (likely bug)")
    }
  }
}

///|
fn main {
  @async.run_async_main(golden_check_main)
}
