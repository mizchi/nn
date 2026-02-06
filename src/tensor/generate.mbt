///|
/// Greedy text generation: forward → argmax → append, one token at a time
pub fn generate_greedy(
  params : TransformerParams,
  config : TransformerConfig,
  prompt_ids : Array[Int],
  max_len : Int,
) -> Result[Array[Int], String] {
  let ids = prompt_ids.copy()
  for _step = 0; _step < max_len; _step = _step + 1 {
    let seq_len = ids.length()
    if seq_len >= config.max_seq_len {
      break
    }
    // Build batch input [1, seq_len]
    let tokens : Array[Array[Int]] = [ids]
    let mask = match causal_mask(seq_len) {
      Ok(m) => m
      Err(e) => return Err(e)
    }
    let logits = match transformer_forward(tokens, params, config, Some(mask)) {
      Ok(t) => t
      Err(e) => return Err(e)
    }
    // Extract logits at last position: logits[0, seq_len-1, :]
    let vocab_size = config.vocab_size
    let last_logits_data = Array::make(vocab_size, Float::from_int(0))
    for v = 0; v < vocab_size; v = v + 1 {
      last_logits_data[v] = logits.at3(0, seq_len - 1, v)
    }
    let last_logits = tensor_from_array(last_logits_data)
    let next_token = tensor_argmax(last_logits)
    ids.push(next_token)
  }
  Ok(ids)
}

///|
/// Text generation with temperature sampling (LCG-based PRNG)
pub fn generate_with_temperature(
  params : TransformerParams,
  config : TransformerConfig,
  prompt_ids : Array[Int],
  max_len : Int,
  temperature : Float,
  seed : Int,
) -> Result[Array[Int], String] {
  let ids = prompt_ids.copy()
  let mut rng = seed
  for _step = 0; _step < max_len; _step = _step + 1 {
    let seq_len = ids.length()
    if seq_len >= config.max_seq_len {
      break
    }
    let tokens : Array[Array[Int]] = [ids]
    let mask = match causal_mask(seq_len) {
      Ok(m) => m
      Err(e) => return Err(e)
    }
    let logits = match transformer_forward(tokens, params, config, Some(mask)) {
      Ok(t) => t
      Err(e) => return Err(e)
    }
    // Extract last position logits and apply temperature
    let vocab_size = config.vocab_size
    let scaled = Array::make(vocab_size, Float::from_int(0))
    for v = 0; v < vocab_size; v = v + 1 {
      scaled[v] = logits.at3(0, seq_len - 1, v) / temperature
    }
    let scaled_t = tensor_from_array(scaled)
    let probs = match tensor_softmax(scaled_t) {
      Ok(t) => t
      Err(e) => return Err(e)
    }
    // Sample from probability distribution using LCG
    rng = rng * 1103515245 + 12345
    let r = ((rng / 65536 % 32768).abs().to_double() / 32768.0)
      |> Float::from_double
    let mut cumulative = Float::from_int(0)
    let mut next_token = vocab_size - 1
    for v = 0; v < vocab_size; v = v + 1 {
      cumulative = cumulative + probs.at1(v)
      if cumulative > r {
        next_token = v
        break
      }
    }
    ids.push(next_token)
  }
  Ok(ids)
}
