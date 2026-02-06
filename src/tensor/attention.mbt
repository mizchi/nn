///|
/// Scaled Dot-Product Attention
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
///
/// Q: [batch, seq_q, d_k] or [seq_q, d_k]
/// K: [batch, seq_k, d_k] or [seq_k, d_k]
/// V: [batch, seq_k, d_v] or [seq_k, d_v]
/// mask: optional [seq_q, seq_k] causal mask (0 = attend, -inf = mask)
/// Returns: [batch, seq_q, d_v] or [seq_q, d_v]
pub fn attention(
  q : Tensor,
  k : Tensor,
  v : Tensor,
  mask : Tensor?,
) -> Result[Tensor, String] {
  let d_k = q.dim(-1)
  let scale = Float::from_int(1) /
    (d_k.to_double().sqrt() |> Float::from_double)
  // Q @ K^T
  let k_t = match k.transpose(-2, -1) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  let scores = match tensor_matmul(q, k_t) {
    Ok(s) => s
    Err(e) => return Err(e)
  }
  // Scale
  let scaled = tensor_scale(scores, scale)
  // Apply mask if provided
  let masked = match mask {
    None => scaled
    Some(m) =>
      match apply_attention_mask(scaled, m) {
        Ok(t) => t
        Err(e) => return Err(e)
      }
  }
  // Softmax over last dimension
  let attn_weights = match tensor_softmax(masked) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  // @ V
  tensor_matmul(attn_weights, v)
}

///|
/// Apply attention mask: add mask values to scores
/// mask should be 0 for positions to attend, large negative for masked positions
fn apply_attention_mask(
  scores : Tensor,
  mask : Tensor,
) -> Result[Tensor, String] {
  // Broadcast mask to scores shape if needed
  let sc = scores.contiguous()
  let mc = mask.contiguous()
  if scores.ndim() == 2 && mask.ndim() == 2 {
    // [seq_q, seq_k] + [seq_q, seq_k]
    if scores.dim(0) != mask.dim(0) || scores.dim(1) != mask.dim(1) {
      return Err("mask shape mismatch")
    }
    return tensor_add(sc, mc)
  }
  if scores.ndim() == 3 && mask.ndim() == 2 {
    // [batch, seq_q, seq_k] + [seq_q, seq_k] (broadcast)
    let batch = scores.dim(0)
    let seq_q = scores.dim(1)
    let seq_k = scores.dim(2)
    if mask.dim(0) != seq_q || mask.dim(1) != seq_k {
      return Err("mask shape mismatch for broadcast")
    }
    let result = Array::make(batch * seq_q * seq_k, Float::from_int(0))
    for b = 0; b < batch; b = b + 1 {
      for i = 0; i < seq_q; i = i + 1 {
        for j = 0; j < seq_k; j = j + 1 {
          let idx = b * seq_q * seq_k + i * seq_k + j
          result[idx] = sc.at3(b, i, j) + mc.at2(i, j)
        }
      }
    }
    let out_shape = shape_new_unchecked([batch, seq_q, seq_k])
    return tensor_new(out_shape, result)
  }
  Err(
    "unsupported mask broadcast: scores=" +
    scores.shape.to_string() +
    " mask=" +
    mask.shape.to_string(),
  )
}

///|
/// Create causal mask for autoregressive attention
/// Returns [seq_len, seq_len] with 0 for allowed positions, -1e9 for masked
pub fn causal_mask(seq_len : Int) -> Result[Tensor, String] {
  let shape = match shape_new([seq_len, seq_len]) {
    Ok(s) => s
    Err(e) => return Err(e)
  }
  let data = Array::make(seq_len * seq_len, Float::from_int(0))
  let neg_inf = Float::from_double(-1000000000.0)
  for i = 0; i < seq_len; i = i + 1 {
    for j = 0; j < seq_len; j = j + 1 {
      // Mask future positions (j > i)
      if j > i {
        data[i * seq_len + j] = neg_inf
      }
    }
  }
  tensor_new(shape, data)
}

///|
/// Multi-Head Attention
/// Splits Q, K, V into num_heads, applies attention, concatenates
///
/// Q, K, V: [batch, seq, d_model]
/// W_q, W_k, W_v: [d_model, d_model]
/// W_o: [d_model, d_model]
pub fn multi_head_attention(
  q : Tensor,
  k : Tensor,
  v : Tensor,
  w_q : Tensor,
  w_k : Tensor,
  w_v : Tensor,
  w_o : Tensor,
  num_heads : Int,
  mask : Tensor?,
) -> Result[Tensor, String] {
  let batch = q.dim(0)
  let seq_q = q.dim(1)
  let seq_k = k.dim(1)
  let d_model = q.dim(2)
  if d_model % num_heads != 0 {
    return Err("d_model must be divisible by num_heads")
  }
  let d_k = d_model / num_heads
  // Project Q, K, V: [batch, seq, d_model] @ [d_model, d_model] -> [batch, seq, d_model]
  let q_proj = match batched_linear(q, w_q) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  let k_proj = match batched_linear(k, w_k) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  let v_proj = match batched_linear(v, w_v) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  // Reshape to [batch, seq, num_heads, d_k] then transpose to [batch, num_heads, seq, d_k]
  let q_heads = reshape_for_heads(q_proj, batch, seq_q, num_heads, d_k)
  let k_heads = reshape_for_heads(k_proj, batch, seq_k, num_heads, d_k)
  let v_heads = reshape_for_heads(v_proj, batch, seq_k, num_heads, d_k)
  // Apply attention per head: [batch, num_heads, seq_q, d_k]
  let attn_out = multi_head_attention_core(q_heads, k_heads, v_heads, mask)
  // Reshape back: [batch, num_heads, seq_q, d_k] -> [batch, seq_q, d_model]
  let concat = reshape_from_heads(attn_out, batch, seq_q, num_heads, d_k)
  // Output projection
  batched_linear(concat, w_o)
}

///|
/// Batched linear: [batch, seq, in] @ [in, out] -> [batch, seq, out]
fn batched_linear(x : Tensor, w : Tensor) -> Result[Tensor, String] {
  // x: [batch, seq, in], w: [in, out]
  // For each batch and seq position, compute x[b,s,:] @ w
  let batch = x.dim(0)
  let seq = x.dim(1)
  let in_dim = x.dim(2)
  let out_dim = w.dim(1)
  if w.dim(0) != in_dim {
    return Err("batched_linear dimension mismatch")
  }
  let xc = x.contiguous()
  let wc = w.contiguous()
  let out_shape = shape_new_unchecked([batch, seq, out_dim])
  let result = Array::make(batch * seq * out_dim, Float::from_int(0))
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < seq; s = s + 1 {
      for o = 0; o < out_dim; o = o + 1 {
        let mut sum = Float::from_int(0)
        for i = 0; i < in_dim; i = i + 1 {
          sum = sum + xc.at3(b, s, i) * wc.at2(i, o)
        }
        result[b * seq * out_dim + s * out_dim + o] = sum
      }
    }
  }
  tensor_new(out_shape, result)
}

///|
/// Reshape [batch, seq, d_model] -> [batch, num_heads, seq, d_k]
fn reshape_for_heads(
  x : Tensor,
  batch : Int,
  seq : Int,
  num_heads : Int,
  d_k : Int,
) -> Tensor {
  let xc = x.contiguous()
  let out_shape = shape_new_unchecked([batch, num_heads, seq, d_k])
  let result = Array::make(batch * num_heads * seq * d_k, Float::from_int(0))
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < seq; s = s + 1 {
      for h = 0; h < num_heads; h = h + 1 {
        for d = 0; d < d_k; d = d + 1 {
          let src_idx = b * seq * (num_heads * d_k) +
            s * (num_heads * d_k) +
            h * d_k +
            d
          let dst_idx = b * num_heads * seq * d_k + h * seq * d_k + s * d_k + d
          result[dst_idx] = xc.data[src_idx]
        }
      }
    }
  }
  Tensor::{
    data: result,
    shape: out_shape,
    offset: 0,
    strides: out_shape.strides(),
  }
}

///|
/// Reshape [batch, num_heads, seq, d_k] -> [batch, seq, d_model]
fn reshape_from_heads(
  x : Tensor,
  batch : Int,
  seq : Int,
  num_heads : Int,
  d_k : Int,
) -> Tensor {
  let d_model = num_heads * d_k
  let xc = x.contiguous()
  let out_shape = shape_new_unchecked([batch, seq, d_model])
  let result = Array::make(batch * seq * d_model, Float::from_int(0))
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < seq; s = s + 1 {
      for h = 0; h < num_heads; h = h + 1 {
        for d = 0; d < d_k; d = d + 1 {
          let src_idx = b * num_heads * seq * d_k + h * seq * d_k + s * d_k + d
          let dst_idx = b * seq * d_model + s * d_model + h * d_k + d
          result[dst_idx] = xc.data[src_idx]
        }
      }
    }
  }
  Tensor::{
    data: result,
    shape: out_shape,
    offset: 0,
    strides: out_shape.strides(),
  }
}

///|
/// Core multi-head attention on already split heads
/// Q, K, V: [batch, num_heads, seq, d_k]
fn multi_head_attention_core(
  q : Tensor,
  k : Tensor,
  v : Tensor,
  mask : Tensor?,
) -> Tensor {
  let batch = q.dim(0)
  let num_heads = q.dim(1)
  let seq_q = q.dim(2)
  let seq_k = k.dim(2)
  let d_k = q.dim(3)
  let d_v = v.dim(3)
  let scale = Float::from_int(1) /
    (d_k.to_double().sqrt() |> Float::from_double)
  // For each batch and head, compute attention
  let out_shape = shape_new_unchecked([batch, num_heads, seq_q, d_v])
  let result = Array::make(batch * num_heads * seq_q * d_v, Float::from_int(0))
  let qc = q.contiguous()
  let kc = k.contiguous()
  let vc = v.contiguous()
  let mc = match mask {
    None => None
    Some(m) => Some(m.contiguous())
  }
  for b = 0; b < batch; b = b + 1 {
    for h = 0; h < num_heads; h = h + 1 {
      // Compute scores: Q @ K^T scaled
      let scores = Array::make(seq_q * seq_k, Float::from_int(0))
      for i = 0; i < seq_q; i = i + 1 {
        for j = 0; j < seq_k; j = j + 1 {
          let mut dot = Float::from_int(0)
          for d = 0; d < d_k; d = d + 1 {
            let q_idx = b * num_heads * seq_q * d_k +
              h * seq_q * d_k +
              i * d_k +
              d
            let k_idx = b * num_heads * seq_k * d_k +
              h * seq_k * d_k +
              j * d_k +
              d
            dot = dot + qc.data[q_idx] * kc.data[k_idx]
          }
          let mut score = dot * scale
          // Apply mask
          match mc {
            Some(m) => score = score + m.at2(i, j)
            None => ()
          }
          scores[i * seq_k + j] = score
        }
      }
      // Softmax over seq_k dimension
      for i = 0; i < seq_q; i = i + 1 {
        let base = i * seq_k
        let mut max_val = scores[base]
        for j = 1; j < seq_k; j = j + 1 {
          if scores[base + j] > max_val {
            max_val = scores[base + j]
          }
        }
        let mut sum = Float::from_int(0)
        for j = 0; j < seq_k; j = j + 1 {
          let e = @math.exp((scores[base + j] - max_val).to_double())
            |> Float::from_double
          scores[base + j] = e
          sum = sum + e
        }
        for j = 0; j < seq_k; j = j + 1 {
          scores[base + j] = scores[base + j] / sum
        }
      }
      // Multiply by V: attn_weights @ V
      for i = 0; i < seq_q; i = i + 1 {
        for d = 0; d < d_v; d = d + 1 {
          let mut sum = Float::from_int(0)
          for j = 0; j < seq_k; j = j + 1 {
            let v_idx = b * num_heads * seq_k * d_v +
              h * seq_k * d_v +
              j * d_v +
              d
            sum = sum + scores[i * seq_k + j] * vc.data[v_idx]
          }
          let out_idx = b * num_heads * seq_q * d_v +
            h * seq_q * d_v +
            i * d_v +
            d
          result[out_idx] = sum
        }
      }
    }
  }
  Tensor::{
    data: result,
    shape: out_shape,
    offset: 0,
    strides: out_shape.strides(),
  }
}
