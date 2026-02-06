///|
/// Embedding lookup: token index -> embedding row copy
/// Each workgroup thread handles one (batch, seq) position
fn wgsl_embedding_lookup(spec : TransformerSpec, workgroup_size : Int) -> String {
  let total = u32_lit(spec.batch_size * spec.seq_len)
  let d_model = u32_lit(spec.d_model)
  let ws = workgroup_size.to_string()
  let s0 = "// Transformer: embedding lookup\n"
  let s1 = s0 + "const TOTAL : u32 = " + total + ";\n"
  let s2 = s1 + "const D_MODEL : u32 = " + d_model + ";\n"
  let s3 = s2 +
    "@group(0) @binding(0) var<storage, read> tokens : array<u32>;\n"
  let s4 = s3 +
    "@group(0) @binding(1) var<storage, read> embedding : array<f32>;\n"
  let s5 = s4 +
    "@group(0) @binding(2) var<storage, read_write> output : array<f32>;\n"
  let s6 = s5 + "@compute @workgroup_size(" + ws + ")\n"
  let s7 = s6 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s8 = s7 + "  let idx = gid.x;\n"
  let s9 = s8 + "  if (idx >= TOTAL) { return; }\n"
  let s10 = s9 + "  let token_id = tokens[idx];\n"
  let s11 = s10 + "  let src_base = token_id * D_MODEL;\n"
  let s12 = s11 + "  let dst_base = idx * D_MODEL;\n"
  let s13 = s12 + "  for (var d : u32 = 0u; d < D_MODEL; d = d + 1u) {\n"
  let s14 = s13 + "    output[dst_base + d] = embedding[src_base + d];\n"
  let s15 = s14 + "  }\n"
  s15 + "}\n"
}

///|
/// Element-wise vector addition: c[i] = a[i] + b[i]
fn wgsl_add_vectors(total : Int, workgroup_size : Int) -> String {
  let n = u32_lit(total)
  let ws = workgroup_size.to_string()
  let s0 = "// Transformer: element-wise add\n"
  let s1 = s0 + "const TOTAL : u32 = " + n + ";\n"
  let s2 = s1 + "@group(0) @binding(0) var<storage, read> a : array<f32>;\n"
  let s3 = s2 + "@group(0) @binding(1) var<storage, read> b : array<f32>;\n"
  let s4 = s3 +
    "@group(0) @binding(2) var<storage, read_write> c : array<f32>;\n"
  let s5 = s4 + "@compute @workgroup_size(" + ws + ")\n"
  let s6 = s5 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s7 = s6 + "  let i = gid.x;\n"
  let s8 = s7 + "  if (i >= TOTAL) { return; }\n"
  let s9 = s8 + "  c[i] = a[i] + b[i];\n"
  s9 + "}\n"
}

///|
/// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn wgsl_gelu(total : Int, workgroup_size : Int) -> String {
  let n = u32_lit(total)
  let ws = workgroup_size.to_string()
  let s0 = "// Transformer: GELU activation\n"
  let s1 = s0 + "const TOTAL : u32 = " + n + ";\n"
  let s2 = s1 + "const SQRT_2_OVER_PI : f32 = 0.7978845608;\n"
  let s3 = s2 + "const COEFF : f32 = 0.044715;\n"
  let s4 = s3 +
    "@group(0) @binding(0) var<storage, read_write> x : array<f32>;\n"
  let s5 = s4 + "@compute @workgroup_size(" + ws + ")\n"
  let s6 = s5 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s7 = s6 + "  let i = gid.x;\n"
  let s8 = s7 + "  if (i >= TOTAL) { return; }\n"
  let s9 = s8 + "  let v = x[i];\n"
  let s10 = s9 +
    "  let inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);\n"
  let s11 = s10 + "  x[i] = 0.5 * v * (1.0 + tanh(inner));\n"
  s11 + "}\n"
}

///|
/// Add bias: x[i * dim + j] += bias[j]
fn wgsl_add_bias(
  batch_seq : Int,
  dim : Int,
  workgroup_size : Int,
) -> String {
  let total = u32_lit(batch_seq * dim)
  let d = u32_lit(dim)
  let ws = workgroup_size.to_string()
  let s0 = "// Transformer: add bias\n"
  let s1 = s0 + "const TOTAL : u32 = " + total + ";\n"
  let s2 = s1 + "const DIM : u32 = " + d + ";\n"
  let s3 = s2 +
    "@group(0) @binding(0) var<storage, read_write> x : array<f32>;\n"
  let s4 = s3 +
    "@group(0) @binding(1) var<storage, read> bias : array<f32>;\n"
  let s5 = s4 + "@compute @workgroup_size(" + ws + ")\n"
  let s6 = s5 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s7 = s6 + "  let i = gid.x;\n"
  let s8 = s7 + "  if (i >= TOTAL) { return; }\n"
  let s9 = s8 + "  let j = i % DIM;\n"
  let s10 = s9 + "  x[i] = x[i] + bias[j];\n"
  s10 + "}\n"
}

///|
/// Copy buffer: dst[i] = src[i]
fn wgsl_copy(total : Int, workgroup_size : Int) -> String {
  let n = u32_lit(total)
  let ws = workgroup_size.to_string()
  let s0 = "// Transformer: buffer copy\n"
  let s1 = s0 + "const TOTAL : u32 = " + n + ";\n"
  let s2 = s1 + "@group(0) @binding(0) var<storage, read> src : array<f32>;\n"
  let s3 = s2 +
    "@group(0) @binding(1) var<storage, read_write> dst : array<f32>;\n"
  let s4 = s3 + "@compute @workgroup_size(" + ws + ")\n"
  let s5 = s4 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s6 = s5 + "  let i = gid.x;\n"
  let s7 = s6 + "  if (i >= TOTAL) { return; }\n"
  let s8 = s7 + "  dst[i] = src[i];\n"
  s8 + "}\n"
}

///|
/// LayerNorm: for each (batch, seq) position, normalize across d_model
/// Each thread handles one (batch, seq) position
fn wgsl_layer_norm(
  batch_seq : Int,
  d_model : Int,
  eps : Float,
  workgroup_size : Int,
) -> String {
  let bs = u32_lit(batch_seq)
  let dm = u32_lit(d_model)
  let ws = workgroup_size.to_string()
  let eps_str = eps.to_double().to_string()
  let s0 = "// Transformer: LayerNorm\n"
  let s1 = s0 + "const BATCH_SEQ : u32 = " + bs + ";\n"
  let s2 = s1 + "const D_MODEL : u32 = " + dm + ";\n"
  let s3 = s2 + "const EPS : f32 = " + eps_str + ";\n"
  let s4 = s3 + "@group(0) @binding(0) var<storage, read> input : array<f32>;\n"
  let s5 = s4 +
    "@group(0) @binding(1) var<storage, read> gamma : array<f32>;\n"
  let s6 = s5 +
    "@group(0) @binding(2) var<storage, read> beta : array<f32>;\n"
  let s7 = s6 +
    "@group(0) @binding(3) var<storage, read_write> output : array<f32>;\n"
  let s8 = s7 + "@compute @workgroup_size(" + ws + ")\n"
  let s9 = s8 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s10 = s9 + "  let pos = gid.x;\n"
  let s11 = s10 + "  if (pos >= BATCH_SEQ) { return; }\n"
  let s12 = s11 + "  let base = pos * D_MODEL;\n"
  // Compute mean
  let s13 = s12 + "  var sum = 0.0;\n"
  let s14 = s13 + "  for (var d : u32 = 0u; d < D_MODEL; d = d + 1u) {\n"
  let s15 = s14 + "    sum = sum + input[base + d];\n"
  let s16 = s15 + "  }\n"
  let s17 = s16 + "  let mean = sum / f32(D_MODEL);\n"
  // Compute variance
  let s18 = s17 + "  var var_sum = 0.0;\n"
  let s19 = s18 + "  for (var d : u32 = 0u; d < D_MODEL; d = d + 1u) {\n"
  let s20 = s19 + "    let diff = input[base + d] - mean;\n"
  let s21 = s20 + "    var_sum = var_sum + diff * diff;\n"
  let s22 = s21 + "  }\n"
  let s23 = s22 + "  let variance = var_sum / f32(D_MODEL);\n"
  let s24 = s23 + "  let inv_std = 1.0 / sqrt(variance + EPS);\n"
  // Normalize
  let s25 = s24 + "  for (var d : u32 = 0u; d < D_MODEL; d = d + 1u) {\n"
  let s26 = s25 + "    let normalized = (input[base + d] - mean) * inv_std;\n"
  let s27 = s26 +
    "    output[base + d] = gamma[d] * normalized + beta[d];\n"
  let s28 = s27 + "  }\n"
  s28 + "}\n"
}

///|
/// Batched matmul: output[i, k] = sum_j(input[i, j] * weight[j, k])
/// Each thread computes one output element
/// input: [batch_seq, in_dim], weight: [in_dim, out_dim], output: [batch_seq, out_dim]
fn wgsl_batched_matmul(
  batch_seq : Int,
  in_dim : Int,
  out_dim : Int,
  workgroup_size : Int,
) -> String {
  let total = u32_lit(batch_seq * out_dim)
  let in_d = u32_lit(in_dim)
  let out_d = u32_lit(out_dim)
  let ws = workgroup_size.to_string()
  let s0 = "// Transformer: batched matmul\n"
  let s1 = s0 + "const TOTAL : u32 = " + total + ";\n"
  let s2 = s1 + "const IN_DIM : u32 = " + in_d + ";\n"
  let s3 = s2 + "const OUT_DIM : u32 = " + out_d + ";\n"
  let s4 = s3 + "@group(0) @binding(0) var<storage, read> input : array<f32>;\n"
  let s5 = s4 +
    "@group(0) @binding(1) var<storage, read> weight : array<f32>;\n"
  let s6 = s5 +
    "@group(0) @binding(2) var<storage, read_write> output : array<f32>;\n"
  let s7 = s6 + "@compute @workgroup_size(" + ws + ")\n"
  let s8 = s7 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s9 = s8 + "  let idx = gid.x;\n"
  let s10 = s9 + "  if (idx >= TOTAL) { return; }\n"
  let s11 = s10 + "  let row = idx / OUT_DIM;\n"
  let s12 = s11 + "  let col = idx % OUT_DIM;\n"
  let s13 = s12 + "  var acc = 0.0;\n"
  let s14 = s13 + "  for (var j : u32 = 0u; j < IN_DIM; j = j + 1u) {\n"
  let s15 = s14 +
    "    acc = acc + input[row * IN_DIM + j] * weight[j * OUT_DIM + col];\n"
  let s16 = s15 + "  }\n"
  let s17 = s16 + "  output[idx] = acc;\n"
  s17 + "}\n"
}

///|
/// Reshape for heads: [batch, seq, d_model] -> [batch, heads, seq, d_k]
/// where d_model = heads * d_k
fn wgsl_reshape_for_heads(
  batch : Int,
  seq : Int,
  heads : Int,
  d_k : Int,
  workgroup_size : Int,
) -> String {
  let total = u32_lit(batch * heads * seq * d_k)
  let seq_u = u32_lit(seq)
  let heads_u = u32_lit(heads)
  let dk_u = u32_lit(d_k)
  let d_model_u = u32_lit(heads * d_k)
  let ws = workgroup_size.to_string()
  let s0 = "// Transformer: reshape for heads [b,s,d] -> [b,h,s,dk]\n"
  let s1 = s0 + "const TOTAL : u32 = " + total + ";\n"
  let s2 = s1 + "const SEQ : u32 = " + seq_u + ";\n"
  let s3 = s2 + "const HEADS : u32 = " + heads_u + ";\n"
  let s4 = s3 + "const D_K : u32 = " + dk_u + ";\n"
  let s5 = s4 + "const D_MODEL : u32 = " + d_model_u + ";\n"
  let s6 = s5 + "@group(0) @binding(0) var<storage, read> input : array<f32>;\n"
  let s7 = s6 +
    "@group(0) @binding(1) var<storage, read_write> output : array<f32>;\n"
  let s8 = s7 + "@compute @workgroup_size(" + ws + ")\n"
  let s9 = s8 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s10 = s9 + "  let i = gid.x;\n"
  let s11 = s10 + "  if (i >= TOTAL) { return; }\n"
  // Decode output index [b, h, s, dk]
  let s12 = s11 + "  let dk_idx = i % D_K;\n"
  let s13 = s12 + "  let s_idx = (i / D_K) % SEQ;\n"
  let s14 = s13 + "  let h_idx = (i / (D_K * SEQ)) % HEADS;\n"
  let s15 = s14 + "  let b_idx = i / (D_K * SEQ * HEADS);\n"
  // Source index in [b, s, d_model] where d_model = h * d_k
  let s16 = s15 +
    "  let src = b_idx * SEQ * D_MODEL + s_idx * D_MODEL + h_idx * D_K + dk_idx;\n"
  let s17 = s16 + "  output[i] = input[src];\n"
  s17 + "}\n"
}

///|
/// Reshape from heads: [batch, heads, seq, d_k] -> [batch, seq, d_model]
fn wgsl_reshape_from_heads(
  batch : Int,
  seq : Int,
  heads : Int,
  d_k : Int,
  workgroup_size : Int,
) -> String {
  let total = u32_lit(batch * seq * heads * d_k)
  let seq_u = u32_lit(seq)
  let heads_u = u32_lit(heads)
  let dk_u = u32_lit(d_k)
  let d_model_u = u32_lit(heads * d_k)
  let ws = workgroup_size.to_string()
  let s0 = "// Transformer: reshape from heads [b,h,s,dk] -> [b,s,d]\n"
  let s1 = s0 + "const TOTAL : u32 = " + total + ";\n"
  let s2 = s1 + "const SEQ : u32 = " + seq_u + ";\n"
  let s3 = s2 + "const HEADS : u32 = " + heads_u + ";\n"
  let s4 = s3 + "const D_K : u32 = " + dk_u + ";\n"
  let s5 = s4 + "const D_MODEL : u32 = " + d_model_u + ";\n"
  let s6 = s5 +
    "@group(0) @binding(0) var<storage, read> input : array<f32>;\n"
  let s7 = s6 +
    "@group(0) @binding(1) var<storage, read_write> output : array<f32>;\n"
  let s8 = s7 + "@compute @workgroup_size(" + ws + ")\n"
  let s9 = s8 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s10 = s9 + "  let i = gid.x;\n"
  let s11 = s10 + "  if (i >= TOTAL) { return; }\n"
  // Decode output index [b, s, d_model] where d_model = h * d_k
  let s12 = s11 + "  let d_idx = i % D_MODEL;\n"
  let s13 = s12 + "  let s_idx = (i / D_MODEL) % SEQ;\n"
  let s14 = s13 + "  let b_idx = i / (D_MODEL * SEQ);\n"
  let s15 = s14 + "  let h_idx = d_idx / D_K;\n"
  let s16 = s15 + "  let dk_idx = d_idx % D_K;\n"
  // Source index in [b, h, s, d_k]
  let s17 = s16 +
    "  let src = b_idx * HEADS * SEQ * D_K + h_idx * SEQ * D_K + s_idx * D_K + dk_idx;\n"
  let s18 = s17 + "  output[i] = input[src];\n"
  s18 + "}\n"
}

///|
/// Attention core: fused Q@K^T/sqrt(dk) + causal mask + softmax + @V
/// Each thread handles one (batch, head, seq_q) position
/// Q: [batch, heads, seq_q, d_k]
/// K: [batch, heads, seq_k, d_k]
/// V: [batch, heads, seq_k, d_k]
/// mask: [seq_q, seq_k] (causal mask, shared across batch/heads)
/// output: [batch, heads, seq_q, d_k]
fn wgsl_attention_core(
  batch : Int,
  heads : Int,
  seq_q : Int,
  seq_k : Int,
  d_k : Int,
  workgroup_size : Int,
) -> String {
  let total = u32_lit(batch * heads * seq_q)
  let seq_q_u = u32_lit(seq_q)
  let seq_k_u = u32_lit(seq_k)
  let dk_u = u32_lit(d_k)
  let heads_u = u32_lit(heads)
  let scale_str = (1.0 / d_k.to_double().sqrt()).to_string()
  let ws = workgroup_size.to_string()
  let s0 = "// Transformer: attention core (Q@K^T/sqrt(dk) + mask + softmax + @V)\n"
  let s1 = s0 + "const TOTAL : u32 = " + total + ";\n"
  let s2 = s1 + "const SEQ_Q : u32 = " + seq_q_u + ";\n"
  let s3 = s2 + "const SEQ_K : u32 = " + seq_k_u + ";\n"
  let s4 = s3 + "const D_K : u32 = " + dk_u + ";\n"
  let s5 = s4 + "const HEADS : u32 = " + heads_u + ";\n"
  let s6 = s5 + "const SCALE : f32 = " + scale_str + ";\n"
  let s7 = s6 + "@group(0) @binding(0) var<storage, read> q : array<f32>;\n"
  let s8 = s7 + "@group(0) @binding(1) var<storage, read> k : array<f32>;\n"
  let s9 = s8 + "@group(0) @binding(2) var<storage, read> v : array<f32>;\n"
  let s10 = s9 +
    "@group(0) @binding(3) var<storage, read> mask : array<f32>;\n"
  let s11 = s10 +
    "@group(0) @binding(4) var<storage, read_write> output : array<f32>;\n"
  let s12 = s11 + "@compute @workgroup_size(" + ws + ")\n"
  let s13 = s12 + "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
  let s14 = s13 + "  let idx = gid.x;\n"
  let s15 = s14 + "  if (idx >= TOTAL) { return; }\n"
  // Decode [batch, head, seq_q]
  let s16 = s15 + "  let sq = idx % SEQ_Q;\n"
  let s17 = s16 + "  let h = (idx / SEQ_Q) % HEADS;\n"
  let s18 = s17 + "  let b = idx / (SEQ_Q * HEADS);\n"
  // Base offsets for Q, K, V in [batch, heads, seq, d_k]
  let s19 = s18 + "  let q_base = b * HEADS * SEQ_Q * D_K + h * SEQ_Q * D_K + sq * D_K;\n"
  let s20 = s19 + "  let kv_base = b * HEADS * SEQ_K * D_K + h * SEQ_K * D_K;\n"
  // Compute Q @ K^T for this query position, apply scale and mask
  // Cache scores in a fixed-size array to avoid recomputation
  let s21 = s20 + "  var scores : array<f32, " + seq_k_u + ">;\n"
  // First pass: compute all scores and find max
  let s22 = s21 + "  var max_score = -1e30;\n"
  let s23 = s22 + "  for (var sk : u32 = 0u; sk < SEQ_K; sk = sk + 1u) {\n"
  let s24 = s23 + "    var dot = 0.0;\n"
  let s25 = s24 + "    for (var d : u32 = 0u; d < D_K; d = d + 1u) {\n"
  let s26 = s25 + "      dot = dot + q[q_base + d] * k[kv_base + sk * D_K + d];\n"
  let s27 = s26 + "    }\n"
  let s28 = s27 + "    scores[sk] = dot * SCALE + mask[sq * SEQ_K + sk];\n"
  let s29 = s28 + "    if (scores[sk] > max_score) { max_score = scores[sk]; }\n"
  let s30 = s29 + "  }\n"
  // Second pass: compute exp weights and sum
  let s31 = s30 + "  var weights : array<f32, " + seq_k_u + ">;\n"
  let s32 = s31 + "  var exp_sum = 0.0;\n"
  let s33 = s32 + "  for (var sk : u32 = 0u; sk < SEQ_K; sk = sk + 1u) {\n"
  let s34 = s33 + "    weights[sk] = exp(scores[sk] - max_score);\n"
  let s35 = s34 + "    exp_sum = exp_sum + weights[sk];\n"
  let s36 = s35 + "  }\n"
  // Normalize weights
  let s37 = s36 + "  for (var sk : u32 = 0u; sk < SEQ_K; sk = sk + 1u) {\n"
  let s38 = s37 + "    weights[sk] = weights[sk] / exp_sum;\n"
  let s39 = s38 + "  }\n"
  // Third pass: compute weighted sum of V
  let s40 = s39 + "  let out_base = b * HEADS * SEQ_Q * D_K + h * SEQ_Q * D_K + sq * D_K;\n"
  let s41 = s40 + "  for (var d : u32 = 0u; d < D_K; d = d + 1u) {\n"
  let s42 = s41 + "    var weighted = 0.0;\n"
  let s43 = s42 + "    for (var sk : u32 = 0u; sk < SEQ_K; sk = sk + 1u) {\n"
  let s44 = s43 +
    "      weighted = weighted + weights[sk] * v[kv_base + sk * D_K + d];\n"
  let s45 = s44 + "    }\n"
  let s46 = s45 + "    output[out_base + d] = weighted;\n"
  let s47 = s46 + "  }\n"
  s47 + "}\n"
}
