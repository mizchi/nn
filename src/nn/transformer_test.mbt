///|
test "transformer_spec_new_valid" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  inspect(spec.vocab_size, content="8")
  inspect(spec.d_model, content="8")
  inspect(spec.num_heads, content="2")
  inspect(spec.num_layers, content="1")
  inspect(spec.d_ff, content="32")
  inspect(spec.max_seq_len, content="16")
  inspect(spec.batch_size, content="2")
  inspect(spec.seq_len, content="4")
}

///|
test "transformer_spec_new_d_model_not_divisible_by_heads" {
  let result = transformer_spec_new(8, 7, 2, 1, 32, 16, 2, 4)
  inspect(result.is_err(), content="true")
}

///|
test "transformer_plan_buffers_sizes" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  let plan = transformer_plan_buffers(spec)
  // token_embedding: 8 * 8 * 4 = 256
  inspect(plan.token_embedding_bytes, content="256")
  // pos_embedding: 16 * 8 * 4 = 512
  inspect(plan.pos_embedding_bytes, content="512")
  // causal_mask: 4 * 4 * 4 = 64 (seq_len * seq_len * sizeof(f32))
  inspect(plan.causal_mask_bytes, content="64")
  // w_q: 8 * 8 * 4 = 256
  inspect(plan.w_q_bytes, content="256")
  // ff_w1: 8 * 32 * 4 = 1024
  inspect(plan.ff_w1_bytes, content="1024")
  // ff_b1: 32 * 4 = 128
  inspect(plan.ff_b1_bytes, content="128")
  // tokens_in: 2 * 4 * 4 = 32 (u32)
  inspect(plan.tokens_in_bytes, content="32")
  // x: 2 * 4 * 8 * 4 = 256
  inspect(plan.x_bytes, content="256")
  // q: 2 * 2 * 4 * 4 * 4 = 256 (batch * heads * seq * d_k * sizeof(f32))
  inspect(plan.q_bytes, content="256")
  // ff_h: 2 * 4 * 32 * 4 = 1024
  inspect(plan.ff_h_bytes, content="1024")
  // logits: 2 * 4 * 8 * 4 = 256
  inspect(plan.logits_bytes, content="256")
}

///|
test "transformer_plan_resources" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  let res = transformer_plan_resources(spec)
  inspect(res.token_embedding.size, content="256")
  inspect(res.layer_params.length(), content="1")
  inspect(res.layer_params[0].w_q.size, content="256")
  inspect(res.logits.size, content="256")
}

///|
test "transformer_dispatch_plan" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  let dispatch = transformer_dispatch_plan(spec, 64).unwrap()
  // embedding: ceil(2*4 / 64) = 1
  inspect(dispatch.embedding_dispatch_x, content="1")
  // layer_norm: ceil(8 / 64) = 1
  inspect(dispatch.layer_norm_dispatch_x, content="1")
  // attention: ceil(2*2*4 / 64) = 1
  inspect(dispatch.attention_dispatch_x, content="1")
}

///|
test "transformer_shader_plan_generates_wgsl" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  let shaders = transformer_shader_plan(spec, 64).unwrap()
  // Verify all shaders are non-empty
  inspect(shaders.embedding_lookup_wgsl.length() > 0, content="true")
  inspect(shaders.add_vectors_wgsl.length() > 0, content="true")
  inspect(shaders.layer_norm_wgsl.length() > 0, content="true")
  inspect(shaders.batched_matmul_attn_wgsl.length() > 0, content="true")
  inspect(shaders.batched_matmul_ffn_up_wgsl.length() > 0, content="true")
  inspect(shaders.batched_matmul_ffn_down_wgsl.length() > 0, content="true")
  inspect(shaders.batched_matmul_lm_head_wgsl.length() > 0, content="true")
  inspect(shaders.add_bias_ffn_up_wgsl.length() > 0, content="true")
  inspect(shaders.add_bias_ffn_down_wgsl.length() > 0, content="true")
  inspect(shaders.gelu_wgsl.length() > 0, content="true")
  inspect(shaders.reshape_for_heads_wgsl.length() > 0, content="true")
  inspect(shaders.reshape_from_heads_wgsl.length() > 0, content="true")
  inspect(shaders.attention_core_wgsl.length() > 0, content="true")
  inspect(shaders.copy_wgsl.length() > 0, content="true")
}

///|
test "wgsl_embedding_lookup_contains_constants" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  let shaders = transformer_shader_plan(spec, 64).unwrap()
  let wgsl = shaders.embedding_lookup_wgsl
  inspect(wgsl.contains("TOTAL"), content="true")
  inspect(wgsl.contains("D_MODEL"), content="true")
  inspect(wgsl.contains("8u"), content="true")
  inspect(wgsl.contains("embedding"), content="true")
}

///|
test "wgsl_layer_norm_contains_eps" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  let shaders = transformer_shader_plan(spec, 64).unwrap()
  let wgsl = shaders.layer_norm_wgsl
  inspect(wgsl.contains("EPS"), content="true")
  inspect(wgsl.contains("D_MODEL"), content="true")
  inspect(wgsl.contains("gamma"), content="true")
  inspect(wgsl.contains("beta"), content="true")
  inspect(wgsl.contains("variance"), content="true")
}

///|
test "wgsl_attention_core_contains_softmax" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  let shaders = transformer_shader_plan(spec, 64).unwrap()
  let wgsl = shaders.attention_core_wgsl
  inspect(wgsl.contains("SCALE"), content="true")
  inspect(wgsl.contains("max_score"), content="true")
  inspect(wgsl.contains("exp_sum"), content="true")
  inspect(wgsl.contains("mask"), content="true")
}

///|
test "wgsl_gelu_contains_tanh" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  let shaders = transformer_shader_plan(spec, 64).unwrap()
  let wgsl = shaders.gelu_wgsl
  inspect(wgsl.contains("tanh"), content="true")
  inspect(wgsl.contains("SQRT_2_OVER_PI"), content="true")
}

///|
test "wgsl_reshape_for_heads_constants" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  let shaders = transformer_shader_plan(spec, 64).unwrap()
  let wgsl = shaders.reshape_for_heads_wgsl
  inspect(wgsl.contains("HEADS"), content="true")
  inspect(wgsl.contains("D_K"), content="true")
  inspect(wgsl.contains("SEQ"), content="true")
}

///|
test "wgsl_batched_matmul_constants" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  let shaders = transformer_shader_plan(spec, 64).unwrap()
  let wgsl = shaders.batched_matmul_attn_wgsl
  inspect(wgsl.contains("IN_DIM"), content="true")
  inspect(wgsl.contains("OUT_DIM"), content="true")
  // For attn proj: IN_DIM = 8, OUT_DIM = 8
  inspect(wgsl.contains("8u"), content="true")
}

///|
test "wgsl_copy_kernel" {
  let spec = transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4).unwrap()
  let shaders = transformer_shader_plan(spec, 64).unwrap()
  let wgsl = shaders.copy_wgsl
  inspect(wgsl.contains("buffer copy"), content="true")
  inspect(wgsl.contains("dst[i] = src[i]"), content="true")
}
