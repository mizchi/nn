///|
/// GPU Transformer forward pass for inference verification
/// Runs the full transformer forward pass on GPU and returns logits
async fn gpu_transformer_forward(
  spec : @nn.TransformerSpec,
  cpu_params : @tensor.TransformerParams,
  tokens : Array[Array[Int]],
  workgroup_size : Int,
) -> Result[Array[Float], String] {
  let adapter = match
    @wgpu.request_adapter(@wgpu.request_adapter_options_default()) {
    Ok(a) => a
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let device = match
    @wgpu.request_device(adapter, @wgpu.device_descriptor_default()) {
    Ok(d) => d
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  let queue = match @wgpu.device_get_queue(device) {
    Ok(q) => q
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  // Plans
  let buf_plan = @nn.transformer_plan_buffers(spec)
  let dispatch = match @nn.transformer_dispatch_plan(spec, workgroup_size) {
    Ok(d) => d
    Err(err) => return Err(err.to_string())
  }
  let shaders = match @nn.transformer_shader_plan(spec, workgroup_size) {
    Ok(s) => s
    Err(err) => return Err(err.to_string())
  }
  let upload = buffer_usage_storage_upload()
  let readback = @wgpu.buffer_usage_or(upload, @wgpu.buffer_usage_copy_src)
  // Create buffers - parameters
  let tok_emb_buf = match
    gpu_create_buf(device, buf_plan.token_embedding_bytes, upload, "tf_tok_emb") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let pos_emb_buf = match
    gpu_create_buf(device, buf_plan.pos_embedding_bytes, upload, "tf_pos_emb") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let mask_buf = match
    gpu_create_buf(device, buf_plan.causal_mask_bytes, upload, "tf_mask") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let ln_final_g_buf = match
    gpu_create_buf(
      device,
      buf_plan.ln_final_gamma_bytes,
      upload,
      "tf_ln_final_g",
    ) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let ln_final_b_buf = match
    gpu_create_buf(
      device,
      buf_plan.ln_final_beta_bytes,
      upload,
      "tf_ln_final_b",
    ) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let lm_head_buf = match
    gpu_create_buf(device, buf_plan.lm_head_bytes, upload, "tf_lm_head") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // Per-layer param buffers
  let layer_bufs : Array[TransformerLayerBufs] = []
  for i = 0; i < spec.num_layers; i = i + 1 {
    let p = "l" + i.to_string() + "_"
    let ln1_g = match
      gpu_create_buf(device, buf_plan.ln1_gamma_bytes, upload, p + "ln1g") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let ln1_b = match
      gpu_create_buf(device, buf_plan.ln1_beta_bytes, upload, p + "ln1b") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let w_q = match
      gpu_create_buf(device, buf_plan.w_q_bytes, upload, p + "wq") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let w_k = match
      gpu_create_buf(device, buf_plan.w_k_bytes, upload, p + "wk") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let w_v = match
      gpu_create_buf(device, buf_plan.w_v_bytes, upload, p + "wv") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let w_o = match
      gpu_create_buf(device, buf_plan.w_o_bytes, upload, p + "wo") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let ln2_g = match
      gpu_create_buf(device, buf_plan.ln2_gamma_bytes, upload, p + "ln2g") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let ln2_b = match
      gpu_create_buf(device, buf_plan.ln2_beta_bytes, upload, p + "ln2b") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let ff_w1 = match
      gpu_create_buf(device, buf_plan.ff_w1_bytes, upload, p + "ffw1") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let ff_b1 = match
      gpu_create_buf(device, buf_plan.ff_b1_bytes, upload, p + "ffb1") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let ff_w2 = match
      gpu_create_buf(device, buf_plan.ff_w2_bytes, upload, p + "ffw2") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let ff_b2 = match
      gpu_create_buf(device, buf_plan.ff_b2_bytes, upload, p + "ffb2") {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    let lb = TransformerLayerBufs::{
      ln1_g,
      ln1_b,
      w_q,
      w_k,
      w_v,
      w_o,
      ln2_g,
      ln2_b,
      ff_w1,
      ff_b1,
      ff_w2,
      ff_b2,
    }
    layer_bufs.push(lb)
  }
  // Activation buffers
  let tokens_buf = match
    gpu_create_buf(device, buf_plan.tokens_in_bytes, upload, "tf_tokens") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let x_buf = match gpu_create_buf(device, buf_plan.x_bytes, readback, "tf_x") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let x_tmp_buf = match
    gpu_create_buf(device, buf_plan.x_tmp_bytes, readback, "tf_x_tmp") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let x_res_buf = match
    gpu_create_buf(device, buf_plan.x_residual_bytes, readback, "tf_x_res") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let q_buf = match gpu_create_buf(device, buf_plan.q_bytes, readback, "tf_q") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let k_buf = match gpu_create_buf(device, buf_plan.k_bytes, readback, "tf_k") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let v_buf = match gpu_create_buf(device, buf_plan.v_bytes, readback, "tf_v") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let attn_out_buf = match
    gpu_create_buf(device, buf_plan.attn_out_bytes, readback, "tf_attn_out") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let attn_proj_buf = match
    gpu_create_buf(device, buf_plan.attn_proj_bytes, readback, "tf_attn_proj") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let ff_h_buf = match
    gpu_create_buf(device, buf_plan.ff_h_bytes, readback, "tf_ff_h") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let logits_buf = match
    gpu_create_buf(device, buf_plan.logits_bytes, readback, "tf_logits") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // Upload parameters
  let batch = spec.batch_size
  let seq = spec.seq_len
  // Flatten tokens to u32 array
  let flat_tokens = Array::make(batch * seq, 0)
  for b = 0; b < batch; b = b + 1 {
    for s = 0; s < seq; s = s + 1 {
      flat_tokens[b * seq + s] = tokens[b][s]
    }
  }
  gpu_write_u32(queue, tokens_buf, flat_tokens)
  gpu_write_f32(
    queue,
    tok_emb_buf,
    cpu_params.token_embedding.contiguous().data,
  )
  gpu_write_f32(queue, pos_emb_buf, cpu_params.pos_embedding.contiguous().data)
  // Causal mask: 0 for allowed, -1e10 for masked (seq_len x seq_len)
  let mask_data = Array::make(seq * seq, Float::from_int(0))
  for i = 0; i < seq; i = i + 1 {
    for j = 0; j < seq; j = j + 1 {
      if j > i {
        mask_data[i * seq + j] = Float::from_double(-1.0e10)
      }
    }
  }
  gpu_write_f32(queue, mask_buf, mask_data)
  gpu_write_f32(
    queue,
    ln_final_g_buf,
    cpu_params.ln_final_gamma.contiguous().data,
  )
  gpu_write_f32(
    queue,
    ln_final_b_buf,
    cpu_params.ln_final_beta.contiguous().data,
  )
  gpu_write_f32(queue, lm_head_buf, cpu_params.lm_head.contiguous().data)
  for i = 0; i < spec.num_layers; i = i + 1 {
    let bp = cpu_params.blocks[i]
    let lb = layer_bufs[i]
    gpu_write_f32(queue, lb.ln1_g, bp.ln1_gamma.contiguous().data)
    gpu_write_f32(queue, lb.ln1_b, bp.ln1_beta.contiguous().data)
    gpu_write_f32(queue, lb.w_q, bp.w_q.contiguous().data)
    gpu_write_f32(queue, lb.w_k, bp.w_k.contiguous().data)
    gpu_write_f32(queue, lb.w_v, bp.w_v.contiguous().data)
    gpu_write_f32(queue, lb.w_o, bp.w_o.contiguous().data)
    gpu_write_f32(queue, lb.ln2_g, bp.ln2_gamma.contiguous().data)
    gpu_write_f32(queue, lb.ln2_b, bp.ln2_beta.contiguous().data)
    gpu_write_f32(queue, lb.ff_w1, bp.ff_w1.contiguous().data)
    gpu_write_f32(queue, lb.ff_b1, bp.ff_b1.contiguous().data)
    gpu_write_f32(queue, lb.ff_w2, bp.ff_w2.contiguous().data)
    gpu_write_f32(queue, lb.ff_b2, bp.ff_b2.contiguous().data)
  }
  // Create shader modules
  let stage = @wgpu.shader_stage_compute
  let ro = @wgpu.binding_type_read_only_storage_buffer
  let rw = @wgpu.binding_type_storage_buffer
  let emb_mod = match gpu_create_shader(device, shaders.embedding_lookup_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let add_mod = match gpu_create_shader(device, shaders.add_vectors_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let ln_mod = match gpu_create_shader(device, shaders.layer_norm_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let mm_attn_mod = match
    gpu_create_shader(device, shaders.batched_matmul_attn_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let mm_ffn_up_mod = match
    gpu_create_shader(device, shaders.batched_matmul_ffn_up_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let mm_ffn_down_mod = match
    gpu_create_shader(device, shaders.batched_matmul_ffn_down_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let mm_lm_head_mod = match
    gpu_create_shader(device, shaders.batched_matmul_lm_head_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let add_bias_up_mod = match
    gpu_create_shader(device, shaders.add_bias_ffn_up_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let add_bias_down_mod = match
    gpu_create_shader(device, shaders.add_bias_ffn_down_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let gelu_mod = match gpu_create_shader(device, shaders.gelu_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let reshape_for_mod = match
    gpu_create_shader(device, shaders.reshape_for_heads_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let reshape_from_mod = match
    gpu_create_shader(device, shaders.reshape_from_heads_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let attn_core_mod = match
    gpu_create_shader(device, shaders.attention_core_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let copy_mod = match gpu_create_shader(device, shaders.copy_wgsl) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // --- Create layouts, bind groups, pipelines ---
  // embedding_lookup: tokens(ro), embedding(ro), output(rw)
  let emb_layout = match gpu_create_layout(device, stage, [ro, ro, rw]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let emb_group = match
    gpu_create_group(device, emb_layout, [
      (tokens_buf, buf_plan.tokens_in_bytes),
      (tok_emb_buf, buf_plan.token_embedding_bytes),
      (x_buf, buf_plan.x_bytes),
    ]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let emb_pipe = match gpu_create_pipeline(device, emb_layout, emb_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // add_vectors (for pos embedding): a(ro), b(ro), c(rw)
  // We need a special version for pos: x + pos_embed[0:seq*d_model]
  // But our add_vectors works on total elements = batch*seq*d_model
  // pos_embedding is [max_seq_len, d_model], we only need first seq positions
  // We need to upload just the first seq*d_model elements of pos_embedding as a separate buffer
  let bs_dm = batch * seq * spec.d_model
  let pos_slice_bytes = bs_dm * 4
  let pos_slice_buf = match
    gpu_create_buf(device, pos_slice_bytes, upload, "tf_pos_slice") {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // Upload positional embedding slice (replicated for batch)
  let pos_data = cpu_params.pos_embedding.contiguous().data
  let pos_slice = Array::make(bs_dm, Float::from_int(0))
  for b_idx = 0; b_idx < batch; b_idx = b_idx + 1 {
    for s_idx = 0; s_idx < seq; s_idx = s_idx + 1 {
      for d = 0; d < spec.d_model; d = d + 1 {
        pos_slice[b_idx * seq * spec.d_model + s_idx * spec.d_model + d] = pos_data[s_idx *
          spec.d_model +
          d]
      }
    }
  }
  gpu_write_f32(queue, pos_slice_buf, pos_slice)
  let add_layout = match gpu_create_layout(device, stage, [ro, ro, rw]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let add_pos_group = match
    gpu_create_group(device, add_layout, [
      (x_buf, buf_plan.x_bytes),
      (pos_slice_buf, pos_slice_bytes),
      (x_tmp_buf, buf_plan.x_tmp_bytes),
    ]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let add_pipe = match gpu_create_pipeline(device, add_layout, add_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // copy: src(ro), dst(rw)
  let copy_layout = match gpu_create_layout(device, stage, [ro, rw]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let copy_x_to_res = match
    gpu_create_group(device, copy_layout, [
      (x_buf, buf_plan.x_bytes),
      (x_res_buf, buf_plan.x_residual_bytes),
    ]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let copy_tmp_to_x = match
    gpu_create_group(device, copy_layout, [
      (x_tmp_buf, buf_plan.x_tmp_bytes),
      (x_buf, buf_plan.x_bytes),
    ]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let copy_pipe = match gpu_create_pipeline(device, copy_layout, copy_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // layer_norm: input(ro), gamma(ro), beta(ro), output(rw)
  let ln_layout = match gpu_create_layout(device, stage, [ro, ro, ro, rw]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let ln_pipe = match gpu_create_pipeline(device, ln_layout, ln_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // batched_matmul: input(ro), weight(ro), output(rw)
  let mm_layout = match gpu_create_layout(device, stage, [ro, ro, rw]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let mm_attn_pipe = match gpu_create_pipeline(device, mm_layout, mm_attn_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let mm_ffn_up_pipe = match
    gpu_create_pipeline(device, mm_layout, mm_ffn_up_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let mm_ffn_down_pipe = match
    gpu_create_pipeline(device, mm_layout, mm_ffn_down_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let mm_lm_head_pipe = match
    gpu_create_pipeline(device, mm_layout, mm_lm_head_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // reshape: input(ro), output(rw)
  let reshape_layout = match gpu_create_layout(device, stage, [ro, rw]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let reshape_from = match
    gpu_create_group(device, reshape_layout, [
      (attn_out_buf, buf_plan.attn_out_bytes),
      (attn_proj_buf, buf_plan.attn_proj_bytes),
    ]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let reshape_for_pipe = match
    gpu_create_pipeline(device, reshape_layout, reshape_for_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let reshape_from_pipe = match
    gpu_create_pipeline(device, reshape_layout, reshape_from_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // attention_core: q(ro), k(ro), v(ro), mask(ro), output(rw)
  let attn_layout = match
    gpu_create_layout(device, stage, [ro, ro, ro, ro, rw]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let attn_group = match
    gpu_create_group(device, attn_layout, [
      (q_buf, buf_plan.q_bytes),
      (k_buf, buf_plan.k_bytes),
      (v_buf, buf_plan.v_bytes),
      (mask_buf, buf_plan.causal_mask_bytes),
      (attn_out_buf, buf_plan.attn_out_bytes),
    ]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let attn_pipe = match
    gpu_create_pipeline(device, attn_layout, attn_core_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // add_bias (in-place): x(rw), bias(ro)
  let bias_layout = match gpu_create_layout(device, stage, [rw, ro]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let bias_up_pipe = match
    gpu_create_pipeline(device, bias_layout, add_bias_up_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let bias_down_pipe = match
    gpu_create_pipeline(device, bias_layout, add_bias_down_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // gelu (in-place): x(rw)
  let gelu_layout = match gpu_create_layout(device, stage, [rw]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let gelu_group = match
    gpu_create_group(device, gelu_layout, [(ff_h_buf, buf_plan.ff_h_bytes)]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  let gelu_pipe = match gpu_create_pipeline(device, gelu_layout, gelu_mod) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  // --- Dispatch forward pass ---
  // 1. Embedding lookup
  match
    gpu_dispatch(device, emb_pipe, emb_group, dispatch.embedding_dispatch_x) {
    Ok(_) => ()
    Err(err) => return Err(err)
  }
  // 2. Add positional embedding: x_tmp = x + pos_slice
  match
    gpu_dispatch(device, add_pipe, add_pos_group, dispatch.add_pos_dispatch_x) {
    Ok(_) => ()
    Err(err) => return Err(err)
  }
  // Copy x_tmp -> x
  match
    gpu_dispatch(
      device,
      copy_pipe,
      copy_tmp_to_x,
      dispatch.x_elementwise_dispatch_x,
    ) {
    Ok(_) => ()
    Err(err) => return Err(err)
  }
  // 3. For each layer
  for layer_idx = 0; layer_idx < spec.num_layers; layer_idx = layer_idx + 1 {
    let lb = layer_bufs[layer_idx]
    // copy x -> x_residual
    match
      gpu_dispatch(
        device,
        copy_pipe,
        copy_x_to_res,
        dispatch.x_elementwise_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // LayerNorm1: x -> x_tmp
    let ln1_group = match
      gpu_create_group(device, ln_layout, [
        (x_buf, buf_plan.x_bytes),
        (lb.ln1_g, buf_plan.ln1_gamma_bytes),
        (lb.ln1_b, buf_plan.ln1_beta_bytes),
        (x_tmp_buf, buf_plan.x_tmp_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(device, ln_pipe, ln1_group, dispatch.layer_norm_dispatch_x) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // Q projection: x_tmp @ w_q -> attn_proj (reuse as temp for q_flat)
    let q_proj_group = match
      gpu_create_group(device, mm_layout, [
        (x_tmp_buf, buf_plan.x_tmp_bytes),
        (lb.w_q, buf_plan.w_q_bytes),
        (attn_proj_buf, buf_plan.attn_proj_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        mm_attn_pipe,
        q_proj_group,
        dispatch.attn_proj_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // reshape attn_proj -> q
    let reshape_q = match
      gpu_create_group(device, reshape_layout, [
        (attn_proj_buf, buf_plan.attn_proj_bytes),
        (q_buf, buf_plan.q_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        reshape_for_pipe,
        reshape_q,
        dispatch.reshape_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // K projection: x_tmp @ w_k -> attn_proj
    let k_proj_group = match
      gpu_create_group(device, mm_layout, [
        (x_tmp_buf, buf_plan.x_tmp_bytes),
        (lb.w_k, buf_plan.w_k_bytes),
        (attn_proj_buf, buf_plan.attn_proj_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        mm_attn_pipe,
        k_proj_group,
        dispatch.attn_proj_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // reshape attn_proj -> k
    let reshape_k = match
      gpu_create_group(device, reshape_layout, [
        (attn_proj_buf, buf_plan.attn_proj_bytes),
        (k_buf, buf_plan.k_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        reshape_for_pipe,
        reshape_k,
        dispatch.reshape_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // V projection: x_tmp @ w_v -> attn_proj
    let v_proj_group = match
      gpu_create_group(device, mm_layout, [
        (x_tmp_buf, buf_plan.x_tmp_bytes),
        (lb.w_v, buf_plan.w_v_bytes),
        (attn_proj_buf, buf_plan.attn_proj_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        mm_attn_pipe,
        v_proj_group,
        dispatch.attn_proj_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // reshape attn_proj -> v
    let reshape_v = match
      gpu_create_group(device, reshape_layout, [
        (attn_proj_buf, buf_plan.attn_proj_bytes),
        (v_buf, buf_plan.v_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        reshape_for_pipe,
        reshape_v,
        dispatch.reshape_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // Attention core: q, k, v, mask -> attn_out
    match
      gpu_dispatch(device, attn_pipe, attn_group, dispatch.attention_dispatch_x) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // Reshape from heads: attn_out -> attn_proj
    match
      gpu_dispatch(
        device,
        reshape_from_pipe,
        reshape_from,
        dispatch.reshape_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // Output projection: attn_proj @ w_o -> x_tmp
    let out_proj_group = match
      gpu_create_group(device, mm_layout, [
        (attn_proj_buf, buf_plan.attn_proj_bytes),
        (lb.w_o, buf_plan.w_o_bytes),
        (x_tmp_buf, buf_plan.x_tmp_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        mm_attn_pipe,
        out_proj_group,
        dispatch.out_proj_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // Residual: x = x_residual + x_tmp
    let residual1_group = match
      gpu_create_group(device, add_layout, [
        (x_res_buf, buf_plan.x_residual_bytes),
        (x_tmp_buf, buf_plan.x_tmp_bytes),
        (x_buf, buf_plan.x_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        add_pipe,
        residual1_group,
        dispatch.x_elementwise_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // copy x -> x_residual (for FFN residual)
    match
      gpu_dispatch(
        device,
        copy_pipe,
        copy_x_to_res,
        dispatch.x_elementwise_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // LayerNorm2: x -> x_tmp
    let ln2_group = match
      gpu_create_group(device, ln_layout, [
        (x_buf, buf_plan.x_bytes),
        (lb.ln2_g, buf_plan.ln2_gamma_bytes),
        (lb.ln2_b, buf_plan.ln2_beta_bytes),
        (x_tmp_buf, buf_plan.x_tmp_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(device, ln_pipe, ln2_group, dispatch.layer_norm_dispatch_x) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // FFN up: x_tmp @ ff_w1 -> ff_h
    let ffn_up_group = match
      gpu_create_group(device, mm_layout, [
        (x_tmp_buf, buf_plan.x_tmp_bytes),
        (lb.ff_w1, buf_plan.ff_w1_bytes),
        (ff_h_buf, buf_plan.ff_h_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        mm_ffn_up_pipe,
        ffn_up_group,
        dispatch.ffn_up_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // Add bias: ff_h += ff_b1
    let bias_up_group = match
      gpu_create_group(device, bias_layout, [
        (ff_h_buf, buf_plan.ff_h_bytes),
        (lb.ff_b1, buf_plan.ff_b1_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        bias_up_pipe,
        bias_up_group,
        dispatch.ff_h_elementwise_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // GELU: ff_h = gelu(ff_h)
    match
      gpu_dispatch(
        device,
        gelu_pipe,
        gelu_group,
        dispatch.ff_h_elementwise_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // FFN down: ff_h @ ff_w2 -> x_tmp
    let ffn_down_group = match
      gpu_create_group(device, mm_layout, [
        (ff_h_buf, buf_plan.ff_h_bytes),
        (lb.ff_w2, buf_plan.ff_w2_bytes),
        (x_tmp_buf, buf_plan.x_tmp_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        mm_ffn_down_pipe,
        ffn_down_group,
        dispatch.ffn_down_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // Add bias: x_tmp += ff_b2
    let bias_down_group = match
      gpu_create_group(device, bias_layout, [
        (x_tmp_buf, buf_plan.x_tmp_bytes),
        (lb.ff_b2, buf_plan.ff_b2_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        bias_down_pipe,
        bias_down_group,
        dispatch.x_elementwise_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
    // Residual: x = x_residual + x_tmp
    let residual2_group = match
      gpu_create_group(device, add_layout, [
        (x_res_buf, buf_plan.x_residual_bytes),
        (x_tmp_buf, buf_plan.x_tmp_bytes),
        (x_buf, buf_plan.x_bytes),
      ]) {
      Ok(v) => v
      Err(err) => return Err(err)
    }
    match
      gpu_dispatch(
        device,
        add_pipe,
        residual2_group,
        dispatch.x_elementwise_dispatch_x,
      ) {
      Ok(_) => ()
      Err(err) => return Err(err)
    }
  }
  // Final LayerNorm: x -> x_tmp
  let ln_final_group = match
    gpu_create_group(device, ln_layout, [
      (x_buf, buf_plan.x_bytes),
      (ln_final_g_buf, buf_plan.ln_final_gamma_bytes),
      (ln_final_b_buf, buf_plan.ln_final_beta_bytes),
      (x_tmp_buf, buf_plan.x_tmp_bytes),
    ]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  match
    gpu_dispatch(
      device,
      ln_pipe,
      ln_final_group,
      dispatch.layer_norm_dispatch_x,
    ) {
    Ok(_) => ()
    Err(err) => return Err(err)
  }
  // LM head: x_tmp @ lm_head -> logits
  let lm_head_group = match
    gpu_create_group(device, mm_layout, [
      (x_tmp_buf, buf_plan.x_tmp_bytes),
      (lm_head_buf, buf_plan.lm_head_bytes),
      (logits_buf, buf_plan.logits_bytes),
    ]) {
    Ok(v) => v
    Err(err) => return Err(err)
  }
  match
    gpu_dispatch(
      device,
      mm_lm_head_pipe,
      lm_head_group,
      dispatch.lm_head_dispatch_x,
    ) {
    Ok(_) => ()
    Err(err) => return Err(err)
  }
  // Read back logits
  let logits_bytes = match
    @wgpu.device_read_buffer_bytes(device, logits_buf, buf_plan.logits_bytes) {
    Ok(b) => b
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  Ok(bytes_to_f32_array(logits_bytes))
}

///|
struct TransformerLayerBufs {
  ln1_g : @wgpu.Buffer
  ln1_b : @wgpu.Buffer
  w_q : @wgpu.Buffer
  w_k : @wgpu.Buffer
  w_v : @wgpu.Buffer
  w_o : @wgpu.Buffer
  ln2_g : @wgpu.Buffer
  ln2_b : @wgpu.Buffer
  ff_w1 : @wgpu.Buffer
  ff_b1 : @wgpu.Buffer
  ff_w2 : @wgpu.Buffer
  ff_b2 : @wgpu.Buffer
}

///|
fn gpu_create_buf(
  device : @wgpu.Device,
  size : Int,
  usage : @wgpu.BufferUsage,
  label : String,
) -> Result[@wgpu.Buffer, String] {
  match @wgpu.device_create_buffer(device, buffer_desc(size, usage, label)) {
    Ok(b) => Ok(b)
    Err(err) => Err(wgpu_error_to_string(err))
  }
}

///|
fn gpu_write_f32(
  queue : @wgpu.Queue,
  buf : @wgpu.Buffer,
  data : Array[Float],
) -> Unit {
  ignore(
    @wgpu.queue_write_buffer(
      queue,
      buf,
      0,
      float_array_to_bytes(data).to_array(),
    ),
  )
}

///|
fn gpu_write_u32(
  queue : @wgpu.Queue,
  buf : @wgpu.Buffer,
  data : Array[Int],
) -> Unit {
  ignore(
    @wgpu.queue_write_buffer(
      queue,
      buf,
      0,
      labels_array_to_bytes(data).to_array(),
    ),
  )
}

///|
fn gpu_create_shader(
  device : @wgpu.Device,
  wgsl : String,
) -> Result[@wgpu.ShaderModule, String] {
  match
    @wgpu.device_create_shader_module(
      device,
      @wgpu.shader_module_descriptor(wgsl, None),
    ) {
    Ok(m) => Ok(m)
    Err(err) => Err(wgpu_error_to_string(err))
  }
}

///|
fn gpu_create_layout(
  device : @wgpu.Device,
  stage : @wgpu.ShaderStageFlags,
  types : Array[@wgpu.BindingType],
) -> Result[@wgpu.BindGroupLayout, String] {
  let entries = Array::makei(types.length(), fn(i) {
    @wgpu.bind_group_layout_entry(i, stage, types[i])
  })
  match
    @wgpu.device_create_bind_group_layout(
      device,
      @wgpu.bind_group_layout_descriptor(entries, None),
    ) {
    Ok(l) => Ok(l)
    Err(err) => Err(wgpu_error_to_string(err))
  }
}

///|
fn gpu_create_group(
  device : @wgpu.Device,
  layout : @wgpu.BindGroupLayout,
  bufs : Array[(@wgpu.Buffer, Int)],
) -> Result[@wgpu.BindGroup, String] {
  let entries = Array::makei(bufs.length(), fn(i) {
    let (buf, size) = bufs[i]
    @wgpu.bind_group_entry(i, @wgpu.binding_resource_buffer(buf, 0, size))
  })
  match
    @wgpu.device_create_bind_group(
      device,
      @wgpu.bind_group_descriptor(layout, entries, None),
    ) {
    Ok(g) => Ok(g)
    Err(err) => Err(wgpu_error_to_string(err))
  }
}

///|
fn gpu_create_pipeline(
  device : @wgpu.Device,
  layout : @wgpu.BindGroupLayout,
  shader : @wgpu.ShaderModule,
) -> Result[@wgpu.ComputePipeline, String] {
  let pl = match
    @wgpu.device_create_pipeline_layout(
      device,
      @wgpu.pipeline_layout_descriptor([layout], None),
    ) {
    Ok(p) => p
    Err(err) => return Err(wgpu_error_to_string(err))
  }
  match
    @wgpu.device_create_compute_pipeline(
      device,
      @wgpu.compute_pipeline_descriptor(
        pl,
        @wgpu.programmable_stage(shader, "main"),
        None,
      ),
    ) {
    Ok(p) => Ok(p)
    Err(err) => Err(wgpu_error_to_string(err))
  }
}

///|
fn gpu_dispatch(
  device : @wgpu.Device,
  pipeline : @wgpu.ComputePipeline,
  group : @wgpu.BindGroup,
  dispatch_x : Int,
) -> Result[Unit, String] {
  match @wgpu.device_dispatch_compute(device, pipeline, group, dispatch_x) {
    Ok(_) => Ok(())
    Err(err) => Err(wgpu_error_to_string(err))
  }
}

///|
/// Run GPU transformer inference and compare with CPU
async fn gpu_transformer_test() -> Unit {
  println("=== GPU Transformer Inference Test ===")
  // Small model for testing
  let config = match @tensor.transformer_config(8, 8, 2, 1, 32, 16) {
    Ok(c) => c
    Err(err) => {
      println("config error: " + err)
      return
    }
  }
  let spec = match @nn.transformer_spec_new(8, 8, 2, 1, 32, 16, 2, 4) {
    Ok(s) => s
    Err(err) => {
      println("spec error: " + err.to_string())
      return
    }
  }
  let params = @tensor.transformer_init_params(config, 42)
  let tokens : Array[Array[Int]] = [[0, 1, 2, 3], [4, 5, 6, 7]]
  // CPU forward pass
  let cpu_mask = match @tensor.causal_mask(4) {
    Ok(m) => m
    Err(err) => {
      println("mask error: " + err)
      return
    }
  }
  let cpu_logits = match
    @tensor.transformer_forward(tokens, params, config, Some(cpu_mask)) {
    Ok(t) => t
    Err(err) => {
      println("CPU forward error: " + err)
      return
    }
  }
  let cpu_data = cpu_logits.contiguous().data
  println("CPU logits (first 8): " + format_floats(cpu_data, 8))
  // GPU forward pass
  let workgroup_size = 64
  let gpu_logits = match
    gpu_transformer_forward(spec, params, tokens, workgroup_size) {
    Ok(l) => l
    Err(err) => {
      println("GPU forward error: " + err)
      return
    }
  }
  println("GPU logits (first 8): " + format_floats(gpu_logits, 8))
  // Compare
  let total = cpu_data.length()
  if total != gpu_logits.length() {
    println(
      "FAIL: length mismatch: cpu=" +
      total.to_string() +
      " gpu=" +
      gpu_logits.length().to_string(),
    )
    return
  }
  let mut max_diff = Float::from_int(0)
  let mut diff_count = 0
  let tolerance = Float::from_double(0.001)
  for i = 0; i < total; i = i + 1 {
    let diff = cpu_data[i] - gpu_logits[i]
    let abs_diff = if diff < Float::from_int(0) {
      Float::from_int(0) - diff
    } else {
      diff
    }
    if abs_diff > max_diff {
      max_diff = abs_diff
    }
    if abs_diff > tolerance {
      diff_count = diff_count + 1
      if diff_count <= 5 {
        println(
          "  diff[" +
          i.to_string() +
          "]: cpu=" +
          cpu_data[i].to_string() +
          " gpu=" +
          gpu_logits[i].to_string() +
          " diff=" +
          abs_diff.to_string(),
        )
      }
    }
  }
  println(
    "max_diff=" +
    max_diff.to_string() +
    " diff_count=" +
    diff_count.to_string() +
    "/" +
    total.to_string(),
  )
  if diff_count == 0 {
    println("PASS: GPU and CPU results match within tolerance")
  } else {
    println("FAIL: " + diff_count.to_string() + " elements exceed tolerance")
  }
}

///|
fn format_floats(data : Array[Float], n : Int) -> String {
  let sb = StringBuilder::new()
  sb.write_char('[')
  let count = if n < data.length() { n } else { data.length() }
  for i = 0; i < count; i = i + 1 {
    if i > 0 {
      sb.write_string(", ")
    }
    sb.write_string(data[i].to_string())
  }
  sb.write_char(']')
  sb.to_string()
}
