///|
fn forward_single(
  spec : @nn.MlpSpec,
  params : @nn.MlpParams,
  dataset : @nn.MlpDataset,
  index : Int,
  hidden : Array[Float],
  logits : Array[Float],
) -> Unit {
  let input_offset = index * spec.input_size
  let input = @numbt.vec_view(dataset.inputs, input_offset, spec.input_size)
  let hidden_vec = @numbt.vec_view(hidden, 0, spec.hidden_size)
  let logits_vec = @numbt.vec_view(logits, 0, spec.output_size)
  let weight1 = @numbt.mat_view(
    params.weight1,
    spec.input_size,
    spec.hidden_size,
  )
  let weight2 = @numbt.mat_view(
    params.weight2,
    spec.hidden_size,
    spec.output_size,
  )
  let bias1 = @numbt.vec_view(params.bias1, 0, spec.hidden_size)
  let bias2 = @numbt.vec_view(params.bias2, 0, spec.output_size)
  @numbt.matmul_vec_bias_relu_into(weight1, input, bias1, output=hidden_vec)
  @numbt.matmul_vec_bias_into(weight2, hidden_vec, bias2, output=logits_vec)
}

///|
pub fn mlp_eval(
  spec : @nn.MlpSpec,
  params : @nn.MlpParams,
  dataset : @nn.MlpDataset,
) -> Result[@nn.MlpTrainMetrics, @nn.MlpError] {
  @nn.mlp_validate_spec(spec)
  .bind(fn(_) { @nn.mlp_validate_params(spec, params) })
  .bind(fn(_) { @nn.dataset_validate_for_spec(spec, dataset) })
  .map(fn(_) {
    let hidden = Array::make(spec.hidden_size, Float::from_int(0))
    let logits = Array::make(spec.output_size, Float::from_int(0))
    let probs = Array::make(spec.output_size, Float::from_int(0))
    let logits_vec = @numbt.vec_view(logits, 0, spec.output_size)
    let probs_vec = @numbt.vec_view(probs, 0, spec.output_size)
    let mut loss_sum = Float::from_int(0)
    let mut correct = 0
    for i = 0; i < dataset.count; i = i + 1 {
      forward_single(spec, params, dataset, i, hidden, logits)
      @numbt.softmax_into(input=logits_vec, output=probs_vec)
      let label = dataset.labels[i]
      loss_sum = loss_sum + @numbt.cross_entropy_loss(probs_vec, label)
      let pred = @numbt.vec_argmax(logits_vec)
      if pred == label {
        correct = correct + 1
      }
    }
    let count_f = Float::from_int(dataset.count)
    let loss = loss_sum / count_f
    let accuracy = Float::from_int(correct) / count_f
    @nn.mlp_train_metrics(loss, accuracy)
  })
}

///|
fn lcg_next(state : UInt) -> UInt {
  let a = Int::reinterpret_as_uint(1664525)
  let c = Int::reinterpret_as_uint(1013904223)
  state * a + c
}

///|
fn shuffle_indices(indices : Array[Int], seed : Int) -> Unit {
  let mut state = Int::reinterpret_as_uint(seed)
  let mut i = indices.length() - 1
  while i > 0 {
    state = lcg_next(state)
    let j = (state % Int::reinterpret_as_uint(i + 1)).reinterpret_as_int()
    let tmp = indices[i]
    indices[i] = indices[j]
    indices[j] = tmp
    i = i - 1
  }
}

///|
pub fn mlp_train(
  spec : @nn.MlpSpec,
  params : @nn.MlpParams,
  dataset : @nn.MlpDataset,
  config : @nn.MlpTrainConfig,
) -> Result[@nn.MlpTrainResult, @nn.MlpError] {
  if config.epochs <= 0 {
    return Err(@nn.mlp_error_invalid_spec("epochs must be > 0"))
  }
  if config.batch_size <= 0 {
    return Err(@nn.mlp_error_invalid_spec("batch_size must be > 0"))
  }
  @nn.mlp_validate_spec(spec)
  .bind(fn(_) { @nn.mlp_validate_params(spec, params) })
  .bind(fn(_) { @nn.dataset_validate_for_spec(spec, dataset) })
  .map(fn(_) {
    let params = params
    let metrics : Array[@nn.MlpTrainMetrics] = []
    let indices = Array::makei(dataset.count, fn(i) { i })
    let zero = Float::from_int(0)
    let hidden = Array::make(spec.hidden_size, zero)
    let logits = Array::make(spec.output_size, zero)
    let probs = Array::make(spec.output_size, zero)
    let logits_vec = @numbt.vec_view(logits, 0, spec.output_size)
    let probs_vec = @numbt.vec_view(probs, 0, spec.output_size)
    let grad_w1 = Array::make(params.weight1.length(), zero)
    let grad_b1 = Array::make(params.bias1.length(), zero)
    let grad_w2 = Array::make(params.weight2.length(), zero)
    let grad_b2 = Array::make(params.bias2.length(), zero)
    for _epoch = 0; _epoch < config.epochs; _epoch = _epoch + 1 {
      if config.shuffle {
        shuffle_indices(indices, config.seed + _epoch)
      }
      let mut loss_sum = Float::from_int(0)
      let mut correct = 0
      let mut start = 0
      while start < dataset.count {
        let end = if start + config.batch_size < dataset.count {
          start + config.batch_size
        } else {
          dataset.count
        }
        let batch_count = end - start
        Array::fill(grad_w1, zero)
        Array::fill(grad_b1, zero)
        Array::fill(grad_w2, zero)
        Array::fill(grad_b2, zero)
        for bi = 0; bi < batch_count; bi = bi + 1 {
          let idx = indices[start + bi]
          forward_single(spec, params, dataset, idx, hidden, logits)
          @numbt.softmax_into(input=logits_vec, output=probs_vec)
          let label = dataset.labels[idx]
          loss_sum = loss_sum + @numbt.cross_entropy_loss(probs_vec, label)
          let pred = @numbt.vec_argmax(logits_vec)
          if pred == label {
            correct = correct + 1
          }
          probs[label] = probs[label] - Float::from_int(1)
          for k = 0; k < spec.output_size; k = k + 1 {
            grad_b2[k] = grad_b2[k] + probs[k]
          }
          for j = 0; j < spec.hidden_size; j = j + 1 {
            let h = hidden[j]
            for k = 0; k < spec.output_size; k = k + 1 {
              let idx_w2 = j * spec.output_size + k
              grad_w2[idx_w2] = grad_w2[idx_w2] + h * probs[k]
            }
          }
          let input_offset = idx * spec.input_size
          for j = 0; j < spec.hidden_size; j = j + 1 {
            let mut acc = Float::from_int(0)
            for k = 0; k < spec.output_size; k = k + 1 {
              acc = acc + probs[k] * params.weight2[j * spec.output_size + k]
            }
            let dz1 = acc * @numbt.relu_grad(hidden[j])
            grad_b1[j] = grad_b1[j] + dz1
            for i = 0; i < spec.input_size; i = i + 1 {
              let idx_w1 = i * spec.hidden_size + j
              grad_w1[idx_w1] = grad_w1[idx_w1] +
                dataset.inputs[input_offset + i] * dz1
            }
          }
        }
        let scale = config.learning_rate / Float::from_int(batch_count)
        for i = 0; i < params.weight1.length(); i = i + 1 {
          params.weight1[i] = params.weight1[i] - grad_w1[i] * scale
        }
        for i = 0; i < params.bias1.length(); i = i + 1 {
          params.bias1[i] = params.bias1[i] - grad_b1[i] * scale
        }
        for i = 0; i < params.weight2.length(); i = i + 1 {
          params.weight2[i] = params.weight2[i] - grad_w2[i] * scale
        }
        for i = 0; i < params.bias2.length(); i = i + 1 {
          params.bias2[i] = params.bias2[i] - grad_b2[i] * scale
        }
        start = end
      }
      let count_f = Float::from_int(dataset.count)
      let loss = loss_sum / count_f
      let accuracy = Float::from_int(correct) / count_f
      metrics.push(@nn.mlp_train_metrics(loss, accuracy))
    }
    @nn.mlp_train_result(params, metrics)
  })
}
