///|
pub struct MlpDataset {
  input_size : Int
  count : Int
  inputs : Array[Float]
  labels : Array[Int]
} derive(Show, Eq)

///|
pub fn mlp_dataset_new(
  input_size : Int,
  inputs : Array[Float],
  labels : Array[Int],
) -> Result[MlpDataset, MlpError] {
  if input_size <= 0 {
    Err(InvalidSpec("input_size must be > 0"))
  } else if inputs.length() % input_size != 0 {
    Err(InvalidShape("inputs length mismatch"))
  } else {
    let count = inputs.length() / input_size
    if labels.length() != count {
      Err(InvalidShape("labels length mismatch"))
    } else {
      Ok({ input_size, count, inputs, labels })
    }
  }
}

///|
pub struct MlpTrainConfig {
  epochs : Int
  batch_size : Int
  learning_rate : Float
  shuffle : Bool
  seed : Int
} derive(Show, Eq)

///|
pub fn mlp_train_config_default() -> MlpTrainConfig {
  {
    epochs: 5,
    batch_size: 64,
    learning_rate: Float::from_int(1) / Float::from_int(100),
    shuffle: true,
    seed: 0,
  }
}

///|
pub fn mlp_train_config(
  epochs : Int,
  batch_size : Int,
  learning_rate : Float,
  shuffle : Bool,
  seed : Int,
) -> MlpTrainConfig {
  { epochs, batch_size, learning_rate, shuffle, seed }
}

///|
pub struct MlpTrainMetrics {
  loss : Float
  accuracy : Float
} derive(Show, Eq)

///|
pub fn mlp_train_metrics(loss : Float, accuracy : Float) -> MlpTrainMetrics {
  { loss, accuracy }
}

///|
pub fn mlp_train_result(
  params : MlpParams,
  metrics : Array[MlpTrainMetrics],
) -> MlpTrainResult {
  { params, metrics }
}

///|
pub struct MlpTrainResult {
  params : MlpParams
  metrics : Array[MlpTrainMetrics]
} derive(Show, Eq)

///|
fn dataset_validate_for_spec(
  spec : MlpSpec,
  dataset : MlpDataset,
) -> Result[Unit, MlpError] {
  if dataset.input_size != spec.input_size {
    Err(InvalidShape("dataset input_size mismatch"))
  } else if dataset.count <= 0 {
    Err(InvalidShape("dataset empty"))
  } else {
    for i = 0; i < dataset.labels.length(); i = i + 1 {
      let v = dataset.labels[i]
      if v < 0 || v >= spec.output_size {
        return Err(InvalidShape("label out of range"))
      }
    }
    Ok(())
  }
}

///|
fn relu_grad(x : Float) -> Float {
  let zero = Float::from_int(0)
  if x > zero {
    Float::from_int(1)
  } else {
    zero
  }
}

///|
fn softmax_into(
  logits : Array[Float],
  offset : Int,
  size : Int,
  probs : Array[Float],
) -> Unit {
  let mut max = logits[offset]
  for k = 1; k < size; k = k + 1 {
    let v = logits[offset + k]
    if v > max {
      max = v
    }
  }
  let mut sum = Float::from_int(0)
  for k = 0; k < size; k = k + 1 {
    let v = @math.expf(logits[offset + k] - max)
    probs[k] = v
    sum = sum + v
  }
  for k = 0; k < size; k = k + 1 {
    probs[k] = probs[k] / sum
  }
}

///|
fn cross_entropy_loss(probs : Array[Float], label : Int) -> Float {
  let eps = Float::from_int(1) / Float::from_int(1000000)
  let p = probs[label] + eps
  -@math.lnf(p)
}

///|
fn argmax(values : Array[Float], offset : Int, size : Int) -> Int {
  let mut best = 0
  let mut best_value = values[offset]
  for k = 1; k < size; k = k + 1 {
    let v = values[offset + k]
    if v > best_value {
      best_value = v
      best = k
    }
  }
  best
}

///|
fn forward_single(
  spec : MlpSpec,
  params : MlpParams,
  dataset : MlpDataset,
  index : Int,
  hidden : Array[Float],
  logits : Array[Float],
) -> Unit {
  let input_offset = index * spec.input_size
  let zero = Float::from_int(0)
  for j = 0; j < spec.hidden_size; j = j + 1 {
    let mut acc = params.bias1[j]
    for i = 0; i < spec.input_size; i = i + 1 {
      acc = acc +
        dataset.inputs[input_offset + i] *
        params.weight1[i * spec.hidden_size + j]
    }
    hidden[j] = if acc < zero { zero } else { acc }
  }
  for k = 0; k < spec.output_size; k = k + 1 {
    let mut acc = params.bias2[k]
    for j = 0; j < spec.hidden_size; j = j + 1 {
      acc = acc + hidden[j] * params.weight2[j * spec.output_size + k]
    }
    logits[k] = acc
  }
}

///|
pub fn mlp_eval(
  spec : MlpSpec,
  params : MlpParams,
  dataset : MlpDataset,
) -> Result[MlpTrainMetrics, MlpError] {
  mlp_validate_spec(spec)
  .bind(fn(_) { mlp_validate_params(spec, params) })
  .bind(fn(_) { dataset_validate_for_spec(spec, dataset) })
  .map(fn(_) {
    let hidden = Array::make(spec.hidden_size, Float::from_int(0))
    let logits = Array::make(spec.output_size, Float::from_int(0))
    let probs = Array::make(spec.output_size, Float::from_int(0))
    let mut loss_sum = Float::from_int(0)
    let mut correct = 0
    for i = 0; i < dataset.count; i = i + 1 {
      forward_single(spec, params, dataset, i, hidden, logits)
      softmax_into(logits, 0, spec.output_size, probs)
      let label = dataset.labels[i]
      loss_sum = loss_sum + cross_entropy_loss(probs, label)
      let pred = argmax(logits, 0, spec.output_size)
      if pred == label {
        correct = correct + 1
      }
    }
    let count_f = Float::from_int(dataset.count)
    let loss = loss_sum / count_f
    let accuracy = Float::from_int(correct) / count_f
    { loss, accuracy }
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
  spec : MlpSpec,
  params : MlpParams,
  dataset : MlpDataset,
  config : MlpTrainConfig,
) -> Result[MlpTrainResult, MlpError] {
  if config.epochs <= 0 {
    return Err(InvalidSpec("epochs must be > 0"))
  }
  if config.batch_size <= 0 {
    return Err(InvalidSpec("batch_size must be > 0"))
  }
  mlp_validate_spec(spec)
  .bind(fn(_) { mlp_validate_params(spec, params) })
  .bind(fn(_) { dataset_validate_for_spec(spec, dataset) })
  .map(fn(_) {
    let params = params
    let metrics : Array[MlpTrainMetrics] = []
    let indices = Array::makei(dataset.count, fn(i) { i })
    let zero = Float::from_int(0)
    let hidden = Array::make(spec.hidden_size, zero)
    let logits = Array::make(spec.output_size, zero)
    let probs = Array::make(spec.output_size, zero)
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
          softmax_into(logits, 0, spec.output_size, probs)
          let label = dataset.labels[idx]
          loss_sum = loss_sum + cross_entropy_loss(probs, label)
          let pred = argmax(logits, 0, spec.output_size)
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
            let dz1 = acc * relu_grad(hidden[j])
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
      metrics.push({ loss, accuracy })
    }
    { params, metrics }
  })
}
