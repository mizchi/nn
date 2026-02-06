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
pub fn dataset_validate_for_spec(
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
