///|
pub fn mlp_eval_from_bytes(
  spec : MlpSpec,
  bytes : Bytes,
  dataset : MlpDataset
) -> Result[MlpTrainMetrics, MlpError] {
  mlp_params_from_bytes(spec, bytes).bind(fn(params) {
    mlp_eval(spec, params, dataset)
  })
}
