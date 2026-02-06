///|
pub fn mlp_eval_from_bytes(
  spec : @nn.MlpSpec,
  bytes : Bytes,
  dataset : @nn.MlpDataset,
) -> Result[@nn.MlpTrainMetrics, @nn.MlpError] {
  @nn.mlp_params_from_bytes(spec, bytes).bind(fn(params) {
    mlp_eval(spec, params, dataset)
  })
}
