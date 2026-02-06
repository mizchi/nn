///|
pub struct MlpCpuResult {
  hidden : Array[Float]
  output : Array[Float]
} derive(Show, Eq)

///|
pub fn mlp_forward(
  spec : MlpSpec,
  params : MlpParams,
  input : Array[Float],
) -> Result[MlpCpuResult, MlpError] {
  mlp_validate_spec(spec)
  .bind(fn(_) { mlp_validate_params(spec, params) })
  .bind(fn(_) { mlp_validate_input(spec, input) })
  .map(fn(_) {
    let zero = Float::from_int(0)
    let hidden = Array::make(elements_hidden(spec), zero)
    let output = Array::make(elements_output(spec), zero)
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
    for b = 0; b < spec.batch_size; b = b + 1 {
      let input_offset = b * spec.input_size
      let hidden_offset = b * spec.hidden_size
      let output_offset = b * spec.output_size
      let input_vec = @numbt.vec_view(input, input_offset, spec.input_size)
      let hidden_vec = @numbt.vec_view(hidden, hidden_offset, spec.hidden_size)
      let output_vec = @numbt.vec_view(output, output_offset, spec.output_size)
      @numbt.matmul_vec_bias_relu_into(
        weight1,
        input_vec,
        bias1,
        output=hidden_vec,
      )
      @numbt.matmul_vec_bias_into(weight2, hidden_vec, bias2, output=output_vec)
    }
    { hidden, output }
  })
}
