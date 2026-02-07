///|
/// SGD optimizer step: param -= lr * grad for each parameter.
pub fn sgd_step(params : Array[Var], lr : Float) -> Unit {
  for p in params {
    match p.grad() {
      Some(g) => {
        let data = p.data()
        let gc = g.contiguous()
        for i = 0; i < data.data.length(); i = i + 1 {
          data.data[i] = data.data[i] - lr * gc.data[i]
        }
      }
      None => ()
    }
  }
}

///|
/// Zero out all gradients.
pub fn zero_grad(params : Array[Var]) -> Unit {
  for p in params {
    p.tape.nodes[p.id].grad = None
  }
}
