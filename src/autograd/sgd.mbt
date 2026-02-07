///|
/// SGD optimizer step: param -= lr * grad for each parameter.
/// Uses BLAS saxpy: param += (-lr) * grad
pub fn sgd_step(params : Array[Var], lr : Float) -> Unit {
  let neg_lr = Float::from_int(0) - lr
  for p in params {
    match p.grad() {
      Some(g) => {
        let data = p.data()
        let gc = g.contiguous()
        @tensor.saxpy(data.data.length(), neg_lr, gc.data, data.data)
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
