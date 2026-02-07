///|
/// A node in the computation graph.
/// Each node holds the forward output and an optional backward function.
pub struct Node {
  data : @tensor.Tensor
  mut grad : @tensor.Tensor? // accumulated gradient (set during backward)
  requires_grad : Bool
  backward_fn : ((@tensor.Tensor) -> Unit)? // None = leaf node
}

///|
/// Tape records computation graph nodes in topological order.
pub struct Tape {
  nodes : Array[Node]
}

///|
pub fn Tape::new() -> Tape {
  Tape::{ nodes: [] }
}

///|
/// Create a leaf variable (parameter or input).
pub fn Tape::create_var(
  self : Tape,
  data : @tensor.Tensor,
  requires_grad : Bool,
) -> Var {
  let id = self.nodes.length()
  self.nodes.push(Node::{ data, grad: None, requires_grad, backward_fn: None })
  Var::{ id, tape: self }
}

///|
/// Add a computed node to the tape.
fn Tape::push_node(
  self : Tape,
  data : @tensor.Tensor,
  backward_fn : (@tensor.Tensor) -> Unit,
) -> Var {
  let id = self.nodes.length()
  self.nodes.push(Node::{
    data,
    grad: None,
    requires_grad: false,
    backward_fn: Some(backward_fn),
  })
  Var::{ id, tape: self }
}

///|
/// Run backward pass from the given output variable.
/// Sets grad=ones for the output, then traverses tape in reverse.
pub fn Tape::backward(self : Tape, output : Var) -> Unit {
  let ones = @tensor.tensor_ones(output.data().shape)
  self.nodes[output.id].grad = Some(ones)
  for i = self.nodes.length() - 1; i >= 0; i = i - 1 {
    match (self.nodes[i].grad, self.nodes[i].backward_fn) {
      (Some(g), Some(bwd)) => bwd(g)
      _ => ()
    }
  }
}

///|
/// Backward with per-node timing (for profiling).
pub fn Tape::backward_profiled(
  self : Tape,
  output : Var,
) -> Array[(Int, UInt64)] {
  let ones = @tensor.tensor_ones(output.data().shape)
  self.nodes[output.id].grad = Some(ones)
  let timings : Array[(Int, UInt64)] = []
  for i = self.nodes.length() - 1; i >= 0; i = i - 1 {
    match (self.nodes[i].grad, self.nodes[i].backward_fn) {
      (Some(g), Some(bwd)) => {
        let t0 = @tensor.clock_ns()
        bwd(g)
        let elapsed = @tensor.clock_ns() - t0
        timings.push((i, elapsed))
      }
      _ => ()
    }
  }
  timings
}

///|
/// Set gradient for a specific node (for diagnostic use).
pub fn Tape::set_grad(self : Tape, id : Int, g : @tensor.Tensor) -> Unit {
  self.nodes[id].grad = Some(g)
}

///|
/// Run backward for a single node (for diagnostic timing).
pub fn Tape::run_backward_node(self : Tape, id : Int) -> Unit {
  match (self.nodes[id].grad, self.nodes[id].backward_fn) {
    (Some(g), Some(bwd)) => bwd(g)
    _ => ()
  }
}

///|
/// Check if a node needs gradient (either it requires_grad or has a backward_fn).
fn needs_grad(tape : Tape, id : Int) -> Bool {
  let node = tape.nodes[id]
  if node.requires_grad {
    return true
  }
  match node.backward_fn {
    Some(_) => true
    None => false
  }
}

///|
/// Accumulate gradient for node at given id.
/// Uses BLAS saxpy for vectorized addition.
fn accumulate(tape : Tape, id : Int, g : @tensor.Tensor) -> Unit {
  match tape.nodes[id].grad {
    None => tape.nodes[id].grad = Some(g)
    Some(existing) => {
      let gd = g.contiguous().data
      @tensor.accumulate_inplace(existing.data, gd, existing.data.length())
    }
  }
}

///|
/// Accumulate gradient from raw FixedArray data without creating Tensor.
/// Avoids GC pressure from intermediate Tensor objects.
fn accumulate_raw(
  tape : Tape,
  id : Int,
  data : FixedArray[Float],
  shape : @tensor.Shape,
) -> Unit {
  match tape.nodes[id].grad {
    None => tape.nodes[id].grad = Some(@tensor.tensor_new_fixed(shape, data))
    Some(existing) =>
      @tensor.accumulate_inplace(existing.data, data, existing.data.length())
  }
}
