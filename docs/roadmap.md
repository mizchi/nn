# Transformer Implementation Roadmap

## Goal
Implement a minimal Transformer in MoonBit with WebGPU backend.

## Current State
- MatMul (WGSL gemm)
- Softmax (batch version)
- ReLU
- Linear layer (MLP)
- Cross-entropy loss

## Missing Components

### Priority High: Kernels
- LayerNorm
- GELU
- Scaled Dot-Product Attention
- Causal Mask
- Transpose/Permute

### Priority High: Structure
- Embedding layer
- Positional Encoding (sinusoidal or learned)
- Multi-Head Attention
- Residual Connection

### Priority Medium: Inference Optimization
- KV Cache

## Minimal Tensor Abstraction

```moonbit
struct Tensor {
  shape : Array[Int]      // [batch, seq, hidden] etc.
  buffer : BufferHandle   // GPU buffer reference
}

// Required operations (~10)
fn matmul(a: Tensor, b: Tensor) -> Tensor
fn softmax(x: Tensor, dim: Int) -> Tensor
fn layer_norm(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor
fn gelu(x: Tensor) -> Tensor
fn transpose(x: Tensor, dim0: Int, dim1: Int) -> Tensor
fn reshape(x: Tensor, shape: Array[Int]) -> Tensor
fn add(a: Tensor, b: Tensor) -> Tensor  // for residual
fn scale(x: Tensor, s: Float) -> Tensor // 1/sqrt(d_k)
fn mask_fill(x: Tensor, mask: Tensor, value: Float) -> Tensor
fn embedding(indices: Tensor, weight: Tensor) -> Tensor
```

## Implementation Phases

### Phase 1: Single-Head Attention (Validation)
- [x] Tensor abstraction (`src/tensor/`)
- [x] LayerNorm kernel (`tensor_layer_norm`)
- [x] GELU kernel (`tensor_gelu`)
- [x] Scaled Dot-Product Attention kernel (`attention`)
- [x] Causal mask (`causal_mask`)
- [x] Multi-Head Attention (`multi_head_attention`)
- [x] Tests (22 tests passing)

### Phase 2: Multi-Head + FFN
- [ ] Transpose/Reshape (head split)
- [ ] Multi-Head Attention assembly
- [ ] Single Transformer Block

### Phase 3: Stacking
- [ ] Embedding + Positional
- [ ] N-layer stack
- [ ] Small LM (char-level etc.)

### Phase 4: Inference Optimization
- [ ] KV Cache
