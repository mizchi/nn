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
- [x] Transpose/Reshape (head split) (`reshape_for_heads`, `reshape_from_heads`)
- [x] Multi-Head Attention assembly (`multi_head_attention`)
- [x] Feed-Forward Network (`feed_forward`)
- [x] Single Transformer Block (`transformer_block`)
- [x] Embedding + Positional (`embedding`, `add_positional_embedding`)
- [x] N-layer stack (`transformer_forward`)
- [x] Parameter initialization (`transformer_init_params`)
- [x] Tests (28 tests passing)

### Phase 3: Small LM
- [ ] Tokenizer (char-level or BPE)
- [ ] Greedy/sampling generation
- [ ] Training loop (cross-entropy loss)
- [ ] Small LM demo (char-level)

### Phase 4: Inference Optimization
- [ ] KV Cache
