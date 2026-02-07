# Autograd XL Backward Performance Analysis

## Benchmark: MLP [256, 784] -> 1024 -> 10

| Metric | Value |
|--------|-------|
| Forward | 0.38ms |
| **Backward** | **1.27ms** |
| SGD step | 0.03ms |
| **Total** | **1.70ms** |
| Manual backward (no tape) | 0.39ms |

## Root Cause: GC Cache Pollution in Closure

### Per-operation breakdown (inside fused backward closure)

| Operation | Closure | Isolated |
|-----------|---------|----------|
| `FixedArray::make(262144)` | 12us | 0us |
| `relu_backward_managed` | **900us** | **90us** |
| `sgemm dW (1024,784,256)` | 270us | 260us |
| `accumulate_raw` | 0-1us | - |
| `bias_add_backward` | 15us | 10us |

`relu_backward_managed` reads from C-managed buffers (g_grad_buf, g_relu_pre) and writes to MoonBit FixedArray. Takes **10x longer** inside tape backward closure than when called directly.

### What's confirmed

1. **Cache aliasing between old-gen GC buffers**: Workspace buffers (ws.dw_data, ws.dx_data etc.) promoted to GC old-generation have systematic L1/L2 cache set conflicts. `relu_backward_inplace(old_gen_dy, old_gen_x, dx, n)` takes 0.8ms vs 0.07ms with fresh buffers.

2. **C-managed buffers fix the READ side**: `relu_backward_managed` reads from `g_relu_pre` and `g_grad_buf` (page-aligned posix_memalign, outside GC). When tested in isolation: 0.09ms.

3. **But closures kill performance**: The same `relu_backward_managed` call takes 0.9ms when executed inside a tape backward closure. The closure captures Tape, Tensor objects, and workspace references.

4. **Direct closure calls are fast**: An identical closure defined in a simple loop context runs in 0.38ms total (including relu + sgemm + bias).

5. **FixedArray::make is NOT the direct cause**: Removing allocation from the closure (reusing ws.y_data, or using fully C-managed g_relu_dx) does not fix the 10x relu slowdown.

### Hypothesis: GC write barrier or root scanning

The tape backward closure differs from test closures in that:
- It was created during `forward_relu_managed` and stored in `tape.nodes[].backward_fn`
- The `Tape.nodes` Array likely causes the closure to be in a different GC generation/region
- When backward runs, accessing the closure triggers GC write barriers or scanning that pollutes the CPU cache

The `relu_backward_managed` C function only does simple reads/writes to `float*` pointers. The 10x slowdown can only come from CPU cache state being destroyed before/during the function call.

## What was tried

| Approach | Result |
|----------|--------|
| BLAS trans flags (avoid transpose copies) | Already done, working |
| `accumulate_raw` (avoid Tensor alloc) | Already done, working |
| `needs_grad` skip | Already done, working |
| Fused Linear+ReLU (single tape node) | Reduces node count, but relu still slow |
| C-managed relu_pre (`posix_memalign`) | Fixes isolated relu, but not in closure |
| C-managed grad_buf (linear2 dx) | Added, grad_buf_to_fixed works |
| C-managed relu_dx (3rd buffer) | 3 page-aligned buffers alias each other |
| Offset alignment for g_relu_dx | No improvement |
| Reuse ws.y_data as relu_dx | No improvement |
| Pre-allocated relu_dx in workspace | Old-gen → same aliasing |
| Arena approach (single FixedArray) | Arena promoted to old-gen → same issue |
| Fresh alloc per step (no workspace) | GC pressure, worse performance |

## Current Architecture

```
Forward:
  linear1: sgemm_to_relu_pre → bias_add_relu_pre → relu_from_pre(ws1.y_data)
  linear2: sgemm(ws2.y_data) → bias_add_inplace(ws2.y_data)

Backward:
  cross_entropy: softmax-based gradient (tiny)
  linear2 (forward_ws_managed):
    dW: sgemm(dy, x, ws2.dw_data)
    dbias: bias_add_backward(dy, ws2.db_data)
    dx: sgemm_to_grad_buf → grad_buf_to_fixed(ws2.dx_data) → accumulate_raw
  fused_l1_relu (forward_relu_managed):
    relu_bwd: relu_backward_managed(ws1.y_data) ← reads g_grad_buf + g_relu_pre
    dW: sgemm(ws1.y_data, xd.data, ws1.dw_data)
    dbias: bias_add_backward(ws1.y_data, ws1.db_data)
```

## Files Modified

| File | Changes |
|------|---------|
| `src/tensor/blas_stub.c` | C-managed buffers (g_relu_pre, g_grad_buf, g_relu_dx), FFI functions |
| `src/tensor/blas.mbt` | MoonBit FFI wrappers for all C functions |
| `src/autograd/layer.mbt` | `forward_relu_managed`, `forward_ws_managed`, `forward_relu_ws` |
| `src/autograd-bench/main.mbt` | Extensive profiling in `profile_xl()` |

## Next Steps

1. **MoonBit GC investigation**: The 10x relu slowdown inside closures suggests the GC (write barriers, root scanning, or nursery management) is polluting CPU cache. Need MoonBit runtime team input or GC tuning options.

2. **Alternative: bypass closure entirely**: Implement manual backward (no tape/closures) for the MLP case. The manual backward takes 0.39ms. This would require a specialized training loop instead of the general autograd.

3. **Alternative: reduce closure captures**: Minimize what the backward closure captures. Pre-extract `xd.data` as a FixedArray instead of capturing the full Tensor. Avoid capturing the Tape; pass it as an argument.

4. **Alternative: C-side backward**: Move the entire fused backward logic into a single C function that takes all needed pointers. The C function would call sgemm, relu_backward, etc. without any GC interaction.

## PyTorch Comparison

```
PyTorch (MPS): ~0.022ms total
MoonBit:       ~1.70ms total (77x slower)
Manual:        ~0.39ms backward (estimated ~0.60ms total, 27x slower)
```

The 0.39ms manual backward is the theoretical minimum for the current BLAS-based approach (no GPU). The remaining gap is GPU (MPS) vs CPU (Accelerate).
