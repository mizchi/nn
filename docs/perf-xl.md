# Autograd XL Backward Performance Analysis

## Benchmark: MLP [256, 784] -> 1024 -> 10

| Metric | Value |
|--------|-------|
| Forward | 0.39-0.40ms |
| **Backward** | **1.06-1.09ms** |
| SGD step | 0.02ms |
| **Total** | **1.48-1.52ms** |
| Manual backward (no tape) | 0.39ms |

## Update 2026-02-07

### Newly confirmed bottleneck (inside fused C backward)

`tensor_fused_two_layer_relu_backward` を C 内で分解計測した結果:

| Step | Time |
|------|------|
| `dW2` | ~0.01ms |
| `db2` | ~0.001ms |
| `d_a1 = dy @ W2` | ~0.016ms |
| `relu backward` | **~0.76-0.79ms** |
| `dW1 + db1` | ~0.27ms |

`relu` 区間が支配的で、closure dispatch や `accumulate_raw` 自体ではなく、**fused C 関数内でも relu backward が重い**ことを確認した。

### Changes applied in this iteration

1. `fused_two_layer` の C 内訳プロファイル API を追加
2. C-managed バッファのオフセットを cache-line 単位で分離
3. `fused_two_layer` の relu mask 参照を `g_relu_pre` から `g_relu_out` ベースに切替

結果:
- `fused_two_layer` node: 約 `1.11-1.12ms` → 約 `1.06-1.09ms`
- fused C relu 区間: 約 `0.81-0.83ms` → 約 `0.76-0.79ms`

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
| `src/tensor/blas_stub.c` | C-managed buffers (g_relu_pre, g_grad_buf, g_relu_dx, g_relu_out), FFI functions |
| `src/tensor/blas.mbt` | MoonBit FFI wrappers for all C functions |
| `src/autograd/layer.mbt` | `forward_relu_managed`, `forward_ws_managed`, `forward_relu_ws`, `forward_two_layer_relu` |
| `src/autograd-bench/main.mbt` | Extensive profiling in `profile_xl()` |

## What was tried (continued)

| Approach | Result |
|----------|--------|
| C-side fused backward (single C call for relu+sgemm+bias) | No improvement — cache already cold before C call |
| Fused two-layer (1 tape node for linear1+relu+linear2) | Small sizes 2-3x faster, XL unchanged |
| C-managed relu_out (cache relu output for backward) | XL で小幅改善（~3-5%） |

## Next Steps

1. **relu mask の保持方法を再設計**: 現在は ReLU backward が支配的。bitmask 化（1bit/elem）や forward 時に mask を別表現で保持して、backward の read bandwidth を下げる。

2. **`fused_two_layer` の更なる C 側融合**: `d_a1` 出力形式と relu backward を連続最適化し、`g_grad_buf` と mask 参照のメモリストリーム衝突を減らす。

3. **GPT-2 向け中期準備**: 2-layer MLP の fused パターンを Transformer の FFN block (`Linear -> GELU/ReLU -> Linear`) に横展開できる API に整理する。

## Mid-Term Progress (Transformer FFN)

`Linear -> GELU -> Linear` の backward を 1 回の C 呼び出しに融合する実装を追加:

- `tensor_fused_ffn_backward` (C): `dW2/db2 -> GELU' -> dW1/db1/dx` を一括計算
- `fused_ffn_backward` (MoonBit wrapper)
- `feed_forward_backward` で fused 経路を利用
- whitebox test で参照実装との一致を検証（`1e-5` 以内）

一時ベンチ（batch=8, seq=16, d_model=64, d_ff=256, 30iter）では:

- reference: `~0.785ms`
- fused: `~0.286ms`

FFN backward 単体で約 `2.7x` 改善。GPT-2 相当の学習ループでは FFN が支配的になりやすいため、次の中期ゴールに直接効く変更。

## PyTorch Comparison

```
PyTorch (CPU): ~0.82ms total (XL)
MoonBit:       ~1.65ms total (2.0x slower)
Manual:        ~0.39ms backward (estimated ~0.60ms total, 0.73x PyTorch)
```

The 0.39ms manual backward is the theoretical minimum for the current BLAS-based approach. The gap between manual (0.39ms) and autograd (1.22ms) is due to GC old-gen cache aliasing, not algorithmic overhead.
