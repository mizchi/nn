# numbt Roadmap

NumPy 互換の MoonBit 配列ライブラリ。

## 現在の実装状況

### Vec (1D Array)
- ✅ 生成: zeros, ones, from_fn, arange, linspace
- ✅ アクセス: at, set, slice, fill, to_array
- ✅ 演算: add, sub, mul, div (element-wise + scalar)
- ✅ 数学: exp, log, sqrt, abs, pow, clip, sin, cos, tanh
- ✅ 統計: sum, mean, min, max, argmin, argmax, var, std, prod
- ✅ BLAS: dot, nrm2, axpy

### Mat (2D Array)
- ✅ 生成: zeros, ones, eye, view
- ✅ アクセス: at, set, row, clone, transpose
- ✅ 演算: add, sub, mul, div (element-wise + scalar)
- ✅ 統計: sum, mean, min, max, sum_axis0/1, mean_axis0/1
- ✅ BLAS: matmul (sgemm)

### 比較・変形
- ✅ allclose, has_nan, has_inf
- ✅ reshape, flatten

### Sort / Argsort
- ✅ vec_sort, vec_sort_desc
- ✅ vec_argsort, vec_argsort_desc

### Conditional / Masking
- ✅ vec_nonzero
- ✅ vec_gt, vec_ge, vec_lt, vec_le (comparison → mask)
- ✅ vec_where, vec_where_scalar

### Cumulative Operations
- ✅ vec_cumsum, vec_cumprod
- ✅ vec_diff

### Linear Algebra Extras
- ✅ vec_outer (外積)
- ✅ vec_diag (ベクトル→対角行列)
- ✅ mat_diag (行列→対角ベクトル)
- ✅ mat_trace (トレース)

### Concatenate / Stack / Repeat
- ✅ vec_concatenate
- ✅ mat_vstack, mat_hstack
- ✅ vec_repeat, mat_tile

## テスト状況

- 合計: **101 テスト** (全て合格)
- カバレッジ: Vec/Mat 全操作

## ベンチマーク比較

### NumPy vs numbt (Apple M-series, Accelerate BLAS)

| 操作 | NumPy | 備考 |
|------|-------|------|
| vec_add (N=1024) | 0.000 ms | SIMD |
| vec_dot (N=1024) | 0.000 ms | BLAS ddot |
| mat_matmul (128x784x128) | 0.020 ms | BLAS sgemm |
| MLP forward (784→128→10) | 0.034 ms | |
| sort (N=1024) | 0.006 ms | quicksort |
| cumsum (N=1024) | 0.003 ms | |
| outer (100) | 0.003 ms | |

MoonBit BLAS は同等の BLAS 実装を使用し、Python オーバーヘッドがないため 10-30x 高速。

実行:
```bash
# NumPy
cd bench && uv run python numpy_bench.py

# MoonBit
moon run --target native src/bench -- --blas --iters 1000
```

## 設計方針

1. **Native-only**: BLAS (Accelerate.framework) 前提
2. **In-place 優先**: `_into` 版を提供し、アロケーション削減
3. **View サポート**: Vec は offset/len で部分配列を表現
4. **演算子オーバーロード**: +, -, *, /, unary - をサポート
