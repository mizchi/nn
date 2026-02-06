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

## 追加予定 (優先度順)

### 1. sort / argsort ★★★★★
- ML で必須 (top-k, precision-recall など)
```moonbit
pub fn vec_sort(v : Vec) -> Vec
pub fn vec_argsort(v : Vec) -> Array[Int]
```

### 2. where / nonzero ★★★★★
- 条件処理に必須 (ReLU backward など)
```moonbit
pub fn vec_where(mask : Array[Bool], x : Vec, y : Vec) -> Vec
pub fn vec_nonzero(v : Vec) -> Array[Int]
```

### 3. cumsum / cumprod ★★★★
- 確率分布計算に有用
```moonbit
pub fn vec_cumsum(v : Vec) -> Vec
pub fn vec_cumprod(v : Vec) -> Vec
```

### 4. outer / diag / trace ★★★★
- 線形代数の基本
```moonbit
pub fn vec_outer(a : Vec, b : Vec) -> Mat
pub fn vec_diag(v : Vec) -> Mat
pub fn mat_trace(m : Mat) -> Float
```

### 5. concatenate / stack ★★★
- バッチ処理に有用
```moonbit
pub fn vec_concatenate(vecs : Array[Vec]) -> Vec
pub fn mat_vstack(mats : Array[Mat]) -> Mat
pub fn mat_hstack(mats : Array[Mat]) -> Mat
```

## ベンチマーク比較

### MLP Forward (784→128→10, batch=128)

| 実装 | 時間/batch | 備考 |
|------|-----------|------|
| NumPy (Python) | 0.034 ms | Accelerate BLAS |
| MoonBit BLAS | ~0.002 ms | 10-30x faster |

NumPy との比較実行:
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
