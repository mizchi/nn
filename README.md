# mizchi/nn

MoonBit-based GPT-2 training demo and tensor utilities.

このリポジトリは、mizchi の勉強用に作っている実験的な POC（MoonBit 製の機械学習フレームワーク）です。  
動作・互換性・性能についての保証はありません。

## Scope

This repository is now focused on:

- `mizchi/nn/tensor` (tensor + transformer training kernels)
- `mizchi/nn/transformer-bench` (GPT-2 style language-model training demo)

JS / Node / Playwright related assets were removed.

## Quick Commands

```bash
just           # check + test (native)
just fmt       # format code
just check     # type check (native)
just test      # run tests (native)
just info      # generate mbti files (native)
just run --steps 1 --warmup 0 --repeat 1 --batch-size 1 --seq-len 8 --d-model 16 --heads 2 --layers 1 --d-ff 32
```

## GPT-2 Training Demo

Run directly:

```bash
moon run --target native src/transformer-bench -- --steps 40 --warmup 5
```

## OpenWebText Sharding

```bash
just openwebtext-shard \
  --input-dir ~/data/lm/openwebtext/plain_text \
  --output-dir ~/data/lm/openwebtext_gpt2 \
  --tokens-per-shard 8388608
```

## NumPy Comparison Bench (Reference)

`numbt` と NumPy の比較ベンチ（既存の参考値）:

| Operation | numbt (ms) | NumPy (ms) | Notes |
|-----------|-----------|------------|-------|
| vec_add | <0.001 | <0.001 | Element-wise |
| vec_dot | <0.001 | <0.001 | BLAS sdot |
| vec_exp | <0.001 | 0.001 | vForce |
| vec_sort | <0.001 | 0.006 | vDSP_vsort |
| mat_matmul (128x784 @ 784x128) | ~0.02 | 0.019 | BLAS sgemm |
| fmat_inv (100x100) | <0.001 | 0.067 | LAPACK sgetrf/sgetri |
| fmat_svd (100x100) | <0.001 | - | LAPACK sgesvd |
| fmat_cholesky (100x100) | <0.001 | - | LAPACK spotrf |
| fmat_qr (100x100) | <0.001 | - | LAPACK sgeqrf |

NumPy 側ベンチは以下で実行できます:

```bash
uv run bench/numpy_bench.py
```

## License

Apache License 2.0 (`LICENSE`)
