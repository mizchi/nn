# wgpu/nn perf TODO

目的: MNIST MLP の GPU 学習を高速化し、PyTorch (MPS) に近いスループットに寄せる。
計測は `src/train` の `--bench` を基準にする。

## 実装済み
- [x] loss/correct reduce を workgroup 並列化（`wgsl_loss_reduce`）
- [x] `--batch` / `--workgroup` を train CLI に追加（チューニング用）
- [x] MatMul をタイル化して workgroup shared を使う（layer1/layer2）
- [x] loss/correct reduce + epoch accumulate を単一カーネルに融合

## 次にやる（効果が大きい順）
- [ ] カーネル融合（forward + relu + loss、backward の一部）
- [ ] バッチサイズのスイープと最適 workgroup の探索（自動チューナ）
- [ ] データ転送の削減（segment の常駐化/再利用、upload 回数削減）
- [ ] mixed precision（fp16/bf16）の導入
- [ ] GPU timing の分離計測（timestamp query / step ごとの GPU 時間計測）

## メモ
- `segment_samples` は 128MB 制限回避のために導入済み。
- readback の削減は 6% 程度なので、主因はカーネルの効率。

## パッケージ依存の整理 TODO
- [ ] mizchi/blas と mizchi/js の両方を deps に持つため、`moon check` が単一ターゲットで通らない
  - mizchi/blas: native のみ
  - mizchi/js: js のみ
- [ ] 将来的にはパッケージ分離を検討:
  - `mizchi/nn-core`: 共通コード（wgpu_common, nn のWGSL生成など）
  - `mizchi/nn-js`: WebGPU browser bindings
  - `mizchi/nn-native`: wgpu-native bindings + BLAS training
