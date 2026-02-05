# wgpu MNIST トレーニング高速化プラン (PyTorch 同等目標)

## 背景 / 現状
- 対象: MNIST MLP 784-128-10, batch=128, epochs=20, lr=0.1
- 現状ベンチ (2026-02-05)
  - wgpu(native GPU, bench-no-readback): train_ms=93000 / test acc=0.9556
  - MoonBit CPU: train_ms=1810000 / test acc=0.9548
  - PyTorch CPU: train_ms=3311 / test acc=0.9766
  - PyTorch MPS: train_ms=9524 / test acc=0.9762

## 目標
- PyTorch CPU と同等 (±10%) もしくは PyTorch MPS と同等の学習時間を達成
- 目標指標: train_ms <= 10000 (まずは MPS 並み) → 最終 3500 付近
- 精度は test acc 0.95 以上を維持

## 想定ボトルネック
- CPU-GPU 同期や readback コスト (loss/acc 集計)
- バッチ毎の小さな dispatch が多く、起動オーバーヘッドが支配的
- 行列演算が未最適化 (タイル化・共有メモリ・ベクトル化不足)
- バッチごとの入力転送が支配的 (GPU常駐になっていない)

## 実装計画

### Phase 0: 計測と再現性
- [ ] 同一条件でのベンチ基準を固定 (seed, batch, epochs)
- [ ] train/infer で JSON ログを出し、比較しやすいスクリプトを用意
- [ ] `bench-no-readback` と通常訓練の差分を計測 (同期コスト見積り)

### Phase 1: データ常駐化と同期削減
- [ ] 训练データを GPU バッファへまとめて一括 upload (inputs/labels)
- [ ] バッチ分割・シャッフルを GPU 側で行う (index buffer)
- [ ] loss/acc の readback を epoch 末に限定 (途中は GPU のまま)
- [ ] 必要な場合は loss の reduce を GPU で行い、1値だけ readback

### Phase 2: カーネル統合と起動回数削減
- [ ] Forward (W1+ReLU, W2) を 1〜2 パスに統合
- [ ] loss + backward (grad) の統合カーネル化
- [ ] Update (SGD) を grad カーネル内で融合 (in-place update)
- [ ] 1バッチあたりの dispatch 数を現在の 12+ から 3-4 程度へ削減

### Phase 3: 行列演算の最適化
- [ ] matmul のタイル化 (workgroup 内で共有メモリ使用)
- [ ] workgroup サイズの探索 (32/64/128 など)
- [ ] vec4 / packed 型でのロード・演算に切替
- [ ] weight のレイアウト最適化 (row-major/col-major 揃え)

### Phase 4: 精度とメモリ帯域の最適化
- [ ] FP16/FP32 混合精度の検証 (Metal での最適化期待)
- [ ] activation の圧縮 (FP16) で帯域を削減
- [ ] 可能なら fused multiply-add の利用

### Phase 5: 追加最適化
- [ ] バッチサイズ調整 (256/512 で GPU を飽和させる)
- [ ] 余りバッチ処理の整理 (drop vs pad)
- [ ] wgpu-native の pipeline / bindgroup の再利用を徹底

## 検証手順
- 同一データセット・ハイパラで train_ms / acc を比較
- 各フェーズの変更で 10% 以上の改善が見込めるかを判定
- 最終的に PyTorch CPU/MPS と比較し、目標達成 여부を判断

## 期待される到達点
- Phase 1-2 で 5-10 倍
- Phase 3-4 で 2-3 倍
- 合計 10-30 倍の改善を見込み、PyTorch MPS 付近へ

