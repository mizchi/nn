# mizchi/nn

MoonBit-based GPT-2 training demo and tensor utilities.

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
