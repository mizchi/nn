# MoonBit Project Commands

# Default target (js for browser compatibility)
target := "js"

# Default task: check and test
default: check test

# Format code
fmt:
    moon fmt

# Type check
check:
    moon check --deny-warn --target {{target}}

# Run tests
test:
    moon test --target {{target}}

# E2E tests (Playwright)
e2e:
    pnpm e2e

e2e-install:
    pnpm e2e:install

# Benchmarks
bench-cpu *ARGS:
    moon run --target native src/bench -- {{ARGS}}

bench-gpu:
    pnpm bench:gpu

bench-transformer-lm *ARGS:
    moon run --target native src/transformer-bench -- {{ARGS}}

bench-transformer-lm-sweep *ARGS:
    moon run --target native src/transformer-bench -- --sweep {{ARGS}}

# Build wgpu-native (required for native target)
wgpu-native-build:
    bash src/scripts/build-wgpu-native.sh src/build-stamps/wgpu_native_build.stamp

# Browser demo (serves repo root for data/mnist)
serve:
    node e2e/build.mjs
    node tools/server.mjs

# MNIST (native)
mnist_primary := "http://yann.lecun.com/exdb/mnist"
mnist_mirror := "https://storage.googleapis.com/cvdf-datasets/mnist"
data_root := env_var_or_default("NN_DATA_DIR", env_var_or_default("HOME", justfile_directory()) + "/data")
mnist_dir := data_root + "/mnist"
pytorch_mnist_dir := data_root + "/pytorch_mnist"

data-migrate:
    mkdir -p data "{{data_root}}" "{{mnist_dir}}" "{{pytorch_mnist_dir}}"
    if [ -d data/mnist ] && [ ! -L data/mnist ]; then cp -a data/mnist/. "{{mnist_dir}}"/ && rm -rf data/mnist; fi
    if [ -d data/pytorch_mnist ] && [ ! -L data/pytorch_mnist ]; then cp -a data/pytorch_mnist/. "{{pytorch_mnist_dir}}"/ && rm -rf data/pytorch_mnist; fi
    ln -sfn "{{mnist_dir}}" data/mnist
    ln -sfn "{{pytorch_mnist_dir}}" data/pytorch_mnist
    @echo "MNIST data dir: {{mnist_dir}}"

mnist-download: data-migrate
    rm -f "{{mnist_dir}}"/*.gz
    curl -fL -o "{{mnist_dir}}"/train-images-idx3-ubyte.gz {{mnist_primary}}/train-images-idx3-ubyte.gz || curl -fL -o "{{mnist_dir}}"/train-images-idx3-ubyte.gz {{mnist_mirror}}/train-images-idx3-ubyte.gz
    curl -fL -o "{{mnist_dir}}"/train-labels-idx1-ubyte.gz {{mnist_primary}}/train-labels-idx1-ubyte.gz || curl -fL -o "{{mnist_dir}}"/train-labels-idx1-ubyte.gz {{mnist_mirror}}/train-labels-idx1-ubyte.gz
    curl -fL -o "{{mnist_dir}}"/t10k-images-idx3-ubyte.gz {{mnist_primary}}/t10k-images-idx3-ubyte.gz || curl -fL -o "{{mnist_dir}}"/t10k-images-idx3-ubyte.gz {{mnist_mirror}}/t10k-images-idx3-ubyte.gz
    curl -fL -o "{{mnist_dir}}"/t10k-labels-idx1-ubyte.gz {{mnist_primary}}/t10k-labels-idx1-ubyte.gz || curl -fL -o "{{mnist_dir}}"/t10k-labels-idx1-ubyte.gz {{mnist_mirror}}/t10k-labels-idx1-ubyte.gz
    gunzip -f "{{mnist_dir}}"/*.gz

mnist-train: data-migrate
    moon run --target native src/examples/mnist-train

mnist-infer *ARGS: data-migrate
    moon run --target native src/examples/mnist-infer -- {{ARGS}}

# Update snapshot tests
test-update:
    moon test --update --target {{target}}

# Run main
run:
    moon run src/main --target {{target}}

# Generate type definition files
info:
    moon info

# Clean build artifacts
clean:
    moon clean

# Golden data: generate (requires Python + numpy)
golden-gen:
    python3 tools/gen_golden.py

# Golden data: verify MoonBit forward pass against PyTorch
golden-check:
    moon run --target native src/golden-check

# GPU profiling
profile-gpu *ARGS:
    moon run --target native src/examples/mnist-train -- --gpu --profile --bench {{ARGS}}

# Benchmark comparison
bench-compare:
    @echo "=== MoonBit GPU ==="
    moon run --target native src/examples/mnist-train -- --gpu --epochs 5 --limit 1024 --bench --profile
    @echo "=== PyTorch MPS ==="
    cd ~/sandbox/torch-mnist && uv run python main.py --epochs 5 --limit 1024 --device mps --profile

# Pre-release check
release-check: fmt info check test
