# MoonBit Project Commands

# Default target (native)
target := "native"

# Default task: check and test
default: check test

# Format code
fmt:
    moon fmt

# Type check
check:
    moon check --target {{target}}

# Run tests
test:
    moon test --target {{target}}

# Update snapshot tests
test-update:
    moon test --update --target {{target}}

# Generate type definition files
info:
    moon info --target {{target}}

# Run GPT-2 training demo
run *ARGS:
    moon run --target native src/transformer-bench -- {{ARGS}}

# Transformer LM benchmark
bench-transformer-lm *ARGS:
    moon run --target native src/transformer-bench -- {{ARGS}}

bench-transformer-lm-sweep *ARGS:
    moon run --target native src/transformer-bench -- --sweep {{ARGS}}

# OpenWebText preprocessing (parquet -> GPT token shards)
openwebtext-shard *ARGS:
    uv run --with pyarrow --with tiktoken --with numpy python3 scripts/openwebtext_to_shards.py {{ARGS}}

openwebtext-shard-test:
    uv run --with pyarrow --with tiktoken --with numpy --with pytest pytest -q scripts/openwebtext_to_shards_test.py

# Clean build artifacts
clean:
    moon clean

# Pre-release check
release-check: fmt info check test
