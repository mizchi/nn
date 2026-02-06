#!/usr/bin/env python3
"""
NumPy benchmark for comparison with numbt (MoonBit).

Run with: uv run bench/numpy_bench.py
"""

import numpy as np
import time
from typing import Callable

def bench(name: str, fn: Callable, warmup: int = 3, runs: int = 10) -> float:
    """Run benchmark and return average time in ms."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    print(f"{name}: {avg:.3f} ms (± {std:.3f} ms)")
    return avg


def main():
    print("=" * 60)
    print("NumPy Benchmark")
    print("=" * 60)

    # Small vectors (like MNIST hidden layer)
    N_SMALL = 128
    # Medium vectors
    N_MED = 1024
    # Large vectors (like MNIST input)
    N_LARGE = 784 * 100  # 100 samples

    # Matrix sizes
    M, K, N = 128, 784, 128  # MLP: 784 -> 128
    BATCH = 128

    print(f"\n--- Vector operations (N={N_MED}) ---")
    a = np.random.randn(N_MED).astype(np.float32)
    b = np.random.randn(N_MED).astype(np.float32)

    bench("vec_add", lambda: a + b)
    bench("vec_mul", lambda: a * b)
    bench("vec_dot", lambda: np.dot(a, b))
    bench("vec_sum", lambda: np.sum(a))
    bench("vec_mean", lambda: np.mean(a))
    bench("vec_std", lambda: np.std(a))
    bench("vec_exp", lambda: np.exp(a))
    bench("vec_log (abs)", lambda: np.log(np.abs(a) + 1e-6))
    bench("vec_sqrt (abs)", lambda: np.sqrt(np.abs(a)))
    bench("vec_norm", lambda: np.linalg.norm(a))

    print(f"\n--- Large vector operations (N={N_LARGE}) ---")
    a_large = np.random.randn(N_LARGE).astype(np.float32)
    b_large = np.random.randn(N_LARGE).astype(np.float32)

    bench("vec_add_large", lambda: a_large + b_large)
    bench("vec_dot_large", lambda: np.dot(a_large, b_large))
    bench("vec_sum_large", lambda: np.sum(a_large))
    bench("vec_exp_large", lambda: np.exp(a_large))

    print(f"\n--- Matrix operations (M={M}, K={K}, N={N}) ---")
    mat_a = np.random.randn(M, K).astype(np.float32)
    mat_b = np.random.randn(K, N).astype(np.float32)

    bench("mat_matmul", lambda: mat_a @ mat_b)
    bench("mat_transpose", lambda: mat_a.T.copy())  # force copy
    bench("mat_sum", lambda: np.sum(mat_a))
    bench("mat_mean", lambda: np.mean(mat_a))
    bench("mat_sum_axis0", lambda: np.sum(mat_a, axis=0))
    bench("mat_sum_axis1", lambda: np.sum(mat_a, axis=1))

    print(f"\n--- Batch matmul (batch={BATCH}, in={K}, out={N}) ---")
    batch_input = np.random.randn(BATCH, K).astype(np.float32)
    weight = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)

    bench("batch_matmul", lambda: batch_input @ weight)
    bench("batch_matmul_bias", lambda: batch_input @ weight + bias)
    bench("batch_matmul_bias_relu", lambda: np.maximum(0, batch_input @ weight + bias))

    print(f"\n--- MLP forward (MNIST-like: 784->128->10) ---")
    INPUT_DIM = 784
    HIDDEN_DIM = 128
    OUTPUT_DIM = 10

    x = np.random.randn(BATCH, INPUT_DIM).astype(np.float32)
    w1 = np.random.randn(INPUT_DIM, HIDDEN_DIM).astype(np.float32)
    b1 = np.random.randn(HIDDEN_DIM).astype(np.float32)
    w2 = np.random.randn(HIDDEN_DIM, OUTPUT_DIM).astype(np.float32)
    b2 = np.random.randn(OUTPUT_DIM).astype(np.float32)

    def mlp_forward():
        h = np.maximum(0, x @ w1 + b1)  # ReLU
        return h @ w2 + b2

    bench("mlp_forward", mlp_forward)

    def mlp_forward_softmax():
        h = np.maximum(0, x @ w1 + b1)
        logits = h @ w2 + b2
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    bench("mlp_forward_softmax", mlp_forward_softmax)

    print(f"\n--- Array creation ---")
    bench("zeros(1000)", lambda: np.zeros(1000, dtype=np.float32))
    bench("ones(1000)", lambda: np.ones(1000, dtype=np.float32))
    bench("arange(1000)", lambda: np.arange(1000, dtype=np.float32))
    bench("linspace(0,1,1000)", lambda: np.linspace(0, 1, 1000, dtype=np.float32))
    bench("eye(100)", lambda: np.eye(100, dtype=np.float32))

    print(f"\n--- Sort operations (N={N_MED}) ---")
    arr = np.random.randn(N_MED).astype(np.float32)
    bench("sort", lambda: np.sort(arr))
    bench("argsort", lambda: np.argsort(arr))
    bench("argmax", lambda: np.argmax(arr))
    bench("argmin", lambda: np.argmin(arr))

    print(f"\n--- Cumulative operations (N={N_MED}) ---")
    bench("cumsum", lambda: np.cumsum(arr))
    bench("cumprod (clipped)", lambda: np.cumprod(np.clip(arr, 0.1, 2.0)))

    print(f"\n--- Linear algebra extras ---")
    sq = np.random.randn(100, 100).astype(np.float32)
    v = np.random.randn(100).astype(np.float32)

    bench("outer(100)", lambda: np.outer(v, v))
    bench("diag(100)", lambda: np.diag(v))
    bench("trace(100x100)", lambda: np.trace(sq))

    print(f"\n--- Math functions (N={N_MED}) ---")
    bench("floor", lambda: np.floor(a))
    bench("ceil", lambda: np.ceil(a))
    bench("round", lambda: np.round(a))

    print(f"\n--- Statistics ---")
    bench("median", lambda: np.median(a))
    bench("percentile(50)", lambda: np.percentile(a, 50))
    bench("unique", lambda: np.unique(a))

    print(f"\n--- Search ---")
    sorted_arr = np.sort(a)
    bench("searchsorted", lambda: np.searchsorted(sorted_arr, 0.5))

    print(f"\n--- Random (N={N_MED}) ---")
    bench("rand", lambda: np.random.rand(N_MED).astype(np.float32))
    bench("randn", lambda: np.random.randn(N_MED).astype(np.float32))

    print(f"\n--- Linear algebra: inv/solve (N=100) ---")
    # Create invertible matrix
    inv_mat = np.random.randn(100, 100).astype(np.float32)
    inv_mat = inv_mat @ inv_mat.T + np.eye(100, dtype=np.float32)  # Make positive definite
    b_vec = np.random.randn(100).astype(np.float32)

    bench("inv(100x100)", lambda: np.linalg.inv(inv_mat))
    bench("solve(100x100)", lambda: np.linalg.solve(inv_mat, b_vec))

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
