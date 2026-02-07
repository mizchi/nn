#!/usr/bin/env python3
"""PyTorch benchmark for comparison with MoonBit autograd.

Same model architecture: 2-layer MLP with ReLU + cross-entropy loss.
Measures forward, backward, and SGD step separately.
"""

import time
import torch
import torch.nn as nn

def bench_mlp(batch, input_dim, hidden_dim, output_dim, warmup=10, iters=100):
    device = "cpu"
    # Use float32 to match MoonBit's Float (single precision)
    torch.set_default_dtype(torch.float32)

    # Create model
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Fixed input data (same across iterations, like MoonBit bench)
    x = torch.randn(batch, input_dim, device=device)
    labels = torch.randint(0, output_dim, (batch,), device=device)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    # Benchmark
    fwd_times = []
    bwd_times = []
    step_times = []
    total_times = []
    last_loss = 0.0

    for _ in range(iters):
        optimizer.zero_grad()

        t0 = time.perf_counter_ns()
        logits = model(x)
        loss = criterion(logits, labels)
        t1 = time.perf_counter_ns()

        loss.backward()
        t2 = time.perf_counter_ns()

        optimizer.step()
        t3 = time.perf_counter_ns()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
        step_times.append(t3 - t2)
        total_times.append(t3 - t0)
        last_loss = loss.item()

    def ns_to_ms(ns):
        return f"{ns / 1_000_000:.3f}"

    def median(lst):
        s = sorted(lst)
        return s[len(s) // 2]

    def avg(lst):
        return sum(lst) // len(lst)

    print(f"MLP [{batch}, {input_dim}] -> {hidden_dim} -> {output_dim}")
    print(f"  iters: {iters} (warmup: {warmup})")
    print(f"  loss:  {last_loss}")
    print(f"  forward:  avg={ns_to_ms(avg(fwd_times))}ms  p50={ns_to_ms(median(fwd_times))}ms")
    print(f"  backward: avg={ns_to_ms(avg(bwd_times))}ms  p50={ns_to_ms(median(bwd_times))}ms")
    print(f"  sgd_step: avg={ns_to_ms(avg(step_times))}ms  p50={ns_to_ms(median(step_times))}ms")
    print(f"  total:    avg={ns_to_ms(avg(total_times))}ms  p50={ns_to_ms(median(total_times))}ms")
    print()

if __name__ == "__main__":
    print(f"=== PyTorch Benchmark (CPU, float32) ===")
    print(f"PyTorch version: {torch.__version__}")
    print()

    bench_mlp(32, 784, 128, 10, warmup=10, iters=100)
    bench_mlp(64, 784, 256, 10, warmup=10, iters=100)
    bench_mlp(128, 784, 512, 10, warmup=10, iters=50)
    bench_mlp(256, 784, 1024, 10, warmup=10, iters=20)
