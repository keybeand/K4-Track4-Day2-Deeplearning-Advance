"""benchmark.py - Đo độ trễ p50/p95/p99 đúng chuẩn GPU với CUDA synchronize và Warmup.
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn


def measure_latency(model: nn.Module, img_size: int = 224, batch_size: int = 1,
                    device: str = "cuda" if torch.cuda.is_available() else "cpu",
                    num_warmup: int = 10, num_runs: int = 50) -> dict[str, float]:
    """Đo độ trễ đúng chuẩn (slide trang 73, 76):
    1. Warmup 10 lần đầu bỏ đi.
    2. Gọi torch.cuda.synchronize() trước và sau mỗi lần đo.
    3. Đo 50 lần, lấy p50, p95, p99.
    """
    model.to(device)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # 1. Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if "cuda" in device:
                torch.cuda.synchronize()

    # 2. Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if "cuda" in device:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_input)

            if "cuda" in device:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)  # ms

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    throughput = (batch_size * 1000.0) / p50 if p50 > 0 else 0.0

    print(f"Latency ({device}, batch={batch_size}) -> p50: {p50:.2f} ms | p95: {p95:.2f} ms | p99: {p99:.2f} ms | Throughput: {throughput:.1f} img/s")

    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "throughput_img_s": throughput,
    }
