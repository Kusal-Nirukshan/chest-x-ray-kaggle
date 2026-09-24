from __future__ import annotations

from scripts.kaggle_trustworthiness_pair import _flatten_efficiency


def test_flatten_efficiency_uses_real_benchmark_keys():
    row = {
        "params_total": 123,
        "gflops": 1.5,
        "cpu_latency_ms": 10.0,
        "gpu_latency_ms": 2.0,
    }
    assert _flatten_efficiency(row) == {
        "params": 123,
        "gflops": 1.5,
        "cpu_latency_ms": 10.0,
        "gpu_latency_ms": 2.0,
    }
