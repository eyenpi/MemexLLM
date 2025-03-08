import time
from typing import Any, Callable, Dict

import pytest


@pytest.fixture()  # type: ignore
def performance_config() -> Dict[str, Any]:
    """Fixture to provide configuration for performance tests."""
    return {
        "iterations": 100,
        "warmup_iterations": 10,
        "timeout": 60,
    }


@pytest.fixture()  # type: ignore
def timer() -> Callable[[], float]:
    """Fixture to provide a timer function for performance tests."""
    start_time = time.time()
    return lambda: time.time() - start_time


@pytest.fixture()  # type: ignore
def benchmark() -> Callable[[Callable[[], Any]], Dict[str, float]]:
    """Fixture to provide a benchmark function for performance tests."""

    def _benchmark(func: Callable[[], Any], iterations: int = 100) -> Dict[str, float]:
        times = []
        # Warmup
        for _ in range(10):
            func()

        # Actual benchmark
        for _ in range(iterations):
            start = time.time()
            func()
            end = time.time()
            times.append(end - start)

        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "total": sum(times),
        }

    return _benchmark
