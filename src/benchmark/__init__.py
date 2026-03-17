"""Benchmark construction for financial statement verification."""

from benchmark.error_taxonomy import (
    ErrorCategory,
    ErrorSubtype,
    ErrorSeverity,
    DetectionDifficulty,
    ErrorType,
    ERROR_REGISTRY,
)
from benchmark.error_injection import (
    InjectionResult,
    inject_error,
    inject_multiple_errors,
)
from benchmark.dataset_builder import (
    BenchmarkInstance,
    DatasetStatistics,
    build_benchmark_dataset,
)

__all__ = [
    "ErrorCategory",
    "ErrorSubtype",
    "ErrorSeverity",
    "DetectionDifficulty",
    "ErrorType",
    "ERROR_REGISTRY",
    "InjectionResult",
    "inject_error",
    "inject_multiple_errors",
    "BenchmarkInstance",
    "DatasetStatistics",
    "build_benchmark_dataset",
]
