"""Evaluate LLMs on the FinVerBench financial statement verification benchmark.

Loads the benchmark dataset from ``data/benchmark/benchmark.json``, submits
each instance to an LLM backend using one of three prompting strategies
(zero-shot, few-shot, chain-of-thought), parses the response, and writes
structured results to ``results/``.

Supported backends:
  - Anthropic (Claude) via the ``anthropic`` SDK
  - OpenAI (GPT-4) via the ``openai`` SDK
  - Local models via OpenAI-compatible API (e.g. vLLM, llama.cpp server)

API keys are read from environment variables: ANTHROPIC_API_KEY, OPENAI_API_KEY.

Usage:
    python -m src.evaluation.evaluate_llm \\
        --model claude-sonnet-4-20250514 \\
        --prompt-strategy cot \\
        --max-instances 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation.prompts import build_prompt, STRATEGY_BUILDERS

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "benchmark.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_RUNS = 3  # each instance is evaluated this many times for stability
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class InstanceResult:
    """Result for a single benchmark instance across multiple runs."""

    instance_id: str
    model: str
    prompt_strategy: str
    ground_truth: Dict[str, Any]
    runs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def majority_prediction(self) -> Dict[str, Any]:
        """Return the majority-vote prediction across runs.

        For ``has_error``, take the majority.  For ``error_location`` and
        ``explanation``, take the value from the first run that agrees with
        the majority decision.
        """
        if not self.runs:
            return {"has_error": False, "error_location": None, "explanation": ""}
        votes = [bool(r.get("has_error")) for r in self.runs]
        majority = sum(votes) > len(votes) / 2
        # Pick the first run whose decision matches the majority.
        for r in self.runs:
            if bool(r.get("has_error")) == majority:
                return {
                    "has_error": majority,
                    "error_location": r.get("error_location"),
                    "explanation": r.get("explanation", ""),
                }
        return {"has_error": majority, "error_location": None, "explanation": ""}

    @property
    def agreement_rate(self) -> float:
        """Fraction of runs that agree with the majority decision."""
        if not self.runs:
            return 0.0
        votes = [bool(r.get("has_error")) for r in self.runs]
        majority = sum(votes) > len(votes) / 2
        return sum(1 for v in votes if v == majority) / len(votes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "ground_truth": self.ground_truth,
            "runs": self.runs,
            "majority_prediction": self.majority_prediction,
            "agreement_rate": self.agreement_rate,
        }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_json_from_response(text: str) -> Dict[str, Any]:
    """Parse the model response to extract the structured JSON answer.

    Handles responses where the JSON is embedded in markdown code fences
    or surrounded by explanation text.
    """
    # Try to find a JSON block in code fences.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a bare JSON object.
    brace_match = re.search(r"\{[^{}]*\"has_error\"[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try the most permissive: find any { ... } block.
    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if "has_error" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Fallback: heuristic extraction from unstructured text.
    has_error = False
    text_lower = text.lower()
    if any(kw in text_lower for kw in [
        "error found", "inconsistency", "discrepancy", "does not match",
        "do not match", "incorrect", "mismatch", "\"has_error\": true",
        "has_error: true",
    ]):
        has_error = True

    return {
        "has_error": has_error,
        "error_location": None,
        "explanation": text[:500],
        "_parse_failed": True,
    }


def parse_model_response(raw_response: str) -> Dict[str, Any]:
    """Parse a raw model response into a standardised prediction dict.

    Returns a dict with at least ``has_error``, ``error_location``, and
    ``explanation`` keys.
    """
    parsed = _extract_json_from_response(raw_response)
    # Normalise boolean.
    has_error = parsed.get("has_error")
    if isinstance(has_error, str):
        has_error = has_error.lower() in ("true", "yes", "1")
    parsed["has_error"] = bool(has_error)
    # Ensure required keys exist.
    parsed.setdefault("error_location", None)
    parsed.setdefault("explanation", "")
    return parsed


# ---------------------------------------------------------------------------
# LLM backend abstraction
# ---------------------------------------------------------------------------

def _call_anthropic(
    prompt: str,
    model: str,
    temperature: float = 0.0,
) -> str:
    """Call the Anthropic Messages API and return the text response."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for Anthropic models. "
            "Install it with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
) -> str:
    """Call the OpenAI Chat Completions API and return the text response.

    Also supports OpenAI-compatible local servers via *base_url*.
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for OpenAI models. "
            "Install it with: pip install openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and base_url is None:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    kwargs: Dict[str, Any] = {}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    else:
        kwargs["api_key"] = "not-needed"

    client = openai.OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _detect_backend(model: str) -> str:
    """Infer the backend from the model name."""
    model_lower = model.lower()
    if any(tag in model_lower for tag in ["claude", "anthropic"]):
        return "anthropic"
    if any(tag in model_lower for tag in ["gpt", "o1", "o3", "openai"]):
        return "openai"
    # Default to OpenAI-compatible API for local models.
    return "openai_local"


def call_llm(
    prompt: str,
    model: str,
    backend: Optional[str] = None,
    temperature: float = 0.0,
    local_base_url: str = "http://localhost:8000/v1",
) -> str:
    """Call an LLM and return the raw text response.

    Parameters
    ----------
    prompt:
        The complete prompt string.
    model:
        Model identifier (e.g. ``"claude-sonnet-4-20250514"``, ``"gpt-4"``).
    backend:
        Force a backend: ``"anthropic"``, ``"openai"``, or ``"openai_local"``.
        If ``None``, the backend is inferred from the model name.
    temperature:
        Sampling temperature (0 for deterministic).
    local_base_url:
        Base URL for local OpenAI-compatible servers.
    """
    if backend is None:
        backend = _detect_backend(model)

    if backend == "anthropic":
        return _call_anthropic(prompt, model, temperature=temperature)
    elif backend == "openai":
        return _call_openai(prompt, model, temperature=temperature)
    elif backend == "openai_local":
        return _call_openai(
            prompt, model, temperature=temperature, base_url=local_base_url,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def call_llm_with_retries(
    prompt: str,
    model: str,
    backend: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = MAX_RETRIES,
    local_base_url: str = "http://localhost:8000/v1",
) -> str:
    """Call an LLM with exponential-backoff retries for rate limiting.

    Retries on rate-limit errors and transient server errors.  All other
    exceptions are raised immediately.
    """
    backoff = INITIAL_BACKOFF_SECONDS
    last_exception: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return call_llm(
                prompt, model, backend=backend,
                temperature=temperature, local_base_url=local_base_url,
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            is_retryable = any(kw in exc_str for kw in [
                "rate_limit", "rate limit", "429", "overloaded",
                "503", "server error", "timeout", "connection",
            ])
            if not is_retryable:
                raise
            last_exception = exc
            logger.warning(
                "Retryable error on attempt %d/%d: %s. "
                "Backing off %.1fs ...",
                attempt, max_retries, exc, backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    raise RuntimeError(
        f"Failed after {max_retries} retries. Last error: {last_exception}"
    )


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

def load_benchmark(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the benchmark dataset from JSON."""
    path = path or BENCHMARK_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Benchmark file must contain a JSON array.")
    logger.info("Loaded %d benchmark instances from %s", len(data), path)
    return data


def evaluate_instance(
    instance: Dict[str, Any],
    model: str,
    prompt_strategy: str,
    backend: Optional[str] = None,
    num_runs: int = NUM_RUNS,
    local_base_url: str = "http://localhost:8000/v1",
) -> InstanceResult:
    """Evaluate a single benchmark instance, running it ``num_runs`` times.

    Parameters
    ----------
    instance:
        A single benchmark instance dict (from benchmark.json).
    model:
        Model identifier.
    prompt_strategy:
        One of ``"zero_shot"``, ``"few_shot"``, ``"cot"``.
    backend:
        LLM backend override.
    num_runs:
        Number of repeated evaluations for stability assessment.
    local_base_url:
        Base URL for local model servers.

    Returns
    -------
    InstanceResult
        Contains all run outputs and ground truth.
    """
    instance_id = instance["instance_id"]
    formatted_stmts = instance["formatted_statements"]
    ground_truth = instance["ground_truth"]

    prompt = build_prompt(prompt_strategy, formatted_stmts)

    result = InstanceResult(
        instance_id=instance_id,
        model=model,
        prompt_strategy=prompt_strategy,
        ground_truth=ground_truth,
    )

    for run_idx in range(num_runs):
        logger.info(
            "  Instance %s | run %d/%d", instance_id, run_idx + 1, num_runs,
        )
        try:
            raw_response = call_llm_with_retries(
                prompt, model, backend=backend,
                temperature=0.0, local_base_url=local_base_url,
            )
            parsed = parse_model_response(raw_response)
            parsed["raw_response"] = raw_response
            parsed["run_index"] = run_idx
            parsed["error"] = None
        except Exception as exc:
            logger.error(
                "  Instance %s | run %d FAILED: %s",
                instance_id, run_idx + 1, exc,
            )
            parsed = {
                "has_error": False,
                "error_location": None,
                "explanation": "",
                "raw_response": "",
                "run_index": run_idx,
                "error": str(exc),
            }
        result.runs.append(parsed)

    return result


def run_evaluation(
    model: str,
    prompt_strategy: str,
    benchmark_path: Optional[Path] = None,
    max_instances: Optional[int] = None,
    backend: Optional[str] = None,
    num_runs: int = NUM_RUNS,
    local_base_url: str = "http://localhost:8000/v1",
) -> List[InstanceResult]:
    """Run the full evaluation pipeline.

    Parameters
    ----------
    model:
        Model identifier.
    prompt_strategy:
        One of ``"zero_shot"``, ``"few_shot"``, ``"cot"``.
    benchmark_path:
        Path to benchmark.json (default: data/benchmark/benchmark.json).
    max_instances:
        Limit evaluation to the first N instances (for testing).
    backend:
        LLM backend override.
    num_runs:
        Number of repeated runs per instance.
    local_base_url:
        Base URL for local model servers.

    Returns
    -------
    list[InstanceResult]
        One result per benchmark instance.
    """
    instances = load_benchmark(benchmark_path)
    if max_instances is not None:
        instances = instances[:max_instances]

    logger.info(
        "Evaluating %d instances with model=%s, strategy=%s, runs=%d",
        len(instances), model, prompt_strategy, num_runs,
    )

    results: List[InstanceResult] = []
    for idx, instance in enumerate(instances):
        logger.info(
            "Instance %d/%d: %s",
            idx + 1, len(instances), instance.get("instance_id", "?"),
        )
        result = evaluate_instance(
            instance, model, prompt_strategy,
            backend=backend, num_runs=num_runs,
            local_base_url=local_base_url,
        )
        results.append(result)

    return results


def save_results(
    results: List[InstanceResult],
    model: str,
    prompt_strategy: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save evaluation results to a JSON file in the results directory.

    File is named ``<model>__<strategy>__<timestamp>.json``.

    Returns the path to the written file.
    """
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model.replace("/", "_").replace(":", "_")
    filename = f"{safe_model}__{prompt_strategy}__{timestamp}.json"
    filepath = output_dir / filename

    payload = {
        "metadata": {
            "model": model,
            "prompt_strategy": prompt_strategy,
            "num_instances": len(results),
            "num_runs_per_instance": NUM_RUNS,
            "timestamp": timestamp,
        },
        "results": [r.to_dict() for r in results],
    }

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an LLM on the FinVerBench financial statement "
            "verification benchmark."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model identifier, e.g. 'claude-sonnet-4-20250514', 'gpt-4', "
            "or a local model name."
        ),
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        choices=list(STRATEGY_BUILDERS.keys()),
        default="cot",
        help="Prompting strategy (default: cot).",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Evaluate at most N instances (for testing).",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=None,
        help=f"Path to benchmark.json (default: {BENCHMARK_PATH}).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["anthropic", "openai", "openai_local"],
        default=None,
        help="Force LLM backend (default: auto-detect from model name).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=NUM_RUNS,
        help=f"Number of runs per instance for stability (default: {NUM_RUNS}).",
    )
    parser.add_argument(
        "--local-base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for local OpenAI-compatible servers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory for results (default: {RESULTS_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    results = run_evaluation(
        model=args.model,
        prompt_strategy=args.prompt_strategy,
        benchmark_path=args.benchmark_path,
        max_instances=args.max_instances,
        backend=args.backend,
        num_runs=args.num_runs,
        local_base_url=args.local_base_url,
    )

    filepath = save_results(
        results,
        model=args.model,
        prompt_strategy=args.prompt_strategy,
        output_dir=args.output_dir,
    )
    logger.info(
        "Evaluation complete. %d instances evaluated. Results: %s",
        len(results), filepath,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
