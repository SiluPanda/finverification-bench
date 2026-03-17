#!/bin/bash
# FinVerBench — Full Pipeline Runner
# Usage: ./run_pipeline.sh [step]
# Steps: fetch, parse, inject, evaluate, analyze, all

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$BASE_DIR/src"
DATA_DIR="$BASE_DIR/data"
RESULTS_DIR="$BASE_DIR/results"

step="${1:-all}"

fetch_data() {
    echo "=== Step 1: Fetching SEC EDGAR data ==="
    python3 "$SRC_DIR/data/fetch_filings.py" --output-dir "$DATA_DIR/raw"
}

parse_data() {
    echo "=== Step 2: Parsing financial statements ==="
    python3 "$SRC_DIR/data/parse_financials.py" \
        --input-dir "$DATA_DIR/raw" \
        --output-dir "$DATA_DIR/processed"
}

inject_errors() {
    echo "=== Step 3: Building benchmark dataset ==="
    python3 "$SRC_DIR/benchmark/dataset_builder.py" \
        --input-dir "$DATA_DIR/processed" \
        --output-dir "$DATA_DIR/benchmark"
}

evaluate() {
    echo "=== Step 4: Evaluating LLMs ==="
    local models=("claude-sonnet-4-20250514" "gpt-4o")
    local strategies=("zero_shot" "few_shot" "chain_of_thought")

    for model in "${models[@]}"; do
        for strategy in "${strategies[@]}"; do
            echo "  Evaluating $model with $strategy..."
            python3 "$SRC_DIR/evaluation/evaluate_llm.py" \
                --model "$model" \
                --prompt-strategy "$strategy" \
                --input "$DATA_DIR/benchmark/benchmark.json" \
                --output "$RESULTS_DIR/${model}_${strategy}.json" \
                --max-instances 0
        done
    done
}

analyze() {
    echo "=== Step 5: Analyzing results ==="
    python3 "$SRC_DIR/analysis/analyze_results.py" --results-dir "$RESULTS_DIR"
    python3 "$SRC_DIR/analysis/plot_results.py" \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$BASE_DIR/paper/figures"
}

case "$step" in
    fetch)    fetch_data ;;
    parse)    parse_data ;;
    inject)   inject_errors ;;
    evaluate) evaluate ;;
    analyze)  analyze ;;
    all)
        fetch_data
        parse_data
        inject_errors
        evaluate
        analyze
        echo "=== Pipeline complete ==="
        ;;
    *)
        echo "Usage: $0 {fetch|parse|inject|evaluate|analyze|all}"
        exit 1
        ;;
esac
