#!/usr/bin/env bash
# Run TurkLegalBench evaluation via OpenRouter.
#
# Usage:
#   bash scripts/run_eval.sh [model] [limit]
#
# Model names follow OpenRouter's provider/name convention:
#   openai/gpt-4.1-mini          (default)
#   openai/gpt-4.1
#   anthropic/claude-sonnet-4
#   anthropic/claude-opus-4
#   google/gemini-2.5-pro
#   meta-llama/llama-4-maverick
#
# Full model list: https://openrouter.ai/models
#
# Requires OPENROUTER_API_KEY in .env

set -euo pipefail

MODEL="${1:-openai/gpt-4.1-mini}"
LIMIT="${2:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

source "$ROOT/.env" 2>/dev/null || true

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Error: OPENROUTER_API_KEY is not set in .env" >&2
  exit 1
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

SAFE_MODEL="${MODEL//\//_}"
TIMESTAMP=$(date +%Y%m%dT%H%M%S)
OUT_DIR="$ROOT/results/$SAFE_MODEL/$TIMESTAMP"

echo "Model   : $MODEL"
echo "Limit   : ${LIMIT:-all (test split)}"
echo "Output  : $OUT_DIR"
echo ""

uv run lm-eval \
  --model openai-chat-completions \
  --model_args "model=$MODEL,base_url=https://openrouter.ai/api/v1,api_key=$OPENROUTER_API_KEY" \
  --tasks turklegal_bench \
  --include_path "$ROOT/tasks" \
  --apply_chat_template \
  --num_fewshot 0 \
  "${LIMIT_ARGS[@]}" \
  --log_samples \
  --output_path "$OUT_DIR" \
  --seed 42

SAMPLES=$(find "$OUT_DIR" -name "samples_*.jsonl" | head -1)
if [[ -n "$SAMPLES" ]]; then
  echo ""
  echo "Domain Breakdown:"
  uv run python "$ROOT/scripts/domain_breakdown.py" "$SAMPLES" \
    --out "$OUT_DIR/domain_breakdown.json"
fi
