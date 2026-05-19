# TurkLegalBench

A multiple-choice benchmark for evaluating large language models on Turkish statutory law. Derived from the [HukukBERT Cloze Benchmark](https://huggingface.co/datasets/turkhukuk/hukukbert-cloze-benchmark), converted to ARC format for zero-shot / few-shot evaluation.

**Dataset:** [atasoglu/turklegal-llm-bench](https://huggingface.co/datasets/atasoglu/turklegal-llm-bench)

## Dataset

750 multiple-choice questions across 29 legal domains (hukuk alanları).

| Split      | Rows |
|------------|-----:|
| train      |  521 |
| validation |  112 |
| test       |  117 |

Each question has a blank (`_____`) and four candidate answers (A–D):

```json
{
  "question": "Vekalet sözleşmesinde vekil, işi görürken _____ borcu altındadır.",
  "choices": { "text": ["özen", "aldatma", "gabin", "illiyet"], "label": ["A","B","C","D"] },
  "answerKey": "A",
  "law_area": "borclar_hukuku"
}
```

## Setup

```bash
# requires Python 3.12+
uv sync
cp .env.example .env   # add your OPENROUTER_API_KEY
```

## Running an Evaluation

Evaluations run via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) through [OpenRouter](https://openrouter.ai), which provides access to all major models with a single API key.

```bash
# smoke test (10 samples)
bash scripts/run_eval.sh openai/gpt-4.1-mini 10

# full test split
bash scripts/run_eval.sh openai/gpt-4.1
bash scripts/run_eval.sh anthropic/claude-sonnet-4
bash scripts/run_eval.sh google/gemini-2.5-pro
```

Results and per-domain breakdown are saved to `results/<model>/<timestamp>/`.

## Repository Layout

```
tasks/
  turklegal_bench.yaml      # lm-eval task definition
scripts/
  convert_to_arc.py         # HukukBERT → ARC format conversion
  push_to_hf.py             # upload dataset to HuggingFace Hub
  run_eval.sh               # evaluation runner (OpenRouter)
  domain_breakdown.py       # per-domain accuracy from lm-eval samples
data/
  raw/                      # original HukukBERT JSONL
  processed/                # train / validation / test splits
results/                    # evaluation outputs (git-ignored)
```

## Source & License

Original research: [TurkHukuk — HukukBERT](https://www.turkhukuk.ai/blog/hukuk-yapay-zeka-modeli)  
Source dataset: [turkhukuk/hukukbert-cloze-benchmark](https://huggingface.co/datasets/turkhukuk/hukukbert-cloze-benchmark)  
GitHub: [TurkHukuk/hukukbert](https://github.com/TurkHukuk/hukukbert)  
License: Apache 2.0
