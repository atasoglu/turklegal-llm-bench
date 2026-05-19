"""Push the processed TurkLegalBench splits to the HuggingFace Hub."""

import json
import os
from pathlib import Path

from datasets import ClassLabel, DatasetDict, Features, Sequence, Value, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = "atasoglu/turklegal-llm-bench"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
SPLITS = ("train", "validation", "test")

FEATURES = Features(
    {
        "id": Value("string"),
        "question": Value("string"),
        "choices": {
            "text": Sequence(Value("string")),
            "label": Sequence(Value("string")),
        },
        "answerKey": Value("string"),
        "law_area": Value("string"),
        "law_area_raw": Value("string"),
        "mask_type": Value("string"),
        "difficulty": Value("string"),
        "source": Value("string"),
    }
)

DATASET_CARD = """\
---
language:
- tr
license: apache-2.0
task_categories:
- question-answering
task_ids:
- multiple-choice-qa
tags:
- turkish
- legal
- law
- arc
- benchmark
pretty_name: TurkLegalBench
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
  - split: test
    path: data/test-*.parquet
---

# TurkLegalBench

**TurkLegalBench** is a Turkish legal multiple-choice benchmark in ARC format,
derived from the [HukukBERT Cloze Benchmark](https://huggingface.co/datasets/turkhukuk/hukukbert-cloze-benchmark).
It covers 29 canonical legal domains (hukuk alanları) and is designed for
zero-shot / few-shot evaluation of large language models on Turkish statutory law.

## Dataset Summary

| Split      | Rows |
|------------|------|
| train      | 521  |
| validation | 112  |
| test       | 117  |
| **total**  | **750** |

## Format

Each row follows the [ARC](https://allenai.org/data/arc) multiple-choice format:

```json
{
  "id": "bert_cloze_0125",
  "question": "Vekalet sözleşmesinde vekil, işi görürken _____ borcu altındadır.",
  "choices": {
    "text": ["özen", "aldatma", "gabin", "illiyet"],
    "label": ["A", "B", "C", "D"]
  },
  "answerKey": "A",
  "law_area": "borclar_hukuku",
  "law_area_raw": "borclar_hukuku",
  "mask_type": "single_token",
  "difficulty": "easy",
  "source": "synthetic_benchmark"
}
```

The blank `_____` marks the position of the missing legal term. Models are
expected to identify the correct option from the `choices.text` list.

## Fields

| Field           | Type   | Description |
|-----------------|--------|-------------|
| `id`            | string | Unique row identifier from the source dataset |
| `question`      | string | Sentence with `_____` where the legal term is missing |
| `choices.text`  | list   | Four candidate answers |
| `choices.label` | list   | Corresponding labels A–D |
| `answerKey`     | string | Correct label (A / B / C / D) |
| `law_area`      | string | Canonical legal domain (29 categories) |
| `law_area_raw`  | string | Original `law_area` value before normalisation |
| `mask_type`     | string | Granularity of the blank (`single_token`, etc.) |
| `difficulty`    | string | Item difficulty label (`easy`, `medium`, `hard`) |
| `source`        | string | Provenance tag from the source dataset |

## Legal Domains

The 29 canonical `law_area` values:

`aile_hukuku`, `alternatif_uyusmazlik_cozumu`, `anayasa_hukuku`,
`avukatlik_hukuku`, `bankacilik_hukuku`, `bilisim_hukuku`, `borclar_hukuku`,
`cevre_hukuku`, `ceza_hukuku`, `ceza_muhakemesi_hukuku`, `esya_hukuku`,
`fikri_ve_sinai_haklar`, `icra_ve_iflas_hukuku`, `idare_hukuku`,
`idari_yargilama_hukuku`, `infaz_hukuku`, `is_hukuku`, `kiymetli_evrak_hukuku`,
`marka_hukuku`, `medeni_usul_hukuku`, `miras_hukuku`, `rekabet_hukuku`,
`sigorta_hukuku`, `ticaret_hukuku`, `tuketici_hukuku`,
`uluslararasi_kamu_hukuku`, `uluslararasi_ozel_hukuk`, `vergi_hukuku`,
`veri_koruma_hukuku`

## Source & License

Source dataset: [turkhukuk/hukukbert-cloze-benchmark](https://huggingface.co/datasets/turkhukuk/hukukbert-cloze-benchmark)  
Conversion & benchmark packaging: [atasoglu/turklegal-llm-bench](https://github.com/TurkHukuk/hukukbert)  
License: Apache 2.0
"""


def load_splits() -> DatasetDict:
    data_files = {
        split: str(DATA_DIR / f"turklegal_bench_{split}.jsonl") for split in SPLITS
    }
    return load_dataset("json", data_files=data_files, features=FEATURES)


def main() -> None:
    print("Loading splits …")
    ds = load_splits()
    print(ds)

    print(f"\nPushing to {HF_REPO} …")
    ds.push_to_hub(HF_REPO, token=HF_TOKEN, private=False)

    print("Uploading dataset card …")
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    print(f"\nDone — https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
