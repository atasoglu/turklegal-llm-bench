"""
Convert HukukBERT cloze dataset (MLM format) to ARC multiple-choice format.

Source schema (turkhukuk/hukukbert-cloze-benchmark):
  id       : str        — e.g. "bert_cloze_0001"
  sentence : str        — legal sentence with a single [MASK] placeholder
  options  : list[str]  — exactly 4 candidate answers
  gold     : str        — the correct answer string
  metadata : dict       — {law_area, mask_type, difficulty, source}

Target schema (ARC-style):
  id           : str
  question     : str        — sentence with [MASK] placeholder preserved
  choices      : dict       — {"text": [...], "label": ["A","B","C","D"]}
  answerKey    : str        — "A" | "B" | "C" | "D"
  law_area     : str        — canonical domain name (normalised)
  law_area_raw : str        — original value from metadata
  mask_type    : str        — "single_token" | "multi_token"
  difficulty   : str        — "easy" | "medium" | "hard"
  source       : str        — "synthetic_benchmark" | other
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import datasets
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LABELS = ["A", "B", "C", "D"]
HF_DATASET = "turkhukuk/hukukbert-cloze-benchmark"
OUT_DIR = Path(__file__).parent.parent / "data"
RANDOM_SEED = 42

# Re-partition the single upstream "test" split into train / validation / test.
# 70 % train is used for few-shot context by evaluation harnesses.
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}

# Collapses the 55 inconsistently-named raw law_area values into canonical
# snake_case domain names. Any raw value not listed here is kept as-is.
LAW_AREA_NORM: dict[str, str] = {
    "aile": "aile_hukuku",
    "anayasa": "anayasa_hukuku",
    "arabuluculuk": "alternatif_uyusmazlik_cozumu",
    "tahkim": "alternatif_uyusmazlik_cozumu",
    "avukatlik": "avukatlik_hukuku",
    "bilisim-suclari": "bilisim_hukuku",
    "borclar": "borclar_hukuku",
    "borclar-kira": "borclar_hukuku",
    "borclar-tazminat": "borclar_hukuku",
    "cevre": "cevre_hukuku",
    "ceza": "ceza_hukuku",
    "ceza-muhakemesi": "ceza_muhakemesi_hukuku",
    "esya": "esya_hukuku",
    "fikri-haklar": "fikri_ve_sinai_haklar",
    "icra-iflas": "icra_ve_iflas_hukuku",
    "idare": "idare_hukuku",
    "idare-kamulastirma": "idare_hukuku",
    "imar": "idare_hukuku",
    "is-hukuku": "is_hukuku",
    "kvkk": "veri_koruma_hukuku",
    "kıymetli_evrak_hukuku": "kiymetli_evrak_hukuku",  # unicode ı → i
    "ticari-senet": "kiymetli_evrak_hukuku",
    "medeni": "medeni_usul_hukuku",
    "usul": "medeni_usul_hukuku",
    "miras": "miras_hukuku",
    "sigorta": "sigorta_hukuku",
    "ticaret": "ticaret_hukuku",
    "ticaret-sirketler": "ticaret_hukuku",
    "tüketici_hukuku": "tuketici_hukuku",  # unicode ü → u
    "vergi": "vergi_hukuku",
}


def convert_row(row: dict) -> dict | None:
    """Convert one MLM-format row to ARC format.

    Returns None when the row fails validation; logs a warning in that case.
    """
    rid = row.get("id", "<unknown>")
    sentence: str = row.get("sentence", "")
    options: list[str] = row.get("options", [])
    gold: str = row.get("gold", "")
    meta: dict = row.get("metadata", {})

    if not sentence:
        log.warning("Row %s: empty sentence — skipping", rid)
        return None

    if "[MASK]" not in sentence:
        log.warning("Row %s: no [MASK] token in sentence — skipping", rid)
        return None

    if len(options) != 4:
        log.warning("Row %s: expected 4 options, got %d — skipping", rid, len(options))
        return None

    if gold not in options:
        log.warning("Row %s: gold %r not in options %s — skipping", rid, gold, options)
        return None

    answer_key = LABELS[options.index(gold)]
    raw_law_area = meta.get("law_area", "")

    return {
        "id": rid,
        "question": sentence.replace("[MASK]", "_____"),
        "choices": {
            "text": list(options),
            "label": LABELS[: len(options)],
        },
        "answerKey": answer_key,
        "law_area": LAW_AREA_NORM.get(raw_law_area, raw_law_area),
        "law_area_raw": raw_law_area,
        "mask_type": meta.get("mask_type", ""),
        "difficulty": meta.get("difficulty", ""),
        "source": meta.get("source", ""),
    }


def quality_report(rows: list[dict], tag: str = "") -> None:
    df = pd.DataFrame(rows)
    log.info("=== Quality report%s ===", f" [{tag}]" if tag else "")
    log.info("  Total rows : %d", len(df))
    log.info("  law_area   : %d unique — %s", df["law_area"].nunique(), sorted(df["law_area"].unique()))
    log.info("  mask_type  : %s", df["mask_type"].value_counts().to_dict())
    log.info("  difficulty : %s", df["difficulty"].value_counts().to_dict())
    log.info("  source     : %s", df["source"].value_counts().to_dict())
    log.info("  answerKey  : %s", df["answerKey"].value_counts().sort_index().to_dict())


def split_dataset(rows: list[dict]) -> dict[str, list[dict]]:
    """Deterministic stratified split by canonical law_area."""
    rng = np.random.default_rng(RANDOM_SEED)

    by_area: dict[str, list[dict]] = {}
    for row in rows:
        by_area.setdefault(row["law_area"], []).append(row)

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for area_rows in by_area.values():
        area_rows = list(area_rows)
        rng.shuffle(area_rows)
        n = len(area_rows)
        n_val = max(1, round(n * SPLIT_RATIOS["validation"]))
        n_test = max(1, round(n * SPLIT_RATIOS["test"]))
        n_train = n - n_val - n_test
        splits["train"].extend(area_rows[:n_train])
        splits["validation"].extend(area_rows[n_train: n_train + n_val])
        splits["test"].extend(area_rows[n_train + n_val:])

    for key in splits:
        rng.shuffle(splits[key])
        splits[key] = [dict(r) for r in splits[key]]

    return splits


def main() -> None:
    log.info("Loading dataset: %s", HF_DATASET)
    raw_ds = datasets.load_dataset(HF_DATASET, split="test")
    log.info("Loaded %d rows", len(raw_ds))

    raw_dir = OUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "hukukbert_cloze.jsonl"
    raw_ds.to_json(str(raw_path), force_ascii=False)
    log.info("Raw data saved → %s", raw_path)

    converted: list[dict] = []
    skipped = 0
    for row in raw_ds:
        result = convert_row(row)
        if result is None:
            skipped += 1
        else:
            converted.append(result)

    log.info("Converted %d rows (%d skipped)", len(converted), skipped)
    quality_report(converted, tag="full")

    splits = split_dataset(converted)
    for name, rows in splits.items():
        log.info("  %s: %d rows", name, len(rows))

    proc_dir = OUT_DIR / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in splits.items():
        out_path = proc_dir / f"turklegal_bench_{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("Saved %s → %s", split_name, out_path)

    combined_path = proc_dir / "turklegal_bench_all.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for split_name, rows in splits.items():
            for row in rows:
                f.write(json.dumps({**row, "split": split_name}, ensure_ascii=False) + "\n")
    log.info("Combined file saved → %s", combined_path)

    df_all = pd.DataFrame(converted)
    normalised_count = sum(1 for r in converted if r["law_area"] != r["law_area_raw"])
    summary = {
        "total_converted": len(converted),
        "total_skipped": skipped,
        "splits": {k: len(v) for k, v in splits.items()},
        "law_areas_canonical": sorted({r["law_area"] for r in converted}),
        "law_area_counts": df_all["law_area"].value_counts().sort_index().to_dict(),
        "law_area_normalised_rows": normalised_count,
    }
    summary_path = proc_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log.info("Summary saved → %s", summary_path)


if __name__ == "__main__":
    main()
