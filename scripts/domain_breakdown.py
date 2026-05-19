"""
Compute per-domain accuracy from an lm-eval --log_samples output file.

Usage:
    uv run python scripts/domain_breakdown.py <samples.jsonl> [--out results.json]

The samples file is the JSONL produced by lm-eval when run with --log_samples.
Each line must contain at least: doc.law_area, exact_match (or acc).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def compute_breakdown(samples: list[dict]) -> dict:
    metric_key = "exact_match" if "exact_match" in samples[0] else "acc"

    domain_correct: dict[str, int] = defaultdict(int)
    domain_total: dict[str, int] = defaultdict(int)

    for s in samples:
        law_area = s["doc"].get("law_area", "unknown")
        score = s.get(metric_key, 0)
        domain_correct[law_area] += int(score == 1.0)
        domain_total[law_area] += 1

    breakdown = {}
    for domain in sorted(domain_total):
        total = domain_total[domain]
        correct = domain_correct[domain]
        breakdown[domain] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }

    overall_total = sum(domain_total.values())
    overall_correct = sum(domain_correct.values())
    breakdown["_overall"] = {
        "correct": overall_correct,
        "total": overall_total,
        "accuracy": round(overall_correct / overall_total, 4) if overall_total else 0.0,
    }

    return breakdown


def print_table(breakdown: dict) -> None:
    header = f"{'Domain':<40} {'Correct':>7} {'Total':>7} {'Accuracy':>9}"
    print(header)
    print("-" * len(header))
    for domain, stats in breakdown.items():
        if domain == "_overall":
            continue
        print(
            f"{domain:<40} {stats['correct']:>7} {stats['total']:>7} {stats['accuracy']:>9.1%}"
        )
    print("-" * len(header))
    o = breakdown["_overall"]
    print(f"{'OVERALL':<40} {o['correct']:>7} {o['total']:>7} {o['accuracy']:>9.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="lm-eval samples JSONL file")
    parser.add_argument("--out", type=Path, default=None, help="Save breakdown to JSON")
    args = parser.parse_args()

    if not args.samples.exists():
        print(f"Error: {args.samples} not found", file=sys.stderr)
        sys.exit(1)

    samples = load_samples(args.samples)
    breakdown = compute_breakdown(samples)

    print_table(breakdown)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(breakdown, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
