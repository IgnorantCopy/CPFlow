#!/usr/bin/env python3
"""
Compare generated protein sequence sets from CPFlow/CPDiffusion outputs.

This is a lightweight first-pass evaluator: it only needs predict.csv files
with columns id, seq, and optionally recovery. It does not need WT sequences,
structure prediction, BLAST, or FoldSeek.

Example:
  python -m protein_DIFF.eval.compare_sequence_sets \
      --set origin=result/unpacked/origin/predict/predict.csv \
      --set flow_50=result/unpacked/flow_50/predict/predict.csv \
      --set flow_500=result/unpacked/flow_500/predict/predict.csv \
      --output result/unpacked/sequence_compare.json
"""

import argparse
import csv
import itertools
import json
import math
import os
from collections import Counter
from statistics import mean, median, pstdev


AMINO_ACIDS = set("ARNDCQEGHILKMFPSTWYV")


def parse_set_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--set must be NAME=CSV_PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise argparse.ArgumentTypeError("set name cannot be empty")
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"CSV not found: {path}")
    return name, path


def load_predict_csv(path):
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "seq"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            seq = (row.get("seq") or "").strip()
            rec_raw = row.get("recovery", "")
            recovery = None
            if rec_raw not in ("", None):
                try:
                    recovery = float(rec_raw)
                except ValueError:
                    recovery = None
            rows.append({
                "id": str(row.get("id", "")).strip(),
                "seq": seq,
                "recovery": recovery,
            })
    return rows


def stat(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
        "median": median(values),
    }


def seq_identity(seq_a, seq_b):
    length = min(len(seq_a), len(seq_b))
    if length == 0:
        return 0.0
    return sum(a == b for a, b in zip(seq_a[:length], seq_b[:length])) / length


def pairwise_identity_stats(seqs):
    values = [
        seq_identity(a, b)
        for a, b in itertools.combinations(seqs, 2)
    ]
    out = stat(values)
    out["num_pairs"] = len(values)
    if values:
        out["diversity_mean"] = 1.0 - out["mean"]
    return out


def aa_composition(seqs):
    counts = Counter()
    total = 0
    invalid = Counter()
    for seq in seqs:
        for aa in seq:
            if aa in AMINO_ACIDS:
                counts[aa] += 1
                total += 1
            else:
                invalid[aa] += 1
    comp = {aa: (counts[aa] / total if total else 0.0) for aa in sorted(AMINO_ACIDS)}
    grouped = {
        "hydrophobic": sum(counts[a] for a in "AVLIPFWM") / total if total else 0.0,
        "polar_uncharged": sum(counts[a] for a in "GSTCYNQ") / total if total else 0.0,
        "positive": sum(counts[a] for a in "RHK") / total if total else 0.0,
        "negative": sum(counts[a] for a in "DE") / total if total else 0.0,
        "charged": sum(counts[a] for a in "RHKDE") / total if total else 0.0,
    }
    return {
        "per_aa": comp,
        "groups": grouped,
        "invalid_residues": dict(invalid),
    }


def summarize_set(rows):
    seqs = [r["seq"] for r in rows]
    recoveries = [r["recovery"] for r in rows]
    lengths = [len(s) for s in seqs]
    unique_seqs = set(seqs)
    id_counts = Counter(r["id"] for r in rows)
    duplicate_ids = {sid: count for sid, count in id_counts.items() if count > 1}

    return {
        "num_sequences": len(rows),
        "num_unique_sequences": len(unique_seqs),
        "duplicate_sequence_count": len(rows) - len(unique_seqs),
        "duplicate_sequence_rate": (len(rows) - len(unique_seqs)) / len(rows) if rows else 0.0,
        "duplicate_ids": duplicate_ids,
        "length": stat(lengths),
        "unique_lengths": sorted(set(lengths)),
        "recovery": stat(recoveries),
        "pairwise_identity": pairwise_identity_stats(seqs),
        "aa_composition": aa_composition(seqs),
    }


def cross_set_stats(rows_a, rows_b):
    by_id_a = {r["id"]: r["seq"] for r in rows_a}
    by_id_b = {r["id"]: r["seq"] for r in rows_b}
    common_ids = sorted(set(by_id_a) & set(by_id_b), key=lambda x: int(x) if x.isdigit() else x)
    same_id_values = [seq_identity(by_id_a[sid], by_id_b[sid]) for sid in common_ids]

    all_values = [
        seq_identity(a["seq"], b["seq"])
        for a in rows_a
        for b in rows_b
    ]

    return {
        "common_id_count": len(common_ids),
        "same_id_identity": stat(same_id_values),
        "all_pairs_identity": {
            **stat(all_values),
            "num_pairs": len(all_values),
        },
    }


def print_report(report):
    print("\nSequence Set Comparison")
    print("=" * 80)
    print(f"{'set':<12} {'n':>5} {'len':>12} {'rec_mean':>10} {'rec_std':>9} "
          f"{'pw_ident':>10} {'diversity':>10} {'dups':>6}")
    for name, summary in report["sets"].items():
        length = summary["length"]
        recovery = summary["recovery"]
        pairwise = summary["pairwise_identity"]
        length_text = "-"
        if length.get("count"):
            if len(summary["unique_lengths"]) == 1:
                length_text = str(summary["unique_lengths"][0])
            else:
                length_text = f"{length['min']}-{length['max']}"
        print(
            f"{name:<12} {summary['num_sequences']:>5} {length_text:>12} "
            f"{recovery.get('mean', 0.0):>10.4f} {recovery.get('std', 0.0):>9.4f} "
            f"{pairwise.get('mean', 0.0):>10.4f} {pairwise.get('diversity_mean', 0.0):>10.4f} "
            f"{summary['duplicate_sequence_count']:>6}"
        )

    print("\nCross-set identity")
    print("-" * 80)
    print(f"{'pair':<25} {'same_id_n':>9} {'same_id_mean':>14} {'all_pair_mean':>14}")
    for pair_name, stats in report["cross_sets"].items():
        print(
            f"{pair_name:<25} {stats['common_id_count']:>9} "
            f"{stats['same_id_identity'].get('mean', 0.0):>14.4f} "
            f"{stats['all_pairs_identity'].get('mean', 0.0):>14.4f}"
        )
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare generated protein sequence sets")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        type=parse_set_arg,
        required=True,
        help="Named CSV input, formatted as NAME=PATH. Repeat for multiple sets.",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    loaded = {name: load_predict_csv(path) for name, path in args.sets}

    report = {
        "inputs": {name: path for name, path in args.sets},
        "sets": {name: summarize_set(rows) for name, rows in loaded.items()},
        "cross_sets": {},
    }

    for name_a, name_b in itertools.combinations(loaded.keys(), 2):
        report["cross_sets"][f"{name_a}__vs__{name_b}"] = cross_set_stats(
            loaded[name_a], loaded[name_b]
        )

    print_report(report)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"Saved JSON report to {args.output}")


if __name__ == "__main__":
    main()
