#!/usr/bin/env python3
"""
CPFlow: Amino acid property preservation analysis.
Corresponds to paper Supplementary Fig. S4.

Evaluates whether generated sequences preserve the physicochemical
properties of amino acids at each position, compared to WT.

Property groups (standard classification):
  hydrophobic:     A, V, L, I, P, F, W, M
  polar_uncharged: G, S, T, C, Y, N, Q
  positive:        R, H, K
  negative:        D, E
  special:         C (disulfide), G (flexibility), P (kink)

The paper found: CPDiffusion correctly preserves polar AA charges while
ProteinMPNN mis-generates many polar AAs with opposite charges.

Usage:
  python protein_DIFF/eval/metrics_aa_properties.py \
      --csv result/predict/predict.csv \
      --wt_fasta dataset/Ago/wt_kmago.fasta \
      --output result/metrics_aa_properties.json
"""

import argparse, json, os, sys
import numpy as np
import pandas as pd

AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")

# ─── AA property groups ───
AA_GROUPS = {
    "hydrophobic":     set("AVLIPFWM"),
    "polar_uncharged": set("GSTCYNQ"),
    "positive":        set("RHK"),
    "negative":        set("DE"),
    "aromatic":        set("FWY"),
    "small":           set("GAS"),
    "tiny":            set("GAS"),  # subset of small
    "aliphatic":       set("AVLI"),
    "charged":         set("RHKDE"),
    "polar":           set("GSTCYNQRHKDE"),  # all except hydrophobic
}

# Charge polarity groups (used in paper Supp Fig. S4)
POLAR_POSITIVE = set("RHK")
POLAR_NEGATIVE = set("DE")
POLAR_UNCHARGED = set("GSTCYNQ")
HYDROPHOBIC = set("AVLIPFWM")


def load_sequences(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    return {str(row["id"]): row["seq"] for _, row in df.iterrows()}


def load_wt_sequence(wt_path: str) -> str:
    with open(wt_path) as f:
        lines = f.readlines()
    return "".join(l.strip() for l in lines[1:])


def classify_aa(aa: str) -> str:
    """Classify a single amino acid into its property group."""
    if aa in HYDROPHOBIC:
        return "hydrophobic"
    elif aa in POLAR_POSITIVE:
        return "polar_positive"
    elif aa in POLAR_NEGATIVE:
        return "polar_negative"
    elif aa in POLAR_UNCHARGED:
        return "polar_uncharged"
    return "other"


def compare_aa_properties(seqs: dict, wt_seq: str) -> dict:
    """Compare AA property distributions between generated and WT.

    For each position, checks whether the generated AA belongs to
    the same property group as the WT residue at that position.
    """
    wt_len = len(wt_seq)
    wt_groups = [classify_aa(aa) for aa in wt_seq]

    # Per-sequence property preservation
    per_seq = {}
    all_preserved_counts = {g: [] for g in
                            ["hydrophobic", "polar_positive", "polar_negative", "polar_uncharged"]}

    for sid, seq in seqs.items():
        length = min(len(seq), wt_len)
        total = 0
        correct = 0
        group_correct = {g: 0 for g in all_preserved_counts}
        group_total = {g: 0 for g in all_preserved_counts}

        for i in range(length):
            wt_group = wt_groups[i]
            ap_group = classify_aa(seq[i])
            total += 1
            if wt_group in group_total:
                group_total[wt_group] += 1
            if ap_group == wt_group:
                correct += 1
                if wt_group in group_correct:
                    group_correct[wt_group] += 1

        overall = correct / total if total > 0 else 0
        per_seq[sid] = {
            "overall_preservation": round(overall, 4),
            "per_group": {}
        }
        for g in all_preserved_counts:
            ratio = group_correct[g] / group_total[g] if group_total[g] > 0 else 0
            per_seq[sid]["per_group"][g] = round(ratio, 4)
            all_preserved_counts[g].append(ratio)

    # Charge-flip analysis (paper Fig. S4 focus)
    charge_flips = {
        "positive_to_negative": 0,
        "negative_to_positive": 0,
        "charge_flip_total": 0,
        "polar_uncharged_to_charged": 0,
        "charged_to_polar_uncharged": 0,
    }
    charge_positions = {"positive": [], "negative": []}
    for i, wt_aa in enumerate(wt_seq):
        if wt_aa in POLAR_POSITIVE:
            charge_positions["positive"].append(i)
        elif wt_aa in POLAR_NEGATIVE:
            charge_positions["negative"].append(i)

    for sid, seq in seqs.items():
        for i in charge_positions["positive"]:
            if i < len(seq) and seq[i] in POLAR_NEGATIVE:
                charge_flips["positive_to_negative"] += 1
                charge_flips["charge_flip_total"] += 1
        for i in charge_positions["negative"]:
            if i < len(seq) and seq[i] in POLAR_POSITIVE:
                charge_flips["negative_to_positive"] += 1
                charge_flips["charge_flip_total"] += 1

    # Charge flips normalized by (num_sequences * num_charged_positions)
    n_seqs = len(seqs)
    n_pos = len(charge_positions["positive"])
    n_neg = len(charge_positions["negative"])
    charge_flips["positive_to_negative_rate"] = round(
        charge_flips["positive_to_negative"] / (n_seqs * n_pos), 6) if n_pos * n_seqs > 0 else 0
    charge_flips["negative_to_positive_rate"] = round(
        charge_flips["negative_to_positive"] / (n_seqs * n_neg), 6) if n_neg * n_seqs > 0 else 0
    charge_flips["num_positive_positions_wt"] = n_pos
    charge_flips["num_negative_positions_wt"] = n_neg

    # Summary
    summary = {}
    for g, vals in all_preserved_counts.items():
        if vals:
            summary[g] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

    overall_vals = [v["overall_preservation"] for v in per_seq.values()]
    summary["overall"] = {
        "mean": round(float(np.mean(overall_vals)), 4),
        "std": round(float(np.std(overall_vals)), 4),
        "note": "Fraction of positions where generated AA shares WT property group",
    }

    return {
        "num_sequences": len(seqs),
        "wt_length": wt_len,
        "wt_composition": {
            g: round(sum(1 for aa in wt_seq if classify_aa(aa) == g) / wt_len, 4)
            for g in ["hydrophobic", "polar_positive", "polar_negative", "polar_uncharged"]
        },
        "group_preservation": summary,
        "charge_flips": charge_flips,
        "per_sequence": per_seq,
    }


def print_summary(report: dict):
    print("\n" + "=" * 60)
    print("  CPFlow — Amino Acid Property Preservation")
    print("=" * 60)

    wt = report.get("wt_composition", {})
    print(f"\n  WT composition:")
    for g, frac in wt.items():
        print(f"    {g}: {frac:.1%}")

    gp = report.get("group_preservation", {})
    print(f"\n  Group preservation (per-sequence mean ± std):")
    for g in ["hydrophobic", "polar_uncharged", "polar_positive", "polar_negative"]:
        s = gp.get(g, {})
        print(f"    {g}: {s.get('mean', 0):.3f} ± {s.get('std', 0):.3f}  "
              f"range=[{s.get('min', 0):.3f}, {s.get('max', 0):.3f}]")

    o = gp.get("overall", {})
    print(f"\n    OVERALL: {o.get('mean', 0):.3f} ± {o.get('std', 0):.3f}")
    print(f"    ({o.get('note', '')})")

    cf = report.get("charge_flips", {})
    print(f"\n  Charge flips (paper Supp Fig. S4 focus):")
    print(f"    positive→negative: {cf.get('positive_to_negative', 0)} "
          f"({cf.get('positive_to_negative_rate', 0):.4%})")
    print(f"    negative→positive: {cf.get('negative_to_positive', 0)} "
          f"({cf.get('negative_to_positive_rate', 0):.4%})")
    print(f"    total flips: {cf.get('charge_flip_total', 0)}")

    if cf.get("positive_to_negative_rate", 0) > 0.05:
        print(f"    ⚠️  High charge flip rate — check polar AA preservation!")
    else:
        print(f"    ✅ Charge preservation looks good.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AA property preservation")
    parser.add_argument("--csv", required=True, help="Path to predict.csv")
    parser.add_argument("--wt_fasta", required=True, help="WT template FASTA")
    parser.add_argument("--output", default="result/metrics_aa_properties.json")
    args = parser.parse_args()

    seqs = load_sequences(args.csv)
    wt_seq = load_wt_sequence(args.wt_fasta)
    print(f"Loaded {len(seqs)} sequences, WT length: {len(wt_seq)}")

    report = compare_aa_properties(seqs, wt_seq)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[metrics_aa_properties] Report saved → {args.output}")

    print_summary(report)
