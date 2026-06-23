#!/usr/bin/env python3
"""
CPFlow Phase 2: Training & inference efficiency metrics.
Zero external dependencies beyond Python stdlib + torch.

Evaluates:
  - Parameter count
  - GPU memory usage
  - Inference time per sequence
  - Diffusion steps vs recovery quality

Usage:
  # After training, parse the run_pt output CSV
  python protein_DIFF/eval/metrics_efficiency.py \
      --training_csv result/Ago/Jun_5_ago_dataset=CATH_result_lr=0.0005_...csv \
      --output result/Ago/metrics_efficiency.json

  # Compare two runs
  python protein_DIFF/eval/metrics_efficiency.py \
      --training_csv result/Ago_original/metrics.csv \
      --compare_csv result/Ago_improved/metrics.csv \
      --output result/comparison.json
"""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# Training log parsing
# ═══════════════════════════════════════════════════════════════

def parse_training_log(csv_path: str) -> dict:
    """Extract key metrics from run_pt training output CSV."""
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)

    metrics = {}
    for col in ["train_loss", "val_loss", "recovery", "perplexity"]:
        if col in df.columns:
            values = df[col].dropna().values
            if len(values) > 0:
                metrics[col] = {
                    "final": float(values[-1]),
                    "best": float(np.min(values)) if "loss" in col or col == "perplexity" else float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "num_epochs": len(values),
                    "first": float(values[0]),
                }

    # Convergence: epoch at which recovery reaches 90% of final value
    if "recovery" in df.columns:
        vals = df["recovery"].dropna().values
        if len(vals) > 1 and vals[-1] > 0:
            target = vals[-1] * 0.9
            converged_at = None
            for i, v in enumerate(vals):
                if v >= target:
                    converged_at = int(i)
                    break
            metrics["convergence"] = {
                "final_recovery": float(vals[-1]),
                "epoch_to_90pct_final": converged_at,
                "recovery_per_epoch": float((vals[-1] - vals[0]) / len(vals)) if len(vals) > 1 else 0,
            }

    return metrics


# ═══════════════════════════════════════════════════════════════
# Model / hardware metrics
# ═══════════════════════════════════════════════════════════════

def measure_model_params(model) -> dict:
    """Measure parameter count (callable with a PyTorch model)."""
    import torch
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_M": round(total / 1e6, 2),
        "trainable_params_M": round(trainable / 1e6, 2),
    }


def measure_gpu_memory() -> dict:
    """Record peak GPU memory since last reset."""
    import torch
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    return {
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "max_allocated_GB": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "max_reserved_GB": round(torch.cuda.max_memory_reserved() / (1024**3), 3),
    }


# ═══════════════════════════════════════════════════════════════
# Inference timing
# ═══════════════════════════════════════════════════════════════

class InferTimer:
    """Context manager / decorator for timing inference steps."""

    def __init__(self):
        self.records = []

    def record(self, label: str, elapsed: float, extras: dict = None):
        self.records.append({"label": label, "elapsed_sec": round(elapsed, 4),
                             **(extras or {})})

    def summary(self) -> dict:
        if not self.records:
            return {}
        labels = set(r["label"] for r in self.records)
        summary = {}
        for lbl in labels:
            times = [r["elapsed_sec"] for r in self.records if r["label"] == lbl]
            summary[lbl] = {
                "count": len(times),
                "total_sec": round(sum(times), 3),
                "mean_sec": round(np.mean(times), 4),
                "std_sec": round(np.std(times), 4),
                "min_sec": round(np.min(times), 4),
                "max_sec": round(np.max(times), 4),
            }
        return summary


def time_sample_call(diffusion, data, num_runs=10, warmup=2) -> dict:
    """Measure inference time for diffusion.sample().

    Usage (inside inference script):
        timer = InferTimer()
        for i in range(num_runs):
            t0 = time.perf_counter()
            zt, sample = diffusion.sample(data, temperature=1.0, stop=0, step=5)
            timer.record("sample", time.perf_counter() - t0)
    """
    return {}  # Placeholder — caller instruments this themselves


# ═══════════════════════════════════════════════════════════════
# Command-line interface
# ═══════════════════════════════════════════════════════════════

def build_report(training_csv: str, compare_csv: str = None) -> dict:
    report = {}

    report["training"] = parse_training_log(training_csv)

    if compare_csv:
        report["training_compare"] = parse_training_log(compare_csv)

        # Delta comparison
        delta = {}
        for key in report["training"]:
            if key in report.get("training_compare", {}):
                t = report["training"][key]
                c = report["training_compare"][key]
                if isinstance(t, dict) and "final" in t and "final" in c:
                    delta[key] = {
                        "final_delta": round(c["final"] - t["final"], 6),
                        "final_delta_pct": round((c["final"] - t["final"]) / (abs(t["final"]) + 1e-10) * 100, 2),
                    }
        report["comparison_delta"] = delta

    return report


def print_report(report: dict):
    print("\n" + "=" * 60)
    print("  CPFlow — Efficiency Metrics Summary")
    print("=" * 60)

    train = report.get("training", {})
    if train:
        print("\n  Training log:")
        for metric, vals in train.items():
            if metric == "convergence":
                c = vals
                print(f"    recovery: final={c['final_recovery']:.4f}, "
                      f"90%-final @ epoch {c['epoch_to_90pct_final']}")
            elif isinstance(vals, dict) and "final" in vals:
                direction = "↓" if "loss" in metric or "perplexity" in metric else "↑"
                print(f"    {metric}: final={vals['final']:.4f} {direction}  "
                      f"(best={vals.get('best',0):.4f},  range=[{vals.get('first',0):.4f}, {vals.get('final',0):.4f}])")

    delta = report.get("comparison_delta", {})
    if delta:
        print("\n  Comparison delta (improved - original):")
        for metric, d in delta.items():
            sign = "+" if d["final_delta"] > 0 else ""
            print(f"    {metric}: {sign}{d['final_delta']:.4f} ({d['final_delta_pct']:+.1f}%)")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPFlow efficiency metrics")
    parser.add_argument("--training_csv", help="Path to run_pt output CSV")
    parser.add_argument("--compare_csv", help="Path to comparison run CSV")
    parser.add_argument("--output", default="result/metrics_efficiency.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if not args.training_csv:
        print("Usage: --training_csv <path> [--compare_csv <path>]", file=sys.stderr)
        sys.exit(1)

    report = build_report(args.training_csv, args.compare_csv)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[metrics_efficiency] Report saved → {args.output}")
    print_report(report)
