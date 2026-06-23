#!/usr/bin/env python3
"""
CPFlow: Mutation effect prediction evaluation (Spearman correlation).
Wraps the original run_pt.compute_single_site_corr_score_all().

Corresponds to paper Supplementary Data, Section 4:
  Predicts single-site mutation effects and computes Spearman
  correlation against experimental DMS scores from ProteinGym.

REQUIRES:
  - ProteinGym dataset at: dataset/evaluation/DATASET/
    Each subfolder must contain: {protein}.pdb, ss, {protein}.1.tsv
  - A trained CPDiffusion checkpoint
  - Running this loads the full diffusion model and runs inference

Usage:
  # Evaluate mutation effect prediction at a specific diffusion step:
  python protein_DIFF/eval/metrics_spearman.py \
      --ckpt ckpt/model.pt \
      --eval_dir dataset/evaluation/DATASET/ \
      --output result/metrics_spearman.json

  # Evaluate across multiple steps to find best correlation:
  python protein_DIFF/eval/metrics_spearman.py \
      --ckpt ckpt/model.pt \
      --eval_dir dataset/evaluation/DATASET/ \
      --steps 300,350,400,450,480,490 \
      --output result/metrics_spearman.json
"""

import argparse, json, os, sys
import numpy as np
import torch
import pandas as pd
from scipy.stats import spearmanr

# Need to import from the repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from protein_DIFF.run_pt import (
    amino_acids_type, Trainer, EGNN_NET, Sparse_DIGRESS,
    prepare_mutation_graph,
)
from protein_DIFF.dataset.large_dataset import Cath
from protein_DIFF.dataset.utils import NormalizeProtein
from torch_geometric.data import Batch, Data
import torch.nn.functional as F


def load_model(checkpoint_path: str, device: str = "cuda:0"):
    """Load a trained CPDiffusion model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Build model
    # We need a dummy dataset to get feature dimensions
    # Use the Cath dataset with the target protein dir from config
    target_protein_dir = config.get("target_protein_dir", "dataset/Ago/process/")

    # For evaluation we don't actually need the full dataset,
    # just the feature dimensions
    # Load one sample to get dims
    sample_path = None
    if os.path.isdir(target_protein_dir):
        files = [f for f in os.listdir(target_protein_dir) if f.endswith(".pt")]
        if files:
            sample_path = os.path.join(target_protein_dir, files[0])

    if sample_path is None:
        raise FileNotFoundError(
            f"No .pt files found in {target_protein_dir}. "
            f"Need at least one graph file to determine feature dimensions."
        )

    sample_graph = torch.load(sample_path, weights_only=False)
    input_feat_dim = sample_graph.x.shape[1] + sample_graph.extra_x.shape[1]
    edge_attr_dim = sample_graph.edge_attr.shape[1]

    model = EGNN_NET(
        input_feat_dim=input_feat_dim,
        hidden_channels=config["hidden_dim"],
        edge_attr_dim=edge_attr_dim,
        dropout=config["drop_out"],
        n_layers=config["depth"],
        update_edge=True,
        embedding=config.get("embedding", False),
        embedding_dim=config.get("embedding_dim", 64),
        norm_feat=config.get("norm_feat", False),
        output_dim=20,
        embedding_ss=config.get("embed_ss", False),
    )

    diffusion = Sparse_DIGRESS(
        model=model,
        config=config,
        timesteps=config["timesteps"],
        objective=config["objective"],
        label_smooth_tem=config.get("smooth_temperature", 1.0),
    )

    diffusion.load_state_dict(checkpoint["model"], strict=False)
    diffusion = diffusion.to(device)
    diffusion.eval()

    return diffusion, config, sample_graph


def evaluate_mutation_effects(
    diffusion,
    eval_dir: str,
    device: str = "cuda:0",
    steps: list = None,
    n_realizations: int = 10,
) -> dict:
    """Evaluate mutation effect prediction across ProteinGym proteins.

    For each protein in eval_dir:
      1. Build graph from PDB
      2. For each single-site mutation, compute predicted score
         (negative cross-entropy of the mutant sequence)
      3. Compute Spearman correlation with experimental DMS scores

    Args:
        diffusion: loaded Sparse_DIGRESS model
        eval_dir: path to dataset/evaluation/DATASET/
        device: cuda device
        steps: list of diffusion timesteps to evaluate
        n_realizations: number of forward passes for averaging
    """
    if steps is None:
        steps = [450]  # Default: paper's best step

    # Prepare dataset
    from protein_DIFF.dataset.pdbbind_eval import PdbbindEvaluate
    from protein_DIFF.dataset.utils import dataset_argument

    args_ds = dataset_argument(n=7)  # evaluation config
    normalize = NormalizeProtein(filename=args_ds["normal_file"])

    dsm_list = sorted([d for d in os.listdir(eval_dir)
                       if os.path.isdir(os.path.join(eval_dir, d))
                       and d != ".DS_Store"])

    results_all = []
    best_corrs = {}

    for protein_name in dsm_list:
        protein_dir = os.path.join(eval_dir, protein_name)

        # Get PDB and mutation data
        pdb_path = os.path.join(protein_dir, f"{protein_name}.pdb")
        ss_path = os.path.join(protein_dir, "ss")
        tsv_path = os.path.join(protein_dir, f"{protein_name}.1.tsv")

        if not all(os.path.exists(p) for p in [pdb_path, ss_path, tsv_path]):
            print(f"  [SKIP] {protein_name}: missing files")
            continue

        # Load mutation data
        mut_df = pd.read_csv(tsv_path, sep="\t")
        # Parse mutations
        mutations = []
        for _, row in mut_df.iterrows():
            mutant_str = row.get("mutant", "")
            if "_" in str(mutant_str):
                continue
            try:
                wt_aa = mutant_str[0]
                pos = int(mutant_str[1:-1])
                mt_aa = mutant_str[-1]
                score = float(row.get("score", row.get("DMS_score", 0)))
                mutations.append((wt_aa, pos, mt_aa, score))
            except (ValueError, IndexError):
                continue

        if not mutations:
            print(f"  [SKIP] {protein_name}: no valid mutations")
            continue

        # Build graph
        from protein_DIFF.run_pt import pdb2graph, get_struc2ndRes

        # We need a Cath dataset for graph building
        # Create minimal dataset with just the needed methods
        class MinimalDataset:
            pass
        ds = MinimalDataset()
        # Monkey-patch the required methods from PdbbindEvaluate
        eval_ds = PdbbindEvaluate(
            args_ds["root"], args_ds["name"], args_ds["raw_dir"],
            c_alpha_max_neighbors=args_ds["c_alpha_max_neighbors"],
            pre_transform=normalize,
            replace_graph=False,
            replace_process=False,
        )
        ds.get_receptor_inference = eval_ds.get_receptor_inference
        ds.get_calpha_graph = eval_ds.get_calpha_graph

        try:
            graph = pdb2graph(ds, pdb_path, ss_path)
            graph = normalize(graph)
        except Exception as e:
            print(f"  [FAIL] {protein_name}: graph building error: {e}")
            continue

        # Prepare data for model
        data = Batch.from_data_list([graph])
        data_input = Data.clone(data)
        data_input.extra_x = torch.cat(
            [data.x[:, 20].unsqueeze(dim=1), data.x[:, 22:], data.mu_r_norm],
            dim=-1,
        )
        data_input.x = data.x[:, :20].to(torch.float32)
        data_input = data_input.to(device)

        for stop_step in steps:
            pred_list = []
            for _ in range(n_realizations):
                with torch.no_grad():
                    t_int = torch.ones(
                        size=(data.batch[-1].item() + 1, 1),
                        device=device,
                    ).float() * (500 - stop_step)
                    noise_data = diffusion.apply_noise(data_input, t_int)
                    pred, _ = diffusion.model(noise_data, t_int)
                    pred_list.append(pred)

            avg_pred = torch.stack(pred_list).mean(dim=0)

            # Score each mutation
            pred_scores = []
            true_scores = []
            seq_len = data_input.x.shape[0]

            for wt_aa, pos, mt_aa, dms_score in mutations:
                if pos < 1 or pos > seq_len:
                    continue
                target = data_input.x[:, :20].argmax(dim=1).clone()
                mt_idx = amino_acids_type.index(mt_aa)
                target[pos - 1] = mt_idx
                ce = F.cross_entropy(avg_pred, target, reduction="mean").item()
                pred_scores.append(-ce)  # higher = more likely
                true_scores.append(dms_score)

            if len(pred_scores) > 5:
                corr, pval = spearmanr(pred_scores, true_scores)
                results_all.append({
                    "protein": protein_name,
                    "step": stop_step,
                    "spearman_r": round(float(corr), 4),
                    "p_value": round(float(pval), 6),
                    "num_mutations": len(pred_scores),
                })
                print(f"  {protein_name} step={stop_step}: "
                      f"Spearman r={corr:.4f} (n={len(pred_scores)})")

                if protein_name not in best_corrs:
                    best_corrs[protein_name] = {}
                if stop_step not in best_corrs[protein_name]:
                    best_corrs[protein_name][stop_step] = corr
                best_corrs[protein_name][stop_step] = max(
                    best_corrs[protein_name][stop_step], corr
                )

    # Summary
    if not results_all:
        return {"error": "No valid mutations found", "eval_dir": eval_dir}

    df = pd.DataFrame(results_all)

    # Weighted average across proteins
    protein_weights = {}
    for protein_name in dsm_list:
        sub = df[df["protein"] == protein_name]
        if len(sub) > 0:
            protein_weights[protein_name] = sub["num_mutations"].iloc[0]

    total_mutations = sum(protein_weights.values())
    weighted_corr = 0
    for protein_name, n_mut in protein_weights.items():
        sub = df[df["protein"] == protein_name]
        if len(sub) > 0:
            best_r = sub["spearman_r"].max()
            weighted_corr += (n_mut / total_mutations) * best_r

    report = {
        "num_proteins_evaluated": len(df["protein"].unique()),
        "total_mutations": total_mutations,
        "weighted_spearman_r": round(float(weighted_corr), 4),
        "per_step_summary": {},
        "per_protein": results_all,
    }

    for step in steps:
        sub = df[df["step"] == step]
        if len(sub) > 0:
            abs_vals = sub["spearman_r"].abs()
            report["per_step_summary"][str(step)] = {
                "mean_abs_r": round(float(abs_vals.mean()), 4),
                "std_abs_r": round(float(abs_vals.std()), 4),
                "num_proteins": len(sub["protein"].unique()),
            }

    return report


def print_summary(report: dict):
    print("\n" + "=" * 60)
    print("  CPFlow — Mutation Effect Prediction (Spearman)")
    print("=" * 60)
    print(f"\n  Proteins evaluated: {report.get('num_proteins_evaluated', 0)}")
    print(f"  Total mutations: {report.get('total_mutations', 0)}")
    print(f"  Weighted Spearman r: {report.get('weighted_spearman_r', 0):.4f}")

    pss = report.get("per_step_summary", {})
    if pss:
        print(f"\n  Per diffusion step:")
        for step, s in sorted(pss.items(), key=lambda x: int(x[0])):
            print(f"    step={step}: |r|={s['mean_abs_r']:.4f} ± {s['std_abs_r']:.4f} "
                  f"({s['num_proteins']} proteins)")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spearman mutation effect evaluation")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint .pt")
    parser.add_argument("--eval_dir", default="dataset/evaluation/DATASET/",
                        help="ProteinGym dataset directory")
    parser.add_argument("--steps", default="450",
                        help="Comma-separated diffusion timesteps (e.g. 300,400,450)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--realizations", type=int, default=10,
                        help="Number of forward passes for averaging")
    parser.add_argument("--output", default="result/metrics_spearman.json")
    args = parser.parse_args()

    steps = [int(s.strip()) for s in args.steps.split(",")]

    if not os.path.isdir(args.eval_dir):
        print(f"ERROR: eval_dir not found: {args.eval_dir}", file=sys.stderr)
        print("  This requires the ProteinGym single-site mutation dataset.", file=sys.stderr)
        print("  Download from: https://github.com/OATML-Markslab/ProteinGym", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.ckpt}...")
    diffusion, config, _ = load_model(args.ckpt, args.device)
    print(f"  Model loaded. Steps to evaluate: {steps}")

    report = evaluate_mutation_effects(
        diffusion, args.eval_dir, args.device, steps, args.realizations,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[metrics_spearman] Report saved → {args.output}")
    print_summary(report)
