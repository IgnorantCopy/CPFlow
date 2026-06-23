#!/usr/bin/env python3
"""
CPFlow Phase 1: Sequence-level evaluation metrics.
Zero external dependencies beyond Python stdlib + numpy.

Evaluates:
  - Catalytic motif preservation (DEDD/DEDH)
  - Sequence identity vs WT template
  - Pairwise sequence identity distribution
  - Sequence diversity statistics

Usage:
  python protein_DIFF/eval/metrics_sequence.py \
      --csv result/predict/predict.csv \
      --wt_fasta dataset/Ago/wt_kmago.fasta \
      --output result/predict/metrics_sequence.json
"""

import argparse, json, os, sys
import numpy as np
import pandas as pd

AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")

# ─── KmAgo / PfAgo catalytic motif positions (0-indexed) ───
CATALYTIC_MOTIFS = {
    "kmago": {
        "positions": [526, 561, 595, 712],  # D527, E562, D596, D713
        "expected": "DEDD",
        "name": "KmAgo DEDD",
    },
    "pfago": {
        # PfAgo DEDH tetrad — paper does not give residue numbers.
        # Set positions to None; if you have verified the PDB numbering,
        # fill in and E1's WT self-check will validate them at runtime.
        "positions": None,
        "expected": "DEDH",
        "name": "PfAgo DEDH",
    },
}


# ═══════════════════════════════════════════════════════════════
# Core metric functions
# ═══════════════════════════════════════════════════════════════

def seq_identity(s1: str, s2: str) -> float:
    """Fraction of identical amino acids at aligned positions."""
    length = min(len(s1), len(s2))
    if length == 0:
        return 0.0
    return sum(a == b for a, b in zip(s1[:length], s2[:length])) / length


def pairwise_identity_matrix(seqs: list) -> np.ndarray:
    """NxN matrix of pairwise sequence identities."""
    n = len(seqs)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ident = seq_identity(seqs[i], seqs[j])
            mat[i][j] = mat[j][i] = ident
    return mat


def check_catalytic_motif(seq: str, positions: list, expected: str) -> dict:
    """Check whether all catalytic residues are preserved."""
    failures = []
    for pos, aa in zip(positions, expected):
        if pos >= len(seq):
            failures.append({"pos": pos, "expected": aa, "found": "OUT_OF_RANGE"})
        elif seq[pos] != aa:
            failures.append({"pos": pos, "expected": aa, "found": seq[pos]})
    return {
        "intact": len(failures) == 0,
        "failures": failures,
        "expected": expected,
        "positions": positions,
    }


def seq_to_onehot(seq: str, max_len: int = None) -> np.ndarray:
    """Convert amino acid sequence to one-hot encoding."""
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    L = max_len if max_len else len(seq)
    feat = np.zeros((L, 20))
    for i, aa in enumerate(seq):
        if i >= L:
            break
        if aa in aa_to_idx:
            feat[i, aa_to_idx[aa]] = 1
    return feat.flatten()


def tsne_embedding(seqs: dict, wt_seq: str = None,
                   perplexity: float = 5.0, random_state: int = 42) -> dict:
    """Compute t-SNE embedding of generated sequences only.

    NOT a reproduction of paper Fig. 5e (which includes natural pAgo background).
    Use for internal visualization of generated sequence diversity.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    ids = sorted(seqs.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if len(ids) < 3:
        raise ValueError(f"Need at least 3 sequences for t-SNE, got {len(ids)}")
    seq_list = [seqs[i] for i in ids]
    max_len = max(len(s) for s in seq_list)

    features = np.array([seq_to_onehot(s, max_len) for s in seq_list])
    if float(np.var(features)) == 0.0:
        raise ValueError("All sequence features are identical; skipping t-SNE")

    effective_perplexity = min(perplexity, len(ids) - 1)
    if effective_perplexity <= 0:
        raise ValueError(f"Invalid t-SNE perplexity: {effective_perplexity}")

    tsne = TSNE(n_components=2, perplexity=effective_perplexity,
                random_state=random_state)
    embedding = tsne.fit_transform(features)

    return {
        "ids": ids,
        "embedding": embedding.tolist(),
        "perplexity": effective_perplexity,
    }


def check_all_catalytic(seqs: dict, motif_cfg: dict) -> dict:
    """Check catalytic motif for all sequences."""
    results = {}
    intact_count = 0
    for sid, seq in seqs.items():
        r = check_catalytic_motif(seq, motif_cfg["positions"], motif_cfg["expected"])
        results[sid] = r
        if r["intact"]:
            intact_count += 1
    return {
        "motif_name": motif_cfg["name"],
        "motif_expected": motif_cfg["expected"],
        "total_sequences": len(seqs),
        "intact_count": intact_count,
        "intact_ratio": intact_count / len(seqs) if seqs else 0,
        "per_sequence": results,
    }


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_sequences_from_csv(csv_path: str) -> dict:
    """Load {id: seq} from predict.csv."""
    df = pd.read_csv(csv_path)
    return {str(row["id"]): row["seq"] for _, row in df.iterrows()}


def load_sequences_from_fasta_dir(fasta_dir: str) -> dict:
    """Load {id: seq} from a directory of FASTA files."""
    seqs = {}
    for fname in sorted(os.listdir(fasta_dir)):
        if not fname.endswith(".fasta"):
            continue
        fpath = os.path.join(fasta_dir, fname)
        with open(fpath) as f:
            lines = f.readlines()
        header = lines[0].strip().lstrip(">")
        seq = "".join(l.strip() for l in lines[1:])
        sid = header.split("|")[0]  # "0|0.71|88935" → "0"
        seqs[sid] = seq
    return seqs


def load_wt_sequence(wt_path: str) -> str:
    """Load WT sequence from a FASTA file or a .pt graph file."""
    if wt_path.endswith(".pt"):
        import torch
        graph = torch.load(wt_path, weights_only=False)
        if hasattr(graph, "x") and graph.x.shape[1] >= 20:
            indices = graph.x[:, :20].argmax(dim=1).cpu().numpy()
            return "".join(AMINO_ACIDS[i] for i in indices)
        raise ValueError("Could not extract WT sequence from .pt graph")
    else:
        with open(wt_path) as f:
            lines = f.readlines()
        return "".join(l.strip() for l in lines[1:])


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def evaluate_sequences(
    seqs: dict,
    wt_seq: str,
    motif_cfg: dict,
    output_path: str = None,
    compute_tsne: bool = False,
) -> dict:
    """Run all sequence-level metrics and return a report dict."""

    # Sort numeric IDs numerically; fall back to lexical for non-numeric IDs
    # without raising (int() would crash on custom string IDs).
    all_ids = sorted(seqs.keys(),
                     key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
    bad_len = [i for i in all_ids if len(seqs[i]) != len(wt_seq)]
    if bad_len:
        raise ValueError(
            f"{len(bad_len)} sequences have length != WT ({len(wt_seq)} aa). "
            f"First offenders: {bad_len[:3]}. "
            f"Cannot compute per-position metrics. Check provenance of results."
        )
    ids = all_ids
    seq_list = [seqs[i] for i in ids]

    # ── 0. Sanity check: WT itself must pass the catalytic motif check ──
    wt_check = check_catalytic_motif(wt_seq, motif_cfg["positions"], motif_cfg["expected"])
    if not wt_check["intact"]:
        raise ValueError(
            f"WT sequence does NOT match expected catalytic motif "
            f"{motif_cfg['name']} at positions {motif_cfg['positions']}. "
            f"Failures: {wt_check['failures']}. "
            f"Check that --motif and --wt_fasta/--wt_graph are correct for this protein."
        )

    # ── 1. Catalytic motif ──
    motif_report = check_all_catalytic(seqs, motif_cfg)

    # ── 2. Identity vs WT ──
    wt_identities = {i: seq_identity(seqs[i], wt_seq) for i in ids}
    wt_values = list(wt_identities.values())

    # ── 3. Pairwise identity ──
    pw_mat = pairwise_identity_matrix(seq_list)
    pw_values = pw_mat[np.triu_indices(len(ids), k=1)]

    # ── 4. Diversity statistics ──
    recovery_from_csv = None  # may be filled if CSV has recovery column

    # ── 5. t-SNE embedding (generated sequences only, not paper Fig. 5e) ──
    tsne_data = None
    if compute_tsne:
        try:
            tsne_data = tsne_embedding(seqs, wt_seq)
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARN] t-SNE failed: {e}. Skipping.")
            tsne_data = None

    report = {
        "num_sequences": len(seqs),
        "wt_length": len(wt_seq),
        "seq_lengths": {i: len(seqs[i]) for i in ids},
        #
        "catalytic_motif": {
            "motif": motif_report["motif_name"],
            "expected": motif_report["motif_expected"],
            "intact_count": motif_report["intact_count"],
            "intact_ratio": motif_report["intact_ratio"],
            "intact": motif_report["intact_ratio"] == 1.0,
        },
        #
        "identity_vs_wt": {
            "mean": float(np.mean(wt_values)),
            "std": float(np.std(wt_values)),
            "min": float(np.min(wt_values)),
            "max": float(np.max(wt_values)),
            "median": float(np.median(wt_values)),
        },
        #
        "pairwise_identity": {
            "mean": float(np.mean(pw_values)) if len(pw_values) > 0 else 0,
            "std": float(np.std(pw_values)) if len(pw_values) > 0 else 0,
            "min": float(np.min(pw_values)) if len(pw_values) > 0 else 0,
            "max": float(np.max(pw_values)) if len(pw_values) > 0 else 0,
            "num_pairs": len(pw_values),
        },
        #
        "diversity_check": {
            "wt_identity_in_range_50_70": float(
                np.mean([1.0 if 0.50 <= v <= 0.70 else 0.0 for v in wt_values])
            ),
            "pairwise_identity_below_80": float(
                np.mean([1.0 if v < 0.80 else 0.0 for v in pw_values])
            ) if len(pw_values) > 0 else 0,
        },
    }

    if tsne_data is not None:
        report["tsne_generated_only"] = tsne_data
        report["tsne_note"] = (
            "t-SNE computed on generated sequences only. "
            "Paper Fig. 5e includes natural pAgo background — not reproduced here."
        )

    # ── Per-sequence detail (useful for debugging) ──
    report["per_sequence"] = {}
    for i in ids:
        report["per_sequence"][i] = {
            "identity_vs_wt": round(wt_identities[i], 4),
            "motif_intact": motif_report["per_sequence"][i]["intact"],
            "length": len(seqs[i]),
        }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[metrics_sequence] Report saved → {output_path}")

    return report


def print_summary(report: dict):
    """Pretty-print a summary of the evaluation report."""
    print("\n" + "=" * 60)
    print("  CPFlow — Sequence-level Evaluation Summary")
    print("=" * 60)

    m = report["catalytic_motif"]
    status = "✅ PASS" if m["intact"] else f"❌ FAIL ({m['intact_count']}/{report['num_sequences']})"
    print(f"\n  Catalytic motif ({m['motif']} {m['expected']}):  {status}")

    iv = report["identity_vs_wt"]
    print(f"\n  Identity vs WT:")
    print(f"    mean={iv['mean']:.3f}  std={iv['std']:.3f}  "
          f"range=[{iv['min']:.3f}, {iv['max']:.3f}]  median={iv['median']:.3f}")

    pw = report["pairwise_identity"]
    if pw["num_pairs"] > 0:
        print(f"\n  Pairwise identity ({pw['num_pairs']} pairs):")
        print(f"    mean={pw['mean']:.3f}  std={pw['std']:.3f}  "
              f"range=[{pw['min']:.3f}, {pw['max']:.3f}]")

    d = report["diversity_check"]
    print(f"\n  Diversity checks:")
    print(f"    WT identity in [50%,70%]:  {d['wt_identity_in_range_50_70']:.1%}")
    print(f"    Pairwise identity < 80%:  {d['pairwise_identity_below_80']:.1%}")

    # Flag warnings
    warnings = []
    if not m["intact"]:
        warnings.append(f"Catalytic motif broken in {report['num_sequences'] - m['intact_count']} sequences!")
    if iv["mean"] < 0.40:
        warnings.append(f"Mean WT identity {iv['mean']:.1%} < 40% — sequences may be too diverged")
    if iv["mean"] > 0.75:
        warnings.append(f"Mean WT identity {iv['mean']:.1%} > 75% — sequences may lack novelty")
    if pw["max"] > 0.85 and pw["num_pairs"] > 0:
        warnings.append(f"Max pairwise identity {pw['max']:.1%} > 85% — possible duplicate sequences")

    if warnings:
        print(f"\n  ⚠️  Warnings:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"\n  ✅ All checks passed.")

    print("\n" + "=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPFlow sequence-level evaluation")
    parser.add_argument("--csv", help="Path to predict.csv from inference output")
    parser.add_argument("--fasta_dir", help="Path to directory containing FASTA files")
    parser.add_argument("--wt_fasta", help="Path to WT template FASTA file")
    parser.add_argument("--wt_graph", help="Path to WT template .pt graph file (fallback)")
    parser.add_argument("--motif", default="kmago", choices=["kmago", "pfago"],
                        help="Which catalytic motif to check")
    parser.add_argument("--fix_pos_file", default=None,
                        help="File with conserved/catalytic positions (one per line: A123). "
                             "Overrides --motif positions.")
    parser.add_argument("--motif_positions", default=None,
                        help="Comma-separated 1-indexed positions (e.g. 558,596,628,745). "
                             "Overrides --motif and --fix_pos_file.")
    parser.add_argument("--output", default="result/predict/metrics_sequence.json",
                        help="Output JSON path")
    parser.add_argument("--compute_tsne", action="store_true",
                        help="Enable t-SNE embedding (off by default; can be slow/unstable)")
    args = parser.parse_args()

    # Load sequences
    if args.csv:
        seqs = load_sequences_from_csv(args.csv)
    elif args.fasta_dir:
        seqs = load_sequences_from_fasta_dir(args.fasta_dir)
    else:
        print("ERROR: need --csv or --fasta_dir", file=sys.stderr)
        sys.exit(1)

    if not seqs:
        print("ERROR: no sequences loaded", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(seqs)} sequences")

    # Load WT sequence
    wt_seq = None
    if args.wt_fasta and os.path.exists(args.wt_fasta):
        wt_seq = load_wt_sequence(args.wt_fasta)
    elif args.wt_graph and os.path.exists(args.wt_graph):
        wt_seq = load_wt_sequence(args.wt_graph)
    else:
        # Fallback: use the first sequence as WT (for quick testing only)
        print("WARNING: No WT sequence provided. Using first generated sequence as reference.")
        print("  Provide --wt_fasta or --wt_graph for accurate evaluation.")
        wt_seq = list(seqs.values())[0]

    print(f"WT sequence length: {len(wt_seq)}")

    # Get motif config
    motif_cfg = dict(CATALYTIC_MOTIFS[args.motif])  # copy
    if args.motif_positions:
        pos_list = [int(p.strip()) - 1 for p in args.motif_positions.split(",")]
        aa_list = [wt_seq[p] for p in pos_list if p < len(wt_seq)]
        motif_cfg = {
            "positions": pos_list,
            "expected": "".join(aa_list),
            "name": f"{args.motif} (CLI --motif_positions)",
        }
    elif args.fix_pos_file:
        pos_list = []
        aa_list = []
        with open(args.fix_pos_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    aa = line[0]
                    pos = int(line[1:])
                    aa_list.append(aa)
                    pos_list.append(pos - 1)
        motif_cfg = {
            "positions": pos_list,
            "expected": "".join(aa_list),
            "name": f"{args.motif} (from {args.fix_pos_file})",
        }
    if motif_cfg["positions"] is None:
        if args.motif == "pfago":
            print("ERROR: PfAgo DEDH catalytic tetrad residue numbers not hardcoded.\n"
                  "  Use --fix_pos_file dataset/Ago/pfago.piwi.fix.txt\n"
                  "  Or --motif_positions 558,596,628,745\n"
                  "  (D558, E596, D628, H745 — verify against your PfAgo PDB)",
                  file=sys.stderr)
        else:
            print(f"ERROR: motif positions not defined for {args.motif}",
                  file=sys.stderr)
        sys.exit(1)

    # Run evaluation
    report = evaluate_sequences(
        seqs, wt_seq, motif_cfg,
        output_path=args.output,
        compute_tsne=args.compute_tsne,
    )
    print_summary(report)
