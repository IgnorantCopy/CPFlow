#!/usr/bin/env python3
"""
CPFlow Phase 5: Phylogeny & conservation analysis.

Requires: conda install -c bioconda muscle iqtree

Evaluates:
  - Multiple sequence alignment (MUSCLE v5)
  - Per-residue conservation score R_seq (Shannon entropy)
  - Phylogenetic tree (IQ-TREE with BLOSUM62 + ultrafast bootstrap)

API verified:
  MUSCLE v5:  muscle -align input.fa -output aligned.afa
  IQ-TREE:    iqtree -s aligned.fasta -m BLOSUM62 -B 1500 -T AUTO
              Output: .treefile (ML tree), .iqtree (report)

Usage:
  python protein_DIFF/eval/metrics_phylogeny.py \
      --csv result/predict/predict.csv \
      --wt_fasta dataset/Ago/wt_kmago.fasta \
      --output_dir result/phylogeny/
"""

import argparse, json, os, subprocess, sys, tempfile
import numpy as np
import pandas as pd

AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")


# ═══════════════════════════════════════════════════════════════
# MUSCLE MSA via subprocess
# ═══════════════════════════════════════════════════════════════

def run_muscle(input_fasta: str, output_fasta: str, super5: bool = False):
    """Run MUSCLE v5 multiple sequence alignment.

    API: muscle -align input.fa -output aligned.afa
    For large sets: muscle -super5 input.fa -output aligned.afa

    Args:
        input_fasta: path to input FASTA (all sequences + WT)
        output_fasta: path to output aligned FASTA
        super5: use Super5 algorithm for large datasets (>1000 seqs)
    """
    cmd = ["muscle", "-super5" if super5 else "-align",
           input_fasta, "-output", output_fasta]
    print(f"  [MUSCLE] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise ImportError(
            "MUSCLE v5 is required. Install:  conda install -c bioconda muscle"
        )
    print(f"  [MUSCLE] Alignment saved → {output_fasta}")


# ═══════════════════════════════════════════════════════════════
# Conservation score R_seq
# ═══════════════════════════════════════════════════════════════

def compute_conservation_scores(msa_path: str) -> dict:
    """Compute per-column conservation scores from a FASTA MSA.

    R_seq = log2(20) + Σ p_n * log2(p_n)
    where p_n is the observed frequency of amino acid n at that column.
    A gap penalty is applied: score *= (non-gap fraction).

    The paper (Sequence and structure analysis):
      "Residues in the Ago database scoring above 2.5 were selected."

    Returns dict:
      scores: (L,) array of conservation scores
      high_conserved_positions: indices where score > 2.5
      num_positions: total alignment length
    """
    # Parse MSA (simple FASTA parser)
    seqs = []
    current_seq = []
    with open(msa_path) as f:
        for line in f:
            if line.startswith(">"):
                if current_seq:
                    seqs.append("".join(current_seq))
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_seq:
            seqs.append("".join(current_seq))

    if not seqs:
        return {"scores": [], "error": "No sequences in MSA"}

    L = len(seqs[0])
    scores = []

    for col in range(L):
        col_chars = [s[col] for s in seqs]
        non_gap = [c for c in col_chars if c != "-"]
        n_non_gap = len(non_gap)

        if n_non_gap == 0:
            scores.append(0.0)
            continue

        # Frequency of each amino acid (only count valid AA codes)
        counts = {}
        for c in non_gap:
            if c in AMINO_ACIDS:
                counts[c] = counts.get(c, 0) + 1

        total = sum(counts.values())
        if total == 0:
            scores.append(0.0)
            continue

        # Shannon entropy: -Σ p * log2(p)
        entropy = -sum(
            (c / total) * np.log2(c / total)
            for c in counts.values()
        )
        R_seq = np.log2(20) - entropy

        # Gap penalty
        gap_fraction = n_non_gap / len(col_chars)
        scores.append(float(R_seq * gap_fraction))

    scores_arr = np.array(scores)
    high_pos = np.where(scores_arr > 2.5)[0].tolist()

    return {
        "scores": scores,
        "num_positions": L,
        "high_conserved_count": len(high_pos),
        "high_conserved_positions": high_pos,
        "mean_score": float(np.mean(scores_arr)),
        "std_score": float(np.std(scores_arr)),
        "max_score": float(np.max(scores_arr)),
    }


# ═══════════════════════════════════════════════════════════════
# IQ-TREE phylogeny via subprocess
# ═══════════════════════════════════════════════════════════════

def run_iqtree(msa_path: str, output_prefix: str, threads: str = "AUTO"):
    """Run IQ-TREE to build a maximum-likelihood phylogenetic tree.

    API: iqtree -s aligned.fasta -m BLOSUM62 -B 1500 -T AUTO
         -s   : input alignment (FASTA/PHYLIP)
         -m   : substitution model (BLOSUM62 for proteins)
         -B   : ultrafast bootstrap replicates
         -T   : number of threads (AUTO = auto-detect)
    Output: .treefile (ML tree in NEWICK format)
            .iqtree  (text report)
            .ufboot  (bootstrap trees)
    """
    cmd = [
        "iqtree",
        "-s", msa_path,
        "-m", "BLOSUM62",
        "-B", "1500",
        "-T", threads,
        "--prefix", output_prefix,
    ]
    print(f"  [IQ-TREE] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise ImportError(
            "IQ-TREE is required. Install:  conda install -c bioconda iqtree"
        )

    treefile = output_prefix + ".treefile"
    report_file = output_prefix + ".iqtree"

    # Read the tree (NEWICK format)
    tree_str = ""
    if os.path.exists(treefile):
        with open(treefile) as f:
            tree_str = f.read().strip()

    return {
        "treefile": treefile,
        "report": report_file,
        "tree_newick": tree_str,
    }


# ═══════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_phylogeny(csv_path: str, wt_fasta: str,
                       output_dir: str) -> dict:
    """Run full phylogeny & conservation pipeline.

    Steps:
      1. Combine WT + generated sequences into one FASTA
      2. Run MUSCLE alignment
      3. Compute conservation scores
      4. Run IQ-TREE phylogeny
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load sequences ──
    df = pd.read_csv(csv_path)
    seqs = {str(row["id"]): row["seq"] for _, row in df.iterrows()}

    # Load WT
    with open(wt_fasta) as f:
        wt_lines = f.readlines()
    wt_seq = "".join(l.strip() for l in wt_lines[1:])
    seqs["WT"] = wt_seq

    # ── Write combined FASTA ──
    combined_fa = os.path.join(output_dir, "combined.fasta")
    with open(combined_fa, "w") as f:
        f.write(f">WT\n{wt_seq}\n")
        for sid, s in seqs.items():
            if sid != "WT":
                f.write(f">{sid}\n{s}\n")
    print(f"  Combined FASTA: {len(seqs)} sequences → {combined_fa}")

    # ── MUSCLE ──
    aligned_fa = os.path.join(output_dir, "aligned.fasta")
    run_muscle(combined_fa, aligned_fa)

    # ── Conservation ──
    cons = compute_conservation_scores(aligned_fa)

    # ── IQ-TREE ──
    tree_prefix = os.path.join(output_dir, "phylo_tree")
    tree = run_iqtree(aligned_fa, tree_prefix)

    # ── Report ──
    report = {
        "num_sequences": len(seqs),
        "wt_length": len(wt_seq),
        "conservation": cons,
        "phylogeny": {
            "treefile": tree["treefile"],
            "report": tree["report"],
            "tree_newick_preview": tree["tree_newick"][:200] + "..."
            if len(tree["tree_newick"]) > 200 else tree["tree_newick"],
        },
    }

    report_path = os.path.join(output_dir, "metrics_phylogeny.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[metrics_phylogeny] Report saved → {report_path}")

    return report


def print_summary(report: dict):
    print("\n" + "=" * 60)
    print("  CPFlow — Phylogeny & Conservation Summary")
    print("=" * 60)

    c = report.get("conservation", {})
    print(f"\n  Conservation (R_seq = log2(20) + Σp·log2(p)):")
    print(f"    alignment length: {c.get('num_positions', '?')}")
    print(f"    high-conserved (R_seq > 2.5): {c.get('high_conserved_count', '?')} positions")
    print(f"    mean R_seq: {c.get('mean_score', 0):.3f}")
    print(f"    max  R_seq: {c.get('max_score', 0):.3f}")

    p = report.get("phylogeny", {})
    print(f"\n  Phylogenetic tree: {p.get('treefile', 'N/A')}")
    print(f"  Tree preview: {p.get('tree_newick_preview', 'N/A')}")

    print("\n" + "=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPFlow phylogeny & conservation")
    parser.add_argument("--csv", required=True,
                        help="Path to predict.csv")
    parser.add_argument("--wt_fasta", required=True,
                        help="Path to WT template FASTA")
    parser.add_argument("--output_dir", default="result/phylogeny",
                        help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: csv not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.wt_fasta):
        print(f"ERROR: wt_fasta not found: {args.wt_fasta}", file=sys.stderr)
        sys.exit(1)

    report = evaluate_phylogeny(args.csv, args.wt_fasta, args.output_dir)
    print_summary(report)
