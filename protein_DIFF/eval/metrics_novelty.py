#!/usr/bin/env python3
"""
CPFlow Phase 6: Novelty analysis.

Two complementary approaches:
  A) Sequence-level novelty: BLAST vs NCBI NR (online, no local DB needed)
  B) Structure-level novelty: FoldSeek vs PDB (requires FoldSeek installation)

API verified:
  Bio.Blast.NCBIWWW.qblast("blastp", "nr", seq, ...)
    → handle parsed by NCBIXML.read() → .alignments[].hsps[].identities
    Set Bio.Blast.email first (NCBI requirement).

  FoldSeek: foldseek easy-search query.pdb database aln.tsv tmp/ ...
    Reuses ReQFlow's run_foldseek_parallel.sh for parallel execution.

Usage:
  # Sequence novelty (BLAST — online, no install beyond biopython):
  python protein_DIFF/eval/metrics_novelty.py blast \
      --csv result/predict/predict.csv \
      --output result/novelty_blast.json

  # Structure novelty (FoldSeek — needs foldseek + PDB database):
  python protein_DIFF/eval/metrics_novelty.py foldseek \
      --pdb_list result/All_Sampled_PDB.txt \
      --designable_list result/All_Sampled_PDB_Designable.txt \
      --script_path protein_DIFF/eval/run_foldseek_parallel.sh \
      --dataset_dir /path/to/FoldSeek_PDB_Database
"""

import argparse, json, os, subprocess, sys, time
import numpy as np
import pandas as pd

from io import StringIO

# ═══════════════════════════════════════════════════════════════
# Common helpers
# ═══════════════════════════════════════════════════════════════

def load_sequences_from_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    return {str(row["id"]): row["seq"] for _, row in df.iterrows()}


# ═══════════════════════════════════════════════════════════════
# A) Sequence-level novelty: BLAST vs NCBI NR
# ═══════════════════════════════════════════════════════════════

def blast_against_nr(seqs: dict, wt_seq: str = None, max_seqs: int = None,
                     email: str = None, hitlist_size: int = 10,
                     output_path: str = None) -> dict:
    """BLAST each sequence against NCBI NR database (online).

    API (BioPython):
      from Bio.Blast import NCBIWWW
      handle = NCBIWWW.qblast("blastp", "nr", seq, hitlist_size=10, expect=10.0)
      from Bio.Blast import NCBIXML
      record = NCBIXML.read(handle)

    IMPORTANT: NCBI requires an email address. Set:
      from Bio import Blast
      Blast.email = "your@email.com"

    The paper reports: "sequence identity < 40% compared to other WT
    proteins from NCBI (except for the template)" as a novelty metric.

    Template exclusion: hits with > 95% identity over > 90% of the query
    length are treated as self/template matches and excluded from max_identity.

    For each sequence, we extract:
      - max_identity: highest % identity to any hit in NR (template excluded)
      - top_hit_description: description of the best-matching non-template
      - num_hits: number of significant hits (e-value < 0.001)

    Note: NCBI BLAST API has rate limits. For 100 sequences, this
    can take 30-60 minutes.
    """
    try:
        from Bio import Blast
        from Bio.Blast import NCBIWWW, NCBIXML
    except ImportError:
        raise ImportError(
            "BioPython is required. Install:  pip install biopython"
        )

    if email:
        Blast.email = email

    ids = sorted(seqs.keys(), key=lambda x: int(x))
    if max_seqs:
        ids = ids[:max_seqs]

    results = {}
    for i, sid in enumerate(ids):
        seq = seqs[sid]
        print(f"  [BLAST {i+1}/{len(ids)}] sequence {sid} ({len(seq)} AA)...")

        try:
            handle = NCBIWWW.qblast(
                "blastp", "nr", seq,
                hitlist_size=hitlist_size,
                expect=10.0,
                format_type="XML",
            )
            blast_xml = handle.read()
            handle.close()

            record = NCBIXML.read(StringIO(blast_xml))

            hits = []
            for alignment in record.alignments:
                for hsp in alignment.hsps:
                    identity_pct = (hsp.identities / hsp.align_length * 100
                                    if hsp.align_length > 0 else 0)
                    hits.append({
                        "title": alignment.title[:100],
                        "length": alignment.length,
                        "e_value": float(hsp.expect),
                        "identity_pct": round(identity_pct, 2),
                        "align_length": hsp.align_length,
                    })
                    break  # Only take the best HSP per alignment

            significant = [h for h in hits if h["e_value"] < 0.001]
            # Exclude template self-matches (>95% identity over >90% query length)
            non_template = [
                h for h in significant
                if not (h["identity_pct"] > 95.0
                        and h["align_length"] > len(seq) * 0.9)
            ]
            max_ident = max((h["identity_pct"] for h in non_template),
                            default=0.0)
            top_hit = non_template[0]["title"] if non_template else "no_significant_non_template_hits"

            results[sid] = {
                "max_identity_pct": round(max_ident, 2),
                "num_significant_hits": len(significant),
                "top_hit": top_hit,
                "status": "ok",
            }
            print(f"    max identity vs NR: {max_ident:.1f}%")

        except Exception as e:
            results[sid] = {"status": "error", "error": str(e)}
            print(f"    [FAIL] {e}")

        # Rate limiting: NCBI asks for ~1 request per second for unauthenticated
        time.sleep(2)

    # Summary
    identities = [r["max_identity_pct"] for r in results.values()
                  if r.get("status") == "ok"]
    report = {
        "num_sequences_tested": len(identities),
        "max_identity_pct": {
            "mean": round(float(np.mean(identities)), 2) if identities else 0,
            "std": round(float(np.std(identities)), 2) if identities else 0,
            "min": round(float(np.min(identities)), 2) if identities else 0,
            "max": round(float(np.max(identities)), 2) if identities else 0,
        },
        "novelty_check": {
            "below_40pct_ratio": round(
                float(np.mean([1.0 if i < 40 else 0.0 for i in identities])), 4
            ) if identities else 0,
            "paper_threshold_met": bool(
                np.mean(identities) < 40 if identities else False
            ),
        },
        "per_sequence": results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[BLAST novelty] Report saved → {output_path}")

    return report


# ═══════════════════════════════════════════════════════════════
# B) Structure-level novelty: FoldSeek vs PDB
# ═══════════════════════════════════════════════════════════════

def run_foldseek_novelty(pdb_list_path: str,
                         designable_list_path: str,
                         script_path: str,
                         dataset_dir: str,
                         output_dir: str = None,
                         database: str = "pdb") -> dict:
    """Run FoldSeek structural novelty search.

    Uses ReQFlow's run_foldseek_parallel.sh for parallel execution.
    Searches each generated structure against the PDB database.

    API (FoldSeek):
      foldseek easy-search query.pdb database aln.tsv tmp/ \
        --alignment-type 1 --exhaustive-search --max-seqs 10000000000 \
        --tmscore-threshold 0.0 --format-output query,target,alntmscore,lddt,evalue

    Requirements:
      - FoldSeek installed (conda install -c bioconda foldseek)
      - PDB database created: foldseek databases PDB pdb tmp/
      - GNU parallel: apt install parallel

    Args:
        pdb_list_path: file listing all PDB paths (one per line)
        designable_list_path: file listing designable PDB paths
        script_path: path to run_foldseek_parallel.sh
        dataset_dir: directory containing the FoldSeek PDB database
        output_dir: output directory (default: parent of pdb_list)
        database: database name (default: "pdb")
    """
    if output_dir is None:
        output_dir = os.path.dirname(pdb_list_path) or "."

    result_summary = os.path.join(output_dir, "summary_tmscore.csv")

    cmd = [
        "bash", script_path,
        pdb_list_path,
        designable_list_path,
        output_dir,
        database,
        result_summary,
    ]

    print(f"  [FoldSeek] Running parallel search...")
    print(f"    CWD: {dataset_dir}")
    print(f"    {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, cwd=dataset_dir)
    except FileNotFoundError as e:
        raise ImportError(
            f"FoldSeek or GNU parallel not found: {e}\n"
            "Install: conda install -c bioconda foldseek && apt install parallel"
        )
    except subprocess.CalledProcessError as e:
        print(f"  [FoldSeek] ERROR: {e}")
        return {"status": "error", "error": str(e)}

    # Parse results
    if not os.path.exists(result_summary):
        return {"status": "error", "error": f"No output at {result_summary}"}

    df = pd.read_csv(result_summary)

    # Parse max TM-score (FoldSeek stores TM-score in E-value column per
    # https://github.com/steineggerlab/foldseek/issues/323)
    all_max_tm = []
    designable_max_tm = []

    for _, row in df.iterrows():
        tm_val = row.get("Max TM-score", None)
        if tm_val is None or tm_val == "N/A":
            continue
        try:
            tm = float(tm_val)
        except (ValueError, TypeError):
            continue
        all_max_tm.append(tm)
        if row.get("Designable", 0) == 1:
            designable_max_tm.append(tm)

    report = {
        "num_structures_searched": len(df),
        "all_structures": {
            "count": len(all_max_tm),
            "max_tm_mean": round(float(np.mean(all_max_tm)), 4) if all_max_tm else 0,
            "max_tm_std": round(float(np.std(all_max_tm)), 4) if all_max_tm else 0,
            "novelty_ratio_tm_lt_0.5": round(
                float(np.mean([1.0 if t < 0.5 else 0.0 for t in all_max_tm])), 4
            ) if all_max_tm else 0,
        },
        "designable_only": {
            "count": len(designable_max_tm),
            "max_tm_mean": round(float(np.mean(designable_max_tm)), 4) if designable_max_tm else 0,
            "max_tm_std": round(float(np.std(designable_max_tm)), 4) if designable_max_tm else 0,
            "novelty_ratio_tm_lt_0.5": round(
                float(np.mean([1.0 if t < 0.5 else 0.0 for t in designable_max_tm])), 4
            ) if designable_max_tm else 0,
        },
    }

    report_path = os.path.join(output_dir, "metrics_foldseek.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[FoldSeek novelty] Report saved → {report_path}")

    return report


def print_blast_summary(report: dict):
    print("\n" + "=" * 60)
    print("  CPFlow — Sequence Novelty (BLAST vs NCBI NR)")
    print("=" * 60)
    mi = report.get("max_identity_pct", {})
    nc = report.get("novelty_check", {})
    print(f"\n  Sequences tested: {report.get('num_sequences_tested', 0)}")
    print(f"  Max identity vs NR: mean={mi.get('mean',0):.1f}% ± {mi.get('std',0):.1f}%  "
          f"range=[{mi.get('min',0):.1f}%, {mi.get('max',0):.1f}%]")
    print(f"  Below 40% (novel): {nc.get('below_40pct_ratio', 0):.1%}")
    status = "✅ < 40% (novel)" if nc.get("paper_threshold_met") else "⚠️ > 40%"
    print(f"  Paper threshold: {status}")
    print("\n" + "=" * 60 + "\n")


def print_foldseek_summary(report: dict):
    print("\n" + "=" * 60)
    print("  CPFlow — Structure Novelty (FoldSeek vs PDB)")
    print("=" * 60)
    for label in ["all_structures", "designable_only"]:
        s = report.get(label, {})
        print(f"\n  {label}:")
        print(f"    structures: {s.get('count', 0)}")
        print(f"    max TM-score: mean={s.get('max_tm_mean',0):.4f} ± {s.get('max_tm_std',0):.4f}")
        print(f"    TM < 0.5 (novel): {s.get('novelty_ratio_tm_lt_0.5', 0):.1%}")
    print("\n" + "=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPFlow novelty analysis (BLAST + FoldSeek)")
    sub = parser.add_subparsers(dest="command")

    # BLAST
    blast_p = sub.add_parser("blast", help="Online BLAST vs NCBI NR")
    blast_p.add_argument("--csv", required=True)
    blast_p.add_argument("--max_seqs", type=int, default=5,
                         help="Max sequences to BLAST (rate-limited, default 5)")
    blast_p.add_argument("--blast_email", default=None,
                         help="Email for NCBI BLAST (required by NCBI)")
    blast_p.add_argument("--output", default="result/novelty_blast.json")

    # FoldSeek
    fs_p = sub.add_parser("foldseek", help="FoldSeek structural search vs PDB")
    fs_p.add_argument("--pdb_list", required=True,
                      help="File listing all PDB paths")
    fs_p.add_argument("--designable_list", required=True,
                      help="File listing designable PDB paths")
    fs_p.add_argument("--script_path", required=True,
                      help="Path to run_foldseek_parallel.sh")
    fs_p.add_argument("--dataset_dir", required=True,
                      help="Directory with FoldSeek PDB database")
    fs_p.add_argument("--output_dir", default=None)
    fs_p.add_argument("--database", default="pdb")

    args = parser.parse_args()

    if args.command == "blast":
        seqs = load_sequences_from_csv(args.csv)
        if not seqs:
            print("ERROR: no sequences loaded", file=sys.stderr)
            sys.exit(1)
        report = blast_against_nr(
            seqs, max_seqs=args.max_seqs,
            email=args.blast_email, output_path=args.output,
        )
        print_blast_summary(report)

    elif args.command == "foldseek":
        if not os.path.exists(args.pdb_list):
            print(f"ERROR: pdb_list not found: {args.pdb_list}", file=sys.stderr)
            sys.exit(1)
        report = run_foldseek_novelty(
            args.pdb_list, args.designable_list,
            args.script_path, args.dataset_dir,
            args.output_dir, args.database,
        )
        print_foldseek_summary(report)

    else:
        parser.print_help()
