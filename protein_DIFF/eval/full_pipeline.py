#!/usr/bin/env python3
"""
CPFlow full evaluation pipeline — runs all phases in sequence.

Usage:
  # Full pipeline (all phases):
  python protein_DIFF/eval/full_pipeline.py \
      --csv result/predict/predict.csv \
      --wt_fasta dataset/Ago/wt_kmago.fasta \
      --output_dir result/eval_full/

  # Skip structure prediction (if already done):
  python protein_DIFF/eval/full_pipeline.py ... --skip_predict
"""

import argparse, json, os, subprocess, sys, time
from datetime import datetime


def _run_script(name, *args) -> bool:
    """Run a Python script with args, return success."""
    cmd = [sys.executable, "-m", f"protein_DIFF.eval.{name}"] + list(args)
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"  [FAILED] {name}")
        return False
    except FileNotFoundError:
        print(f"  [SKIP] {name} — not available (missing dependencies)")
        return False


def main():
    parser = argparse.ArgumentParser(description="CPFlow full evaluation pipeline")
    parser.add_argument("--csv", required=True, help="Path to predict.csv")
    parser.add_argument("--wt_fasta", required=True, help="WT template FASTA")
    parser.add_argument("--wt_pdb", default=None, help="WT template PDB (optional)")
    parser.add_argument("--output_dir", default="result/eval_full",
                        help="Output directory for all evaluation results")
    parser.add_argument("--fasta_dir", default=None,
                        help="Directory with FASTA files. "
                             "Default: parent-of-csv/fasta/")
    parser.add_argument("--pdb_dir", default=None,
                        help="Directory with predicted PDBs. "
                             "Default: output_dir/structures/")
    parser.add_argument("--engine", default="esmfold",
                        choices=["esmfold", "alphafold"],
                        help="Structure prediction engine")
    parser.add_argument("--training_csv", default=None,
                        help="Training output CSV from run_pt (Phase 2)")
    parser.add_argument("--compare_csv", default=None,
                        help="Comparison training CSV (Phase 2)")
    parser.add_argument("--blast_max_seqs", type=int, default=3,
                        help="Max sequences to BLAST (online, rate-limited)")
    parser.add_argument("--blast_email", default=None,
                        help="Email for NCBI BLAST")
    parser.add_argument("--skip_predict", action="store_true",
                        help="Skip structure prediction (use existing PDBs)")
    parser.add_argument("--skip_blast", action="store_true",
                        help="Skip online BLAST (slow)")
    parser.add_argument("--skip_foldseek", action="store_true",
                        help="Skip FoldSeek (needs large PDB database)")
    parser.add_argument("--motif", default="kmago", choices=["kmago", "pfago"],
                        help="Which catalytic motif to check")
    parser.add_argument("--fix_pos_file", default=None,
                        help="Fix positions file (e.g. dataset/Ago/pfago.piwi.fix.txt)")
    parser.add_argument("--motif_positions", default=None,
                        help="Comma-separated 1-indexed motif positions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.pdb_dir is None:
        args.pdb_dir = os.path.join(args.output_dir, "structures")
    if args.wt_pdb is None:
        args.wt_pdb = os.path.join(args.pdb_dir, "WT.pdb")
    if args.fasta_dir is None:
        args.fasta_dir = os.path.join(os.path.dirname(args.csv), "fasta")

    results = {}
    t_start = time.time()

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Sequence metrics (zero deps, always runs first)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "#" * 60)
    print("#  PHASE 1: Sequence Metrics")
    print("#" * 60)

    seq_out = os.path.join(args.output_dir, "metrics_sequence.json")
    seq_args = ["--csv", args.csv, "--wt_fasta", args.wt_fasta,
                "--motif", args.motif, "--output", seq_out]
    if args.fix_pos_file:
        seq_args += ["--fix_pos_file", args.fix_pos_file]
    if args.motif_positions:
        seq_args += ["--motif_positions", args.motif_positions]
    ok = _run_script("metrics_sequence", *seq_args)
    results["sequence"] = {"output": seq_out, "ok": ok}

    if not ok:
        print("\n[FATAL] Phase 1 failed. Stopping pipeline — fix sequence data first.")
        sys.exit(1)

    # ── Phase 1b: AA property preservation ──
    print("\n" + "#" * 60)
    print("#  PHASE 1b: AA Property Preservation")
    print("#" * 60)
    aa_out = os.path.join(args.output_dir, "metrics_aa_properties.json")
    ok = _run_script("metrics_aa_properties",
                     "--csv", args.csv,
                     "--wt_fasta", args.wt_fasta,
                     "--output", aa_out)
    results["aa_properties"] = {"output": aa_out, "ok": ok}

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Efficiency (if training CSV provided)
    # ═══════════════════════════════════════════════════════════
    if args.training_csv:
        print("\n" + "#" * 60)
        print("#  PHASE 2: Efficiency Metrics")
        print("#" * 60)
        eff_out = os.path.join(args.output_dir, "metrics_efficiency.json")
        eff_args = ["--training_csv", args.training_csv,
                    "--output", eff_out]
        if args.compare_csv:
            eff_args += ["--compare_csv", args.compare_csv]
        ok = _run_script("metrics_efficiency", *eff_args)
        results["efficiency"] = {"output": eff_out, "ok": ok}
    else:
        print("\n[SKIP] Phase 2 — no --training_csv provided")

    # ═══════════════════════════════════════════════════════════
    # Phase 4a: Structure prediction (if not skipped)
    # ═══════════════════════════════════════════════════════════
    if not args.skip_predict:
        print("\n" + "#" * 60)
        print(f"#  PHASE 4a: Structure Prediction ({args.engine})")
        print("#" * 60)

        # Predict WT first, then find and rename the output PDB
        import shutil, glob
        wt_dir = os.path.join(args.output_dir, "wt_structure")
        os.makedirs(wt_dir, exist_ok=True)
        wt_fa_tmp = os.path.join(wt_dir, "WT.fasta")
        shutil.copy(args.wt_fasta, wt_fa_tmp)
        pred = _run_script("predict_structures", "predict",
                           "--fasta_dir", wt_dir,
                           "--output_dir", wt_dir,
                           "--engine", args.engine)
        # ESMFold may directly output WT.pdb, or a variant name.
        # Find any .pdb in wt_dir and use as WT reference.
        # Also searches subdirectories (ColabFold pattern: WT/WT_unrelaxed_rank_001_....pdb)
        wt_pdbs = [f for f in os.listdir(wt_dir) if f.endswith(".pdb")]
        if not wt_pdbs:
            import glob as _g
            wt_pdbs = [os.path.relpath(p, wt_dir) for p in
                       _g.glob(os.path.join(wt_dir, "*", "*rank_001*.pdb"))]
        if wt_pdbs:
            if "WT.pdb" in wt_pdbs:
                args.wt_pdb = os.path.join(wt_dir, "WT.pdb")
            else:
                src = os.path.join(wt_dir, wt_pdbs[0])
                dst = os.path.join(wt_dir, "WT.pdb")
                shutil.move(src, dst)
                args.wt_pdb = dst
            print(f"  WT PDB: {args.wt_pdb}")
        else:
            print("  [WARN] WT prediction produced no PDB. "
                  "Provide --wt_pdb manually for Phase 3-4.")

        # Predict generated sequences
        pred = _run_script("predict_structures", "predict",
                           "--fasta_dir", args.fasta_dir,
                           "--output_dir", args.pdb_dir,
                           "--engine", args.engine)
        results["predict"] = {"output": args.pdb_dir, "ok": pred}

        # If AlphaFold: flatten subdirectories (AF2 outputs PDBs in seq_id/ subdirs)
        if args.engine == "alphafold" and pred:
            import glob as _glob
            for sub in os.listdir(args.pdb_dir):
                sub_path = os.path.join(args.pdb_dir, sub)
                if os.path.isdir(sub_path):
                    for pdb in _glob.glob(os.path.join(sub_path, "*rank_001*.pdb")):
                        dst = os.path.join(args.pdb_dir, os.path.basename(pdb))
                        if not os.path.exists(dst):
                            shutil.copy(pdb, dst)
            print("  [AF2] Flattened rank_001 PDBs to top-level pdb_dir.")
    else:
        print("\n[SKIP] Phase 4a — --skip_predict")

    # ═══════════════════════════════════════════════════════════
    # Phase 4b: pLDDT three-step filter
    # ═══════════════════════════════════════════════════════════
    if os.path.exists(args.wt_pdb) and os.path.isdir(args.pdb_dir):
        print("\n" + "#" * 60)
        print("#  PHASE 4b: pLDDT Three-Step Filter")
        print("#" * 60)
        plddt_out = os.path.join(args.output_dir, "metrics_plddt.json")
        ok = _run_script("predict_structures", "filter",
                         "--pdb_dir", args.pdb_dir,
                         "--wt_pdb", args.wt_pdb,
                         "--output", plddt_out)
        results["plddt"] = {"output": plddt_out, "ok": ok}
    else:
        print("\n[SKIP] Phase 4b — need WT PDB and structure predictions")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Structure metrics (TM-score, RMSD, CA-CA, SS)
    # ═══════════════════════════════════════════════════════════
    if os.path.exists(args.wt_pdb) and os.path.isdir(args.pdb_dir):
        print("\n" + "#" * 60)
        print("#  PHASE 3: Structure Metrics")
        print("#" * 60)
        struct_out = os.path.join(args.output_dir, "metrics_structure.json")
        ok = _run_script("metrics_structure",
                         "--pdb_dir", args.pdb_dir,
                         "--wt_pdb", args.wt_pdb,
                         "--plddt_json", plddt_out,
                         "--output", struct_out)
        results["structure"] = {"output": struct_out, "ok": ok}
    else:
        print("\n[SKIP] Phase 3 — need WT PDB and structure predictions")

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Phylogeny & conservation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "#" * 60)
    print("#  PHASE 5: Phylogeny & Conservation")
    print("#" * 60)
    phy_out = os.path.join(args.output_dir, "phylogeny")
    ok = _run_script("metrics_phylogeny",
                     "--csv", args.csv,
                     "--wt_fasta", args.wt_fasta,
                     "--output_dir", phy_out)
    results["phylogeny"] = {"output": phy_out, "ok": ok}

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Novelty
    # ═══════════════════════════════════════════════════════════
    if not args.skip_blast:
        print("\n" + "#" * 60)
        print("#  PHASE 6a: Sequence Novelty (BLAST)")
        print("#" * 60)
        blast_out = os.path.join(args.output_dir, "novelty_blast.json")
        blast_args = ["blast", "--csv", args.csv,
                      "--max_seqs", str(args.blast_max_seqs),
                      "--output", blast_out]
        if args.blast_email:
            blast_args += ["--blast_email", args.blast_email]
        ok = _run_script("metrics_novelty", *blast_args)
        results["blast"] = {"output": blast_out, "ok": ok}
    else:
        print("\n[SKIP] Phase 6a — --skip_blast")

    if not args.skip_foldseek:
        print("\n" + "#" * 60)
        print("#  PHASE 6b: Structure Novelty (FoldSeek)")
        print("#" * 60)
        print("  [INFO] FoldSeek requires a prepared PDB database.")
        print("  [INFO] Run manually:")
        print(f"    python protein_DIFF/eval/metrics_novelty.py foldseek \\")
        print(f"      --pdb_list {args.pdb_dir}/all_pdbs.txt \\")
        print(f"      --script_path protein_DIFF/eval/run_foldseek_parallel.sh \\")
        print(f"      --dataset_dir /path/to/FoldSeek_PDB_Database")
    else:
        print("\n[SKIP] Phase 6b — --skip_foldseek")

    # ═══════════════════════════════════════════════════════════
    # Final summary
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t_start

    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": args.output_dir,
        "phases": results,
    }

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "#" * 60)
    print("#  EVALUATION COMPLETE")
    print("#" * 60)
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"  Results: {args.output_dir}/")
    for phase, r in results.items():
        status = "✅" if r["ok"] else "❌"
        print(f"    {status} {phase}: {r['output']}")
    print(f"\n  Summary: {summary_path}")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
