#!/usr/bin/env python3
"""
CPFlow Phase 4: Structure prediction & pLDDT three-step filtering.

Supports two engines:
  - ESMFold  (pip install esm)           — fast, no database needed
  - AlphaFold2 via ColabFold             — paper-grade accuracy
    Install: pip install colabfold alphafold-colabfold
    Paper used: AlphaFold2 single-sequence mode with MSA + PDB templates

Usage:
  # ESMFold (fast, recommended for initial screening):
  python protein_DIFF/eval/predict_structures.py predict \
      --fasta_dir result/predict/fasta/ \
      --output_dir result/structures/ \
      --engine esmfold

  # AlphaFold2 via ColabFold (paper-grade, needs GPU + databases):
  python protein_DIFF/eval/predict_structures.py predict \
      --fasta_dir result/predict/fasta/ \
      --output_dir result/structures/ \
      --engine alphafold

  # Three-step pLDDT filter:
  python protein_DIFF/eval/predict_structures.py filter \
      --pdb_dir result/structures/ \
      --wt_pdb result/structures/WT.pdb
"""

import argparse, json, os, subprocess, sys, time, tempfile
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Engine: ESMFold
# ═══════════════════════════════════════════════════════════════

def _predict_esmfold(fasta_dir: str, output_dir: str, device: str = "cuda:0",
                     chunk_size: int = None) -> list:
    """
    ESMFold batch prediction.

    API verified from facebookresearch/esm README:
      model = esm.pretrained.esmfold_v1()
      model = model.eval().cuda()
      model.set_chunk_size(N)        # optional, reduces memory for long seqs
      output = model.infer_pdb(seq)  # returns PDB string

    For ~800 AA sequences (like KmAgo), set chunk_size=64 or 128
    to avoid OOM on consumer GPUs (24 GB).
    """
    try:
        import esm
        import torch
    except ImportError:
        raise ImportError(
            "ESMFold requires:  pip install esm\n"
            "See: https://github.com/facebookresearch/esm"
        )

    os.makedirs(output_dir, exist_ok=True)

    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)

    if chunk_size is not None:
        model.set_chunk_size(chunk_size)

    fasta_files = sorted([f for f in os.listdir(fasta_dir)
                          if f.endswith(".fasta")])
    results = []

    for fname in fasta_files:
        fpath = os.path.join(fasta_dir, fname)
        pdb_path = os.path.join(output_dir, fname.replace(".fasta", ".pdb"))

        if os.path.exists(pdb_path):
            results.append({"id": fname, "status": "cached", "pdb": pdb_path})
            continue

        with open(fpath) as f:
            lines = f.readlines()
        seq = "".join(l.strip() for l in lines[1:])

        try:
            with torch.no_grad():
                pdb_str = model.infer_pdb(seq)
            with open(pdb_path, "w") as f:
                f.write(pdb_str)
            results.append({"id": fname, "status": "ok", "pdb": pdb_path,
                            "length": len(seq)})
            print(f"  [OK] {fname} → {pdb_path}")
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] {fname} ({len(seq)} AA): GPU out of memory. "
                  f"Try --chunk_size smaller or use --engine alphafold")
            results.append({"id": fname, "status": "error",
                            "error": "CUDA OOM"})
        except Exception as e:
            results.append({"id": fname, "status": "error", "error": str(e)})
            print(f"  [FAIL] {fname}: {e}")

    # Save prediction log
    log_path = os.path.join(output_dir, "prediction_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
    fail = sum(1 for r in results if r["status"] == "error")
    print(f"\n[predict_esmfold] Done: {ok} success, {fail} failed")
    return results


# ═══════════════════════════════════════════════════════════════
# Engine: AlphaFold2 via ColabFold
# ═══════════════════════════════════════════════════════════════

def _predict_alphafold(fasta_dir: str, output_dir: str):
    """
    AlphaFold2 via ColabFold batch prediction.

    Paper config (from Materials and Methods):
      - "AlphaFold2 in single-sequence mode with MSA and with PDB templates"
      - "the highest-ranked predicted structure among the five models was used"

    ColabFold CLI API (verified from sokrypton/ColabFold README):
      colabfold_batch INPUT OUTPUT [OPTIONS]
        --templates                  # enable PDB template query (default=False)
        --num-recycle N              # default 3
        --num-models N               # paper uses 5
        --model-type alphafold2_ptm  # monomer with pLDDT output
        --use-gpu-relax              # optional AMBER relaxation

    Output structure:
      output_dir/
        seq_name/
          seq_name_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
          seq_name_unrelaxed_rank_002_...pdb
          ...
          seq_name_scores_rank_001_alphafold2_ptm_model_1_seed_000.json

    Requirement:
      pip install colabfold alphafold-colabfold
      May also need: pip install jax jaxlib (platform-specific)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check colabfold_batch is available
    try:
        result = subprocess.run(["colabfold_batch", "--help"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError("colabfold_batch returned non-zero")
    except FileNotFoundError:
        raise ImportError(
            "AlphaFold2 via ColabFold requires:  pip install colabfold alphafold-colabfold\n"
            "See: https://github.com/sokrypton/ColabFold\n"
            "Or use --engine esmfold for a zero-database alternative."
        )

    fasta_files = sorted([f for f in os.listdir(fasta_dir)
                          if f.endswith(".fasta")])
    results = []

    for fname in fasta_files:
        fpath = os.path.join(fasta_dir, fname)
        seq_id = fname.replace(".fasta", "")
        out_subdir = os.path.join(output_dir, seq_id)

        # Check if already done (rank_001 PDB exists)
        expected_pdb = os.path.join(
            out_subdir,
            f"{seq_id}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
        )
        if os.path.exists(expected_pdb):
            results.append({"id": fname, "status": "cached", "pdb": expected_pdb})
            continue

        # Run colabfold_batch
        # Paper: "with MSA and with PDB templates" / "highest-ranked among five models"
        # --templates enables PDB template query (default=False, must be explicit)
        cmd = [
            "colabfold_batch",
            fpath,
            out_subdir,
            "--templates",
            "--num-recycle", "3",
            "--num-models", "5",
            "--model-type", "alphafold2_ptm",
        ]
        print(f"  [AF2] Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            # Find the rank_001 PDB
            pdb_files = sorted([
                f for f in os.listdir(out_subdir)
                if f.endswith(".pdb") and "rank_001" in f
            ])
            if pdb_files:
                pdb_path = os.path.join(out_subdir, pdb_files[0])
                results.append({"id": fname, "status": "ok", "pdb": pdb_path})
                print(f"  [OK] {fname} → {pdb_path}")
            else:
                results.append({"id": fname, "status": "error",
                                "error": "No rank_001 PDB found"})
        except subprocess.CalledProcessError as e:
            results.append({"id": fname, "status": "error",
                            "error": f"colabfold_batch failed: {e}"})
            print(f"  [FAIL] {fname}: {e}")

    log_path = os.path.join(output_dir, "prediction_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
    fail = sum(1 for r in results if r["status"] == "error")
    print(f"\n[predict_alphafold] Done: {ok} success, {fail} failed")
    return results


# ═══════════════════════════════════════════════════════════════
# Engine dispatch
# ═══════════════════════════════════════════════════════════════

def predict_structures(fasta_dir: str, output_dir: str, engine: str = "esmfold",
                       device: str = "cuda:0", chunk_size: int = None):
    """Dispatch to the selected prediction engine."""
    if engine == "esmfold":
        return _predict_esmfold(fasta_dir, output_dir, device, chunk_size)
    elif engine in ("alphafold", "colabfold"):
        return _predict_alphafold(fasta_dir, output_dir)
    else:
        raise ValueError(f"Unknown engine: {engine}. Use 'esmfold' or 'alphafold'.")


# ═══════════════════════════════════════════════════════════════
# pLDDT extraction (shared across engines — both write to B-factor)
# ═══════════════════════════════════════════════════════════════

def extract_per_residue_plddt(pdb_path: str) -> np.ndarray:
    """Extract per-residue pLDDT from CA atoms' B-factor column.

    Works for both AlphaFold2 and ESMFold PDB output.
    Both write pLDDT (0-100) into the B-factor field (columns 61-66).
    """
    values = []
    current_res = None
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_id = (line[22:26].strip(), line[21])
                if res_id != current_res:
                    values.append(float(line[60:66].strip()))
                    current_res = res_id
    return np.array(values)


# ═══════════════════════════════════════════════════════════════
# Three-step pLDDT filter (paper Fig. 2a)
# ═══════════════════════════════════════════════════════════════

def three_step_filter(pdb_dir: str, wt_pdb: str,
                      output_path: str = None) -> dict:
    """Run CPDiffusion's three-step pLDDT filter.

    Step 1: reject sequences with overall pLDDT < mean − 1σ
    Step 2: reject sequences with σ(ΔpLDDT) > mean + 1σ
    Step 3: reject sequences with count(|ΔpLDDT| > 10) > ceil(mean + 1σ)

    """
    def _finish(report: dict, announce: bool = False) -> dict:
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            if announce:
                print(f"[pLDDT filter] Report saved → {output_path}")
        return report

    wt_plddt = extract_per_residue_plddt(wt_pdb)
    if len(wt_plddt) == 0:
        return _finish({
            "status": "wt_pdb_no_ca",
            "error": f"No CA atoms in WT PDB: {wt_pdb}",
        })
    wt_len = len(wt_plddt)

    pdb_files = sorted([f for f in os.listdir(pdb_dir)
                        if f.endswith(".pdb") and f != os.path.basename(wt_pdb)])

    if not pdb_files:
        return _finish({"status": "no_pdb_files", "pdb_dir": pdb_dir})

    all_data = []
    for fname in pdb_files:
        fpath = os.path.join(pdb_dir, fname)
        try:
            arr = extract_per_residue_plddt(fpath)
            if len(arr) == 0:
                raise ValueError(f"No CA atoms found in {fname}")
            overall = float(arr.mean())
            min_len = min(len(arr), wt_len)
            delta = arr[:min_len] - wt_plddt[:min_len]
            sigma_delta = float(np.std(delta))
            # Paper: count(|ΔpLDDT| > 10) where Δ = pLDDT_AP − pLDDT_WT.
            # Counts both AP > WT + 10 and AP < WT − 10 deviations.
            large_diffs = int(np.sum(np.abs(delta) > 10))
            all_data.append({
                "file": fname, "length": len(arr),
                "overall_plddt": round(overall, 2),
                "sigma_delta": round(sigma_delta, 4),
                "large_diffs_count": large_diffs,
            })
        except Exception as e:
            all_data.append({
                "file": fname, "error": str(e),
                "overall_plddt": 0, "sigma_delta": 99, "large_diffs_count": 999,
            })

    n_total = len(all_data)

    # ── Compute thresholds (exclude failed PDBs per M3) ──
    valid = [d for d in all_data if "error" not in d]
    n_valid = len(valid)
    n_failed = n_total - n_valid

    if n_valid == 0:
        return _finish({
            "status": "all_pdb_parse_failed",
            "num_total": n_total,
            "num_valid": 0,
            "num_failed": n_failed,
            "per_structure": all_data,
        })

    overalls = np.array([d["overall_plddt"] for d in valid])
    sigmas = np.array([d["sigma_delta"] for d in valid])
    counts = np.array([d["large_diffs_count"] for d in valid])

    t_overall = float(np.mean(overalls) - np.std(overalls))
    t_sigma = float(np.mean(sigmas) + np.std(sigmas))
    # Paper: "more than 93 AA positions" are excluded → ≤ ceil(μ+σ) passes.
    # Use integer ceiling for both report and filter to be consistent (M2).
    t_count_int = int(np.ceil(np.mean(counts) + np.std(counts)))

    pass_step1 = [d for d in all_data if d["overall_plddt"] >= t_overall]
    pass_step2 = [d for d in pass_step1 if d["sigma_delta"] <= t_sigma]
    pass_step3 = [d for d in pass_step2 if d["large_diffs_count"] <= t_count_int]

    report = {
        "engine_note": "Works for both ESMFold and AlphaFold2 PDB output.",
        "num_total": n_total,
        "num_valid": n_valid,
        "num_failed": n_failed,
        "thresholds": {
            "overall_plddt_min": round(t_overall, 2),
            "sigma_delta_max": round(t_sigma, 4),
            "large_diffs_max": t_count_int,
        },
        "population_stats": {
            "overall_plddt": {
                "mean": round(float(np.mean(overalls)), 2),
                "std": round(float(np.std(overalls)), 2),
                "computed_on": n_valid,
            },
            "sigma_delta": {
                "mean": round(float(np.mean(sigmas)), 4),
                "std": round(float(np.std(sigmas)), 4),
                "computed_on": n_valid,
            },
            "large_diffs_count": {
                "mean": round(float(np.mean(counts)), 2),
                "std": round(float(np.std(counts)), 2),
                "computed_on": n_valid,
            },
        },
        "filter_results": {
            "step_1_overall_plddt": {
                "passed": len(pass_step1),
                "rejected": n_total - len(pass_step1),
                "pass_rate": round(len(pass_step1) / n_total, 4) if n_total else 0,
            },
            "step_2_sigma_delta": {
                "passed": len(pass_step2),
                "rejected": len(pass_step1) - len(pass_step2),
            },
            "step_3_large_diffs": {
                "passed": len(pass_step3),
                "rejected": len(pass_step2) - len(pass_step3),
            },
            "final_pass_rate": round(len(pass_step3) / n_total, 4) if n_total else 0,
        },
        "passed_sequences": [d["file"] for d in pass_step3],
        "rejected_sequences": [d["file"] for d in all_data
                               if d not in pass_step3],
    }

    return _finish(report, announce=True)


def print_filter_summary(report: dict):
    print("\n" + "=" * 60)
    print("  CPFlow — pLDDT Three-Step Filter Summary")
    print("=" * 60)

    t = report.get("thresholds", {})
    pop = report.get("population_stats", {})
    fr = report.get("filter_results", {})

    print(f"\n  Total structures: {report.get('num_total', 0)}")
    print(f"\n  Thresholds (mean ± 1σ):")
    print(f"    Step 1 - overall pLDDT ≥ {t.get('overall_plddt_min', 'N/A')}")
    print(f"    Step 2 - σ(ΔpLDDT) ≤ {t.get('sigma_delta_max', 'N/A')}")
    print(f"    Step 3 - count(|ΔpLDDT| > 10) ≤ {t.get('large_diffs_max', 'N/A')}")

    if pop:
        o = pop.get("overall_plddt", {})
        print(f"\n  Population: overall_plddt  mean={o.get('mean',0):.1f} ± {o.get('std',0):.1f}")

    print(f"\n  Filter cascade:")
    for name in ["step_1_overall_plddt", "step_2_sigma_delta", "step_3_large_diffs"]:
        s = fr.get(name, {})
        print(f"    {name}: {s.get('passed','?')} passed, {s.get('rejected','?')} rejected")

    final = fr.get("final_pass_rate", 0)
    step3 = fr.get("step_3_large_diffs", {})
    print(f"\n  Final pass rate: {final:.1%} ({step3.get('passed','?')}/{report.get('num_total','?')})")
    print("\n" + "=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPFlow structure prediction & pLDDT filtering")
    sub = parser.add_subparsers(dest="command")

    # predict
    pred = sub.add_parser("predict", help="Batch structure prediction from FASTA")
    pred.add_argument("--fasta_dir", required=True)
    pred.add_argument("--output_dir", required=True)
    pred.add_argument("--engine", default="esmfold",
                      choices=["esmfold", "alphafold"],
                      help="ESMFold (fast, no DB) | AlphaFold2 (paper-grade)")
    pred.add_argument("--device", default="cuda:0")
    pred.add_argument("--chunk_size", type=int, default=None,
                      help="ESMFold chunk size (reduce for OOM, e.g. 64)")

    # filter
    filt = sub.add_parser("filter", help="Three-step pLDDT filtering")
    filt.add_argument("--pdb_dir", required=True)
    filt.add_argument("--wt_pdb", required=True)
    filt.add_argument("--output", default="result/metrics_plddt.json")

    args = parser.parse_args()

    if args.command == "predict":
        predict_structures(
            args.fasta_dir, args.output_dir, args.engine,
            args.device, args.chunk_size,
        )

    elif args.command == "filter":
        if not os.path.isdir(args.pdb_dir):
            print(f"ERROR: pdb_dir not found: {args.pdb_dir}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(args.wt_pdb):
            print(f"ERROR: wt_pdb not found: {args.wt_pdb}", file=sys.stderr)
            print("  Run prediction on the WT FASTA first.", file=sys.stderr)
            sys.exit(1)
        report = three_step_filter(args.pdb_dir, args.wt_pdb, args.output)
        print_filter_summary(report)
        if report.get("status") in {"wt_pdb_no_ca", "no_pdb_files", "all_pdb_parse_failed"}:
            sys.exit(1)

    else:
        parser.print_help()
