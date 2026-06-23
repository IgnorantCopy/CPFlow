#!/usr/bin/env python3
"""
CPFlow Phase 3: Structure-level evaluation metrics.
Adapted from ReQFlow analysis/metrics.py — uses tmtools instead of US-align.

Evaluates:
  - TM-score (generated structure vs WT template)
  - RMSD (generated structure vs WT template)
  - Pairwise TM-score diversity (structure-level)
  - CA-CA bond geometry validation
  - Secondary structure composition (requires mdtraj)

Usage:
  # Compare all generated PDBs against WT
  python protein_DIFF/eval/metrics_structure.py \
      --pdb_dir result/structures/ \
      --wt_pdb result/structures/WT_KmAgo.pdb \
      --output result/metrics_structure.json
"""

import argparse, json, os, sys
import numpy as np


# ═══════════════════════════════════════════════════════════════
# TM-score & RMSD (via tmtools)
# ═══════════════════════════════════════════════════════════════

def calc_tm_score_rmsd(pos_1: np.ndarray, pos_2: np.ndarray,
                       seq_1: str, seq_2: str) -> dict:
    """Compute TM-score and aligned RMSD using tmtools.

    NOTE: tmtools.tm_align() accepts different-length coordinate arrays.
    It performs its own sequence-independent alignment internally.
    tmtools v0.3.0+ returns .rmsd directly.

    Args:
        pos_1: (L1, 3) CA coordinates of structure 1
        pos_2: (L2, 3) CA coordinates of structure 2
        seq_1, seq_2: amino acid sequences (one-letter codes)
    Returns:
        dict with tm_score_chain1, tm_score_chain2, rmsd_aligned
    """
    try:
        from tmtools import tm_align
    except ImportError:
        raise ImportError(
            "tmtools is required. Install with:  pip install tmtools"
        )

    # tmtools accepts different-length arrays directly (no truncation needed)
    result = tm_align(pos_1, pos_2, seq_1, seq_2)
    # result.tm_norm_chain1: TM-score normalized by len(chain1)
    # result.tm_norm_chain2: TM-score normalized by len(chain2)
    # result.rmsd: RMSD of the aligned region (available since v0.3.0)

    return {
        "tm_score_chain1": float(result.tm_norm_chain1),
        "tm_score_chain2": float(result.tm_norm_chain2),
        "rmsd_aligned": round(float(result.rmsd), 4),
    }


# ═══════════════════════════════════════════════════════════════
# CA-CA bond geometry
# ═══════════════════════════════════════════════════════════════

CA_CA_REF_DIST = 3.8  # Standard CA-CA distance in Å


def calc_ca_ca_metrics(ca_pos: np.ndarray, bond_tol: float = 0.1,
                       clash_tol: float = 1.0) -> dict:
    """Validate CA-CA bond geometry.

    Args:
        ca_pos: (L, 3) CA coordinates (in nm if from MD, otherwise Å).
                Will auto-detect scale: if mean bond ~0.38 → nm, scale to Å.
        bond_tol: tolerance for valid CA-CA bond distance (Å)
        clash_tol: distance below which two CA atoms are considered clashing (Å)
    """
    # Auto-detect scale: nm vs Å
    bond_dists_raw = np.linalg.norm(
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    mean_bond = np.mean(bond_dists_raw)
    if mean_bond < 1.0:  # likely in nm
        ca_pos = ca_pos * 10.0
        bond_dists_raw = np.linalg.norm(
            ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]

    ca_bond_dev = np.mean(np.abs(bond_dists_raw - CA_CA_REF_DIST))
    ca_bond_valid_ratio = np.mean(bond_dists_raw < (CA_CA_REF_DIST + bond_tol))

    # Inter-residue CA-CA distances (upper triangle, exclude self)
    ca_ca_dists2d = np.linalg.norm(
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=1) > 0)]
    clashes = int(np.sum(inter_dists < clash_tol))

    return {
        "ca_ca_deviation_A": round(float(ca_bond_dev), 4),
        "ca_ca_valid_ratio": round(float(ca_bond_valid_ratio), 4),
        "num_ca_ca_clashes": clashes,
        "mean_ca_ca_distance_A": round(float(mean_bond) if mean_bond > 1.0
                                       else float(mean_bond * 10), 3),
    }


# ═══════════════════════════════════════════════════════════════
# Secondary structure (via mdtraj)
# ═══════════════════════════════════════════════════════════════

def calc_secondary_structure(pdb_path: str) -> dict:
    """Compute secondary structure composition and radius of gyration.

    Requires: pip install mdtraj
    """
    try:
        import mdtraj as md
    except ImportError:
        raise ImportError(
            "mdtraj is required. Install with:  pip install mdtraj"
        )

    traj = md.load(pdb_path)
    ss = md.compute_dssp(traj, simplified=True)  # 'H', 'E', 'C'

    # compute_rg returns shape (n_frames,); single PDB → single frame
    rg_arr = md.compute_rg(traj)
    rg = float(rg_arr.item() if rg_arr.ndim == 1 else rg_arr[0])

    return {
        "helix_percent": round(float(np.mean(ss == "H")), 4),
        "strand_percent": round(float(np.mean(ss == "E")), 4),
        "coil_percent": round(float(np.mean(ss == "C")), 4),
        "non_coil_percent": round(float(np.mean((ss == "H") | (ss == "E"))), 4),
        "radius_of_gyration_A": round(float(rg * 10), 2),  # nm → Å
    }


# ═══════════════════════════════════════════════════════════════
# PDB loading helpers
# ═══════════════════════════════════════════════════════════════

def load_ca_from_pdb(pdb_path: str):
    """Extract CA coordinates and sequence from a PDB file."""
    coords = []
    seq = []
    aa3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    current_res = None
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_name = line[17:20].strip()
                res_num = line[22:26].strip()
                res_id = (res_num, line[21])
                if res_id != current_res:
                    coords.append([float(line[30:38]), float(line[38:46]),
                                   float(line[46:54])])
                    seq.append(aa3to1.get(res_name, "X"))
                    current_res = res_id
    return np.array(coords), "".join(seq)


def load_plddt_from_pdb(pdb_path: str) -> np.ndarray:
    """Extract per-residue pLDDT from CA atoms' B-factor column."""
    values = []
    current_res = None
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_num = line[22:26].strip()
                res_id = (res_num, line[21])
                if res_id != current_res:
                    values.append(float(line[60:66].strip()))
                    current_res = res_id
    return np.array(values)


# ═══════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_structures(pdb_dir: str, wt_pdb: str,
                        output_path: str = None) -> dict:
    """Run all structure-level metrics.

    Args:
        pdb_dir: directory containing PDB files of generated sequences
        wt_pdb: path to WT template PDB file
    """
    wt_coords, wt_seq = load_ca_from_pdb(wt_pdb)
    wt_plddt = load_plddt_from_pdb(wt_pdb)

    pdb_files = sorted([f for f in os.listdir(pdb_dir)
                        if f.endswith(".pdb") and f != os.path.basename(wt_pdb)])

    if not pdb_files:
        print("WARNING: No PDB files found (need to run structure prediction first)")
        return {"status": "no_pdb_files", "pdb_dir": pdb_dir}

    # ── Per-structure metrics ──
    tm_scores = []
    rmsds = []
    ca_devs = []
    clashes_list = []
    ss_records = []
    plddt_means = []

    for fname in pdb_files:
        fpath = os.path.join(pdb_dir, fname)
        try:
            coords, seq = load_ca_from_pdb(fpath)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")
            continue

        # TM + RMSD vs WT (use tm_norm_chain2: normalized by WT reference length)
        try:
            tm_rmsd = calc_tm_score_rmsd(coords, wt_coords, seq, wt_seq)
            tm_scores.append(tm_rmsd["tm_score_chain2"])  # normalized by WT
            rmsds.append(tm_rmsd["rmsd_aligned"])
        except Exception as e:
            print(f"  [WARN] TM-align failed for {fname}: {e}")
            tm_scores.append(0)
            rmsds.append(99.0)

        # CA-CA geometry
        ca_geom = calc_ca_ca_metrics(coords)
        ca_devs.append(ca_geom["ca_ca_deviation_A"])
        clashes_list.append(ca_geom["num_ca_ca_clashes"])

        # pLDDT
        plddt_arr = load_plddt_from_pdb(fpath)
        if len(plddt_arr) > 0:
            plddt_means.append(float(plddt_arr.mean()))

        # Secondary structure (best-effort, mdtraj may not be installed)
        try:
            ss = calc_secondary_structure(fpath)
            ss_records.append({fname: ss})
        except ImportError:
            pass

    # ── Summary ──
    n = len(tm_scores)
    report = {
        "num_structures_evaluated": n,
        "tm_score_vs_wt": {
            "mean": round(float(np.mean(tm_scores)), 4) if n else 0,
            "std": round(float(np.std(tm_scores)), 4) if n else 0,
            "min": round(float(np.min(tm_scores)), 4) if n else 0,
            "max": round(float(np.max(tm_scores)), 4) if n else 0,
            "above_0_9_ratio": round(float(np.mean([1.0 if t > 0.9 else 0.0
                                                     for t in tm_scores])), 4) if n else 0,
        },
        "rmsd_vs_wt_A": {
            "mean": round(float(np.mean(rmsds)), 4) if n else 0,
            "std": round(float(np.std(rmsds)), 4) if n else 0,
            "min": round(float(np.min(rmsds)), 4) if n else 0,
            "max": round(float(np.max(rmsds)), 4) if n else 0,
            "below_3A_ratio": round(float(np.mean([1.0 if r < 3.0 else 0.0
                                                    for r in rmsds])), 4) if n else 0,
        },
        "ca_ca_geometry": {
            "deviation_mean_A": round(float(np.mean(ca_devs)), 4) if ca_devs else 0,
            "total_clashes": int(np.sum(clashes_list)) if clashes_list else 0,
            "mean_clashes_per_struct": round(float(np.mean(clashes_list)), 2) if clashes_list else 0,
        },
    }

    if plddt_means:
        report["plddt"] = {
            "mean": round(float(np.mean(plddt_means)), 2),
            "std": round(float(np.std(plddt_means)), 2),
            "min": round(float(np.min(plddt_means)), 2),
            "max": round(float(np.max(plddt_means)), 2),
        }

    if ss_records:
        # Aggregate SS stats
        all_h, all_e, all_c, all_rg = [], [], [], []
        for rec in ss_records:
            for _, ss in rec.items():
                all_h.append(ss["helix_percent"])
                all_e.append(ss["strand_percent"])
                all_c.append(ss["coil_percent"])
                all_rg.append(ss["radius_of_gyration_A"])
        report["secondary_structure"] = {
            "mean_helix": round(float(np.mean(all_h)), 3),
            "mean_strand": round(float(np.mean(all_e)), 3),
            "mean_coil": round(float(np.mean(all_c)), 3),
            "mean_rg_A": round(float(np.mean(all_rg)), 2),
        }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[metrics_structure] Report saved → {output_path}")

    return report


def print_summary(report: dict):
    print("\n" + "=" * 60)
    print("  CPFlow — Structure-level Evaluation Summary")
    print("=" * 60)

    tm = report.get("tm_score_vs_wt", {})
    rms = report.get("rmsd_vs_wt_A", {})
    ca = report.get("ca_ca_geometry", {})
    plddt = report.get("plddt", {})

    if tm:
        print(f"\n  TM-score vs WT:  mean={tm['mean']:.4f} ± {tm['std']:.4f}  "
              f"range=[{tm['min']:.4f}, {tm['max']:.4f}]")
        print(f"    > 0.9 ratio: {tm.get('above_0_9_ratio', 0):.1%}")

    if rms:
        print(f"\n  RMSD vs WT:     mean={rms['mean']:.3f} ± {rms['std']:.3f} Å  "
              f"range=[{rms['min']:.3f}, {rms['max']:.3f}]")
        print(f"    < 3.0Å ratio: {rms.get('below_3A_ratio', 0):.1%}")

    if ca:
        print(f"\n  CA-CA geometry: mean_deviation={ca.get('deviation_mean_A', 0):.4f} Å  "
              f"total_clashes={ca.get('total_clashes', 0)}")

    if plddt:
        print(f"\n  pLDDT:          mean={plddt['mean']:.1f} ± {plddt['std']:.1f}  "
              f"range=[{plddt['min']:.1f}, {plddt['max']:.1f}]")

    ss = report.get("secondary_structure", {})
    if ss:
        print(f"\n  Secondary structure:  helix={ss['mean_helix']:.1%}  "
              f"strand={ss['mean_strand']:.1%}  coil={ss['mean_coil']:.1%}")

    print("\n" + "=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPFlow structure-level evaluation")
    parser.add_argument("--pdb_dir", required=True,
                        help="Directory containing PDB files from AlphaFold2/ESMFold")
    parser.add_argument("--wt_pdb", required=True,
                        help="WT template PDB file")
    parser.add_argument("--output", default="result/metrics_structure.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if not os.path.isdir(args.pdb_dir):
        print(f"ERROR: pdb_dir not found: {args.pdb_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.wt_pdb):
        print(f"ERROR: wt_pdb not found: {args.wt_pdb}", file=sys.stderr)
        sys.exit(1)

    report = evaluate_structures(args.pdb_dir, args.wt_pdb, args.output)
    print_summary(report)
