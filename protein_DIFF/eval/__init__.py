# CPFlow evaluation package — Phase 1-6 + Gap Fill
#
# ── Zero-dependency ──
#   metrics_sequence.py       Phase 1: catalytic motif, sequence identity, diversity
#   metrics_aa_properties.py  Phase 1b: AA property preservation (paper Supp Fig. S4)
#
# ── torch only ──
#   metrics_efficiency.py     Phase 2: param count, GPU memory, inference timing
#
# ── pip install tmtools mdtraj ──
#   metrics_structure.py      Phase 3: TM-score, RMSD, CA-CA geometry, SS
#     API: tmtools v0.3.0 → tm_align().rmsd, .tm_norm_chain1/2
#          mdtraj → compute_dssp(), compute_rg()
#
# ── pip install esm  OR  pip install colabfold ──
#   predict_structures.py     Phase 4: ESMFold/AlphaFold2 prediction + pLDDT filter
#     API: esm.pretrained.esmfold_v1().infer_pdb(seq)
#          colabfold_batch --msa-mode single_sequence
#
# ── conda install muscle iqtree ──
#   metrics_phylogeny.py      Phase 5: MUSCLE MSA + IQ-TREE + R_seq conservation
#     API: muscle -align in.fa -output out.afa
#          iqtree -s aln.fa -m BLOSUM62 -B 1500 -T AUTO
#
# ── pip install biopython / conda install foldseek ──
#   metrics_novelty.py        Phase 6: BLAST NCBI NR + FoldSeek PDB
#     API: Bio.Blast.NCBIWWW.qblast("blastp","nr",seq)
#          foldseek easy-search query db output tmp/
#
# ── requires model checkpoint + ProteinGym ──
#   metrics_spearman.py       Gap fill: mutation effect Spearman correlation
#     Wraps run_pt.compute_single_site_corr_score_all()
#
# ── full pipeline ──
#   full_pipeline.py          Orchestrates all phases
#   run_foldseek_parallel.sh  FoldSeek parallel runner (from ReQFlow)
#
# Paper correspondence:
#   Phase 1    → Fig 2c, Fig 5b-c: sequence identity, catalytic motif, diversity
#   Phase 1b   → Supp Fig S4: AA property preservation
#   Phase 2    → Training monitoring
#   Phase 3    → Supp Fig S41-S42: RMSD, TM-score
#   Phase 4    → Fig 2a-b: pLDDT three-step filter
#   Phase 5    → Fig 5a: phylogenetic tree, conservation
#   Phase 6    → Supp Data Section 4: novelty vs NCBI NR and PDB
#   Spearman   → Supp Data Section 4: mutation effect prediction
