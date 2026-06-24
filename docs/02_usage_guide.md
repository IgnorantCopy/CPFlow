# CPFlow 评估工具使用文档

> 版本：Phase 1-6 完整实现
> 代码路径：`protein_DIFF/eval/`

---

## 一、文件总览

```
protein_DIFF/eval/
├── full_pipeline.py              ← 🚀 一键运行全部评估
├── run_foldseek_parallel.sh      ← FoldSeek 并行脚本 (来自 ReQFlow)
│
├── metrics_sequence.py           ← Phase 1: 序列层面指标
├── metrics_efficiency.py         ← Phase 2: 训练效率指标
├── metrics_structure.py          ← Phase 3: 结构质量指标
├── predict_structures.py         ← Phase 4: 结构预测 + pLDDT 筛选
├── metrics_phylogeny.py          ← Phase 5: 保守性 + 系统发育
└── metrics_novelty.py            ← Phase 6: 新颖性分析
```

---

## 二、环境依赖

### 必装（所有环境都需要）

```bash
pip install numpy pandas
```

### 按 Phase 安装

| Phase | 文件 | 安装命令 |
|:--:|------|------|
| 1 | `metrics_sequence.py` | 无额外依赖 |
| 1b | `metrics_aa_properties.py` | 无额外依赖 |
| 2 | `metrics_efficiency.py` | `pip install torch` |
| 3 | `metrics_structure.py` | `pip install "tmtools>=0.3.0" mdtraj` |
| 4 | `predict_structures.py` | 二选一：`pip install esm` (快速) 或 `pip install colabfold` (论文级，MSA+模板+5模型) |
| 5 | `metrics_phylogeny.py` | `conda install -c bioconda muscle iqtree` |
| 6 | `metrics_novelty.py` | `pip install biopython` (BLAST) / `conda install -c bioconda foldseek` (FoldSeek) |
| Spearman | `metrics_spearman.py` | `pip install scipy torch_geometric` + ProteinGym 数据集 |

---

## 三、前置准备

```bash
# 1. 生成序列
python protein_DIFF/inference.py \
    --ckpt ckpt/model.pt \
    --target_protein dataset/Ago/process/AGO_050_model_3_ptm.pt \
    --target_protein_dir dataset/Ago/process/ \
    --gen_num 100 \
    --output_dir result/predict

# 2. 提取 WT 序列（从 graph 文件）
python -c "
import torch
aa = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
graph = torch.load('dataset/Ago/process/AGO_050_model_3_ptm.pt', weights_only=False)
indices = graph.x[:,:20].argmax(dim=1).cpu().numpy()
seq = ''.join(aa[i] for i in indices)
with open('dataset/Ago/wt_kmago.fasta','w') as f:
    f.write(f'>WT_KmAgo\n{seq}\n')
print(f'WT length: {len(seq)}')
"

# 3. 结构预测（ESMFold，推荐先跑这个）
python protein_DIFF/eval/predict_structures.py predict \
    --fasta_dir dataset/Ago/  --output_dir result/wt_structure --engine esmfold
# 找到 WT.pdb 或手动指定

python protein_DIFF/eval/predict_structures.py predict \
    --fasta_dir result/predict/fasta/ --output_dir result/structures --engine esmfold
```

---

## 四、Phase 1：序列层面指标

**零依赖，立即可用。** 评估催化基序完整性、序列一致性和多样性。

```bash
# KmAgo (DEDD 硬编码)
python protein_DIFF/eval/metrics_sequence.py \
    --csv result/predict/predict.csv \
    --wt_fasta dataset/Ago/wt_kmago.fasta \
    --output result/metrics_sequence.json

# PfAgo (DEDH, 从 fix 文件读取)
python protein_DIFF/eval/metrics_sequence.py \
    --csv result/predict/predict.csv \
    --wt_fasta dataset/Ago/wt_pfago.fasta \
    --motif pfago \
    --fix_pos_file dataset/Ago/pfago.piwi.fix.txt \
    --output result/metrics_sequence.json

# 可选: t-SNE 嵌入 (默认关闭, 大样本可能慢/不稳定)
python protein_DIFF/eval/metrics_sequence.py \
    --csv result/predict/predict.csv \
    --wt_fasta dataset/Ago/wt_kmago.fasta \
    --compute_tsne \
    --output result/metrics_sequence.json
```

**输出** (`metrics_sequence.json`):

```json
{
  "num_sequences": 100,
  "catalytic_motif": {
    "motif": "KmAgo DEDD",
    "intact_ratio": 1.0,
    "intact": true
  },
  "identity_vs_wt": {
    "mean": 0.72, "std": 0.02,
    "range": [0.68, 0.77]
  },
  "pairwise_identity": {
    "mean": 0.55, "std": 0.08
  },
  "diversity_check": {
    "wt_identity_in_range_50_70": 0.85,
    "pairwise_identity_below_80": 0.98
  }
}
```

**关键解读**：
- `catalytic_motif.intact = true` → 生物学底线（催化残基错一个蛋白即失活），但论文未用此项筛选
- `identity_vs_wt.mean` 在 50-70% → 序列新颖性合理
- `pairwise_identity.mean` 不太高 → 生成样本覆盖广

---

## 五、Phase 2：训练效率指标

解析 `run_pt.py` 输出的训练 CSV，对比两次运行的效率差异。

```bash
# 单次运行评估
python protein_DIFF/eval/metrics_efficiency.py \
    --training_csv result/Ago/run_metrics.csv \
    --output result/metrics_efficiency.json

# 改进前后对比
python protein_DIFF/eval/metrics_efficiency.py \
    --training_csv result/Ago_original/run_metrics.csv \
    --compare_csv result/Ago_improved/run_metrics.csv \
    --output result/metrics_efficiency.json
```

**输出**：

```json
{
  "training": {
    "train_loss": {"final": 0.32, "best": 0.31},
    "recovery": {"final": 0.73, "best": 0.75},
    "convergence": {"epoch_to_90pct_final": 12}
  },
  "comparison_delta": {
    "recovery": {"final_delta": 0.03, "final_delta_pct": 4.1}
  }
}
```

---

## 六、Phase 4：结构预测 + pLDDT 三级筛选

### 4a. 结构预测

```bash
# ESMFold（快速，推荐初筛用）
python protein_DIFF/eval/predict_structures.py predict \
    --fasta_dir result/predict/fasta/ \
    --output_dir result/structures/ \
    --engine esmfold \
    --chunk_size 64          # 长序列 OOM 时调小

# AlphaFold2（论文级精度）
python protein_DIFF/eval/predict_structures.py predict \
    --fasta_dir result/predict/fasta/ \
    --output_dir result/structures/ \
    --engine alphafold
```

### 4b. pLDDT 三级筛选

```bash
python protein_DIFF/eval/predict_structures.py filter \
    --pdb_dir result/structures/ \
    --wt_pdb result/wt_structure/WT.pdb \
    --output result/metrics_plddt.json
```

**输出**：

```json
{
  "num_total": 100,
  "thresholds": {
    "overall_plddt_min": 82.5,
    "sigma_delta_max": 3.2,
    "large_diffs_max": 93
  },
  "filter_results": {
    "step_1_overall_plddt": {"passed": 85, "rejected": 15},
    "step_2_sigma_delta": {"passed": 80, "rejected": 5},
    "step_3_large_diffs": {"passed": 72, "rejected": 8},
    "final_pass_rate": 0.72
  }
}
```

**关键解读**：
- `final_pass_rate` 三步初筛通过率约 70%（论文中 ~70/100 通过，淘汰 ~30 条），最终精选率 27%（27/100 进入湿实验）
- 阈值是**动态的**（基于当前批次的均值 ± 1σ），不是固定值
- **注意**：论文实际三步 pLDDT 初筛淘汰 ~30/100（通过率 ~70%），最终经过残基级精细对比精选 27 条。此 JSON 示例为独立运行结果（final_pass_rate=0.72），不反映论文数据。

---

## 七、Phase 3：结构质量指标

结构预测完成后，评估 TM-score、RMSD、CA-CA 几何。

```bash
# 全量评估（所有预测结构）
python protein_DIFF/eval/metrics_structure.py \
    --pdb_dir result/structures/ \
    --wt_pdb result/wt_structure/WT.pdb \
    --output result/metrics_structure.json

# 仅评估 pLDDT 通过的候选（论文口径）
python protein_DIFF/eval/metrics_structure.py \
    --pdb_dir result/structures/ \
    --wt_pdb result/wt_structure/WT.pdb \
    --plddt_json result/metrics_plddt.json \
    --output result/metrics_structure_passed.json
```

**输出**：

```json
{
  "tm_score_vs_wt": {
    "mean": 0.93, "std": 0.04,
    "above_0_9_ratio": 0.88
  },
  "rmsd_vs_wt_A": {
    "mean": 1.85, "std": 0.52,
    "below_3A_ratio": 0.95
  },
  "ca_ca_geometry": {
    "deviation_mean_A": 0.08,
    "total_clashes": 12
  }
}
```

---

## 八、Phase 5：保守性 + 系统发育

```bash
python protein_DIFF/eval/metrics_phylogeny.py \
    --csv result/predict/predict.csv \
    --wt_fasta dataset/Ago/wt_kmago.fasta \
    --output_dir result/phylogeny/
```

**输出文件**：

```
result/phylogeny/
├── combined.fasta          ← WT + 生成序列合并
├── aligned.fasta           ← MUSCLE 比对结果
├── phylo_tree.treefile     ← IQ-TREE 最大似然树 (NEWICK)
├── phylo_tree.iqtree       ← IQ-TREE 文本报告
└── metrics_phylogeny.json  ← 保守性评分报告
```

**保守性评分解读**：
- `high_conserved_count`: 论文中 pAgo 家族有 33 个 R_seq > 2.5 的位点
- 生成序列在这些位点上的氨基酸分布应与 WT-pAgo 一致

---

## 九、Phase 6：新颖性分析

### 6a. BLAST vs NCBI NR（序列层面）

```bash
python protein_DIFF/eval/metrics_novelty.py blast \
    --csv result/predict/predict.csv \
    --max_seqs 10 \               # 建议测试 10 条，全量 100 条需要 30-60 分钟
    --blast_email you@email.com \ # NCBI 要求
    --output result/novelty_blast.json
```

### 6b. FoldSeek vs PDB（结构层面）

```bash
# 需要先建 PDB 数据库（只做一次）
foldseek databases PDB pdb tmp/

# 运行结构新颖性搜索
python protein_DIFF/eval/metrics_novelty.py foldseek \
    --pdb_list result/All_Sampled_PDB.txt \
    --designable_list result/All_Sampled_PDB_Designable.txt \
    --script_path protein_DIFF/eval/run_foldseek_parallel.sh \
    --dataset_dir /path/to/FoldSeek_PDB_Database
```

---

## 十、一键运行（full_pipeline.py）

```bash
# 完整流程（从序列到全部评估）
python protein_DIFF/eval/full_pipeline.py \
    --csv result/predict/predict.csv \
    --wt_fasta dataset/Ago/wt_kmago.fasta \
    --output_dir result/eval_full/

# 跳过结构预测（如果已经跑过）
python protein_DIFF/eval/full_pipeline.py \
    --csv result/predict/predict.csv \
    --wt_fasta dataset/Ago/wt_kmago.fasta \
    --output_dir result/eval_full/ \
    --skip_predict \
    --pdb_dir result/structures/ \
    --wt_pdb result/wt_structure/WT.pdb

# 包含效率对比
python protein_DIFF/eval/full_pipeline.py \
    --csv result/predict/predict.csv \
    --wt_fasta dataset/Ago/wt_kmago.fasta \
    --output_dir result/eval_full/ \
    --training_csv result/Ago/run_metrics.csv \
    --compare_csv result/Ago_improved/run_metrics.csv

# 跳过慢速步骤
python protein_DIFF/eval/full_pipeline.py \
    ... \
    --skip_predict --skip_blast --skip_foldseek
```

---

## 十一、各指标的判读标准

| 指标 | 论文阈值 | 论文 KmAgo 值 | 改进论证标准 |
|------|----------|:--:|------|
| 催化基序保留 (KmAgo) | 代码自设 100%（生物学底线）| — | 论文未列入筛选，仅 Fig.5c 事后检验 |
| 催化基序保留 (PfAgo) | 代码自设 100%（生物学底线）| — | 论文未列入筛选；需 --fix_pos_file |
| 恢复率 | — | 64.62%（Suppl. Table S4） | ≥ 原方法 |
| 序列一致性 vs WT | 50-70% | 50-70% | 不超出范围 |
| AA 属性保留 | — | — | 电荷翻转率低 |
| 整体 pLDDT | ≥ μ−1σ | ~85 | 不显著低于原方法 |
| σ(ΔpLDDT) | ≤ μ+1σ | — | 不显著高于原方法 |
| RMSD (KmAgo) | < 3.0Å | < 3.0Å（Suppl: ≤2.5Å）| ≤ 原方法 + 0.5Å |
| RMSD (PfAgo) | < 1.0Å | < 1.0Å | ≤ 原方法 + 0.5Å |
| TM-score (WT 归一) | > 0.9 | > 0.9 | ≥ 原方法 − 0.05 |
| TM-score (PfAgo, WT 归一) | > 0.97（主论文）/ > 0.9（Suppl）| > 0.97 | ≥ 原方法 − 0.05 |
| 三步初筛通过率 | ~70% | ~70/100 | ≥ 原方法 × 0.7 |
| 最终精选率 | 27% | 27/100 | —（论文为残基级人工审图，代码未实现此步骤）|
| NCBI NR 一致性 (排除模板) | < 40% | 30-40% | < 40% |
| Spearman r (突变效应) | — | — | ≥ 原方法 |
| R_seq 保守位点* | ~33 个 | — | 需 694 条天然 pAgo 参考集 |
| 系统发育树* | long-A 分支 | — | 需 694 条天然 pAgo 参考集 |
| 困惑度 | 收敛后低 | — | 不高于原方法 |

> *标注项需额外提供 694 条天然 pAgo FASTA (`--ref_fasta`)，尚未在代码中实现。

---

## 十二、改进论证速查

| 你想证明什么 | 运行 | 看什么 |
|-------------|------|--------|
| 模型逆折叠能力更强 | Phase 2 (compare) | `recovery.final_delta` > 0 |
| 收敛更快 | Phase 2 | `convergence.epoch_to_90pct_final` 更小 |
| 生成序列质量不变 | Phase 1 + 3 + 4 | 催化基序 100%, RMSD/TM 无显著差异 |
| 生成序列更独特 | Phase 1 + 6 | 成对一致性更低, NCBI NR 一致性更低 |
| 结构预测更可信 | Phase 4 | pLDDT 均值更高, σ(ΔpLDDT) 更低 |
