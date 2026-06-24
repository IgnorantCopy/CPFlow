# CPFlow 评估指标实现优先级计划

## 总体原则

1. **零依赖优先**：能纯 Python 解决的先做，不依赖外部安装
2. **每阶段可独立运行**：不互相阻塞
3. **从序列到结构**：先做序列层面指标，再做结构层面
4. **核心指标优先**：催化基序 > 恢复率 > 结构确认 > 其他

---

## Phase 1：序列层面核心指标（零外部依赖，立刻可写）

| 优先级 | 指标 | 文件 | 依赖 | 行数 |
|:--:|------|------|------|:--:|
| P0 | 催化基序保留 (DEDD/DEDH) | `metrics_sequence.py` | 无 | ~15 |
| P0 | 序列一致性 vs WT | `metrics_sequence.py` | 无 | ~15 |
| P0 | 成对序列一致性分布 | `metrics_sequence.py` | 无 | ~30 |
| P1 | 序列多样性统计 | `metrics_sequence.py` | 无 | ~20 |
| P1 | 保守位点 AA 分布对比 (JS 散度) | `metrics_sequence.py` | scipy | ~40 |
| P2 | t-SNE 嵌入可视化 | `metrics_sequence.py` | sklearn | ~15 |

**输出**：metrics_sequence.py，一个独立脚本，输入 FASTA 目录 → 输出 JSON/CSV 报告。

## Phase 2：训练效率指标（零外部依赖）

| 优先级 | 指标 | 文件 | 依赖 | 行数 |
|:--:|------|------|------|:--:|
| P0 | 可学习参数量 | `metrics_efficiency.py` | torch | ~5 |
| P1 | GPU 显存占用 | `metrics_efficiency.py` | torch | ~5 |
| P1 | 推理时间 (单条序列) | `metrics_efficiency.py` | time | ~15 |
| P2 | 扩散步数 vs 恢复率 | `metrics_efficiency.py` | time + run_pt | ~40 |

**输出**：metrics_efficiency.py，用于改进前后的效率对比。

## Phase 3：结构层面指标（引入 tmtools，替代 US-align）

| 优先级 | 指标 | 文件 | 依赖 | 行数 |
|:--:|------|------|------|:--:|
| P0 | TM-score (生成 vs WT) | `metrics_structure.py` | tmtools | ~20 |
| P0 | RMSD (生成 vs WT) | `metrics_structure.py` | tmtools | ~20 |
| P1 | 成对 TM-score 多样性 | `metrics_structure.py` | tmtools | ~30 |
| P1 | CA-CA 键几何验证 | `metrics_structure.py` | numpy | ~25 |
| P2 | 二级结构组成 | `metrics_structure.py` | mdtraj | ~20 |

**输出**：metrics_structure.py，从 ReQFlow metrics.py 移植核心函数。

## Phase 4：pLDDT 三级筛选（引入 ESMFold）

| 优先级 | 指标 | 文件 | 依赖 | 行数 |
|:--:|------|------|------|:--:|
| P0 | 批量 ESMFold 结构预测 | `predict_structures.py` | esm, biotite | ~80 |
| P0 | 逐残基 pLDDT 提取 | `predict_structures.py` | biotite | ~30 |
| P0 | 三级筛选 (整体/σ/count) | `metrics_plddt.py` | numpy | ~60 |
| P1 | 筛选通过率统计 | `metrics_plddt.py` | numpy | ~20 |

**输出**：predict_structures.py + metrics_plddt.py

## Phase 5：进化分析（引入 MUSCLE + IQ-TREE）

| 优先级 | 指标 | 文件 | 依赖 | 行数 |
|:--:|------|------|------|:--:|
| P1 | 多序列比对 (MUSCLE) | `metrics_phylogeny.py` | muscle | ~30 |
| P1 | 保守性评分 R_seq | `metrics_phylogeny.py` | numpy | ~40 |
| P2 | 系统发育树 (IQ-TREE) | `metrics_phylogeny.py` | iqtree | ~40 |

**输出**：metrics_phylogeny.py

## Phase 6：新颖性分析（大工具，可选）

| 优先级 | 指标 | 文件 | 依赖 |
|:--:|------|------|------|
| P2 | BLAST vs NCBI NR | `metrics_blast.py` | blast+ |
| P2 | FoldSeek vs PDB | `metrics_foldseek.py` | foldseek |

---

## 执行顺序

```
✅ Phase 1  (metrics_sequence.py)          — 已完成
✅ Phase 1b (metrics_aa_properties.py)     — 已完成 (AA 属性保留)
✅ Phase 2  (metrics_efficiency.py)        — 已完成
✅ Phase 3  (metrics_structure.py)          — 已完成 (tmtools>=0.3.0)
✅ Phase 4  (predict_structures.py)         — 已完成 (ESMFold + AlphaFold2 + pLDDT)
✅ Phase 5  (metrics_phylogeny.py)          — 已完成 (MUSCLE + IQ-TREE + R_seq)
✅ Phase 6  (metrics_novelty.py)            — 已完成 (BLAST + FoldSeek)
✅ Gap fill (metrics_spearman.py)           — 已完成 (突变效应 Spearman)
  │
  ▼
⬜ B1/B2    --ref_fasta + compare_conserved_positions()
             — 需 694 条天然 pAgo 外部数据，待实现
```

> A1/D1/E1/E2/B3/C1 已修复；B4/C2 接受；B1/B2 依赖外部 694 条天然 pAgo 数据；⚠ 论文残基级精细对比精选步骤未实现（人工审图，无法自动化）。

## 已实现文件

```
protein_DIFF/eval/                          (11 files, ~3100 lines)
├── __init__.py                  # 包初始化 + 论文对应关系
├── full_pipeline.py             # 一键运行全部评估
├── run_foldseek_parallel.sh     # FoldSeek 并行 (from ReQFlow)
├── metrics_sequence.py          # Phase 1: 催化基序 + 序列一致性 + 多样性 + t-SNE
├── metrics_aa_properties.py     # Phase 1b: AA 属性保留 (Supp Fig. S4)
├── metrics_efficiency.py        # Phase 2: 训练/推理效率
├── metrics_structure.py         # Phase 3: TM-score/RMSD/CA-CA/SS
├── predict_structures.py        # Phase 4: ESMFold/AF2 + pLDDT 三级筛选
├── metrics_phylogeny.py         # Phase 5: MUSCLE + IQ-TREE + R_seq
├── metrics_novelty.py           # Phase 6: BLAST + FoldSeek
└── metrics_spearman.py          # Gap fill: 突变效应 Spearman (Supp Data §4)
```

> 审计通过项：A1/D1/E1/E2/B3/C1 已修复；B4/C2 接受；D2/E3 非阻塞。
> 仅剩 B1/B2 (`--ref_fasta`) 待外部数据。

## 已知代码-论文差距（非阻断）

- **metrics_structure.py**：不区分 KmAgo/PfAgo 阈值。PfAgo 需 TM > 0.97 (主论文) 或 > 0.9 (Suppl)、RMSD < 1.0Å，代码统一用 TM > 0.9 / RMSD < 3.0Å。
- **metrics_phylogeny.py**：缺少 R_seq < 0.2 排除（论文 Methods 排除设计中的低保守性比对噪声位点）。
- **恢复率数值**：文档判读表写 71-76%，补充材料 Table S4 为 64.62%（CPDiffusion all pAgos KmAgo）。
