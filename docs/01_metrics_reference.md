# CPDiffusion / CPFlow 干实验评估完整指南

> 论文：*"A conditional protein diffusion model generates artificial programmable endonuclease sequences with enhanced activity"* (Cell Discovery, 2024)
> 代码仓库：CPFlow（CPDiffusion meanflow 版）
> **范围：仅干实验（In Silico），不含湿实验**

---

## 一、概述

### 1.1 论文目标与干实验定位

CPDiffusion 的目标是：**单步生成**与野生型序列差异大（<70% 一致性）、但结构一致、功能关键残基保留的多结构域复杂蛋白新序列。

干实验在整个流程中承担两个职责：
- **筛选**：从大量生成序列中选出"结构可信 + 序列合理"的候选
- **验证**：在进入后续实验前，用计算手段确认序列质量

### 1.2 仓库输出格式

`inference.py` 生成 **FASTA 氨基酸序列**，每条序列一个文件：

```
result/predict/
├── fasta/Ago_0_0.71.fasta    # >0|0.71|88935
├── fasta/Ago_1_0.72.fasta
└── predict.csv               # id, seq, recovery 三列
```

FASTA 头部格式：`>id|恢复率|训练步数`

---

## 二、指标全览

> 每个指标同时标出：论文意义、计算公式（如适用）、可用于何种论证、仓库中的实现状态。

### 图例

| 标记 | 含义 |
|:--:|------|
| 🔵 | 仓库代码已存在，直接可用 |
| 🟡 | 纯 Python 可实现，仓库无代码，需编写 |
| 🟢 | 外部工具可集成（pip/conda + subprocess） |
| 🔴 | 需特殊软件，难以自动化 |

### 2.1 模型训练与生成质量指标

| # | 指标 | 定义 | 论文用途 | 实现 | 仓库状态 |
|:--:|------|------|----------|------|:--:|
| 1 | **恢复率** (Recovery rate) | 生成序列与模板WT逐位氨基酸一致的比例 | 逆折叠能力核心指标；模型质量底线 | `run_pt.py:seq_recovery()` / `inference.py` 自动输出 | 🔵 |
| 2 | **困惑度** (Perplexity) | `exp(交叉熵)`，值越低越好 | 模型对天然序列分布的拟合度 | `run_pt.py:Trainer.train()` 自动记录 | 🔵 |
| 3 | **训练/验证损失** | 交叉熵损失，预测分布 vs 真实标签 | 训练优化监控 | `run_pt.py` 自动记录到 CSV + PNG | 🔵 |
| 4 | **序列多样性** | 生成序列间的差异程度（成对一致性分布） | 避免模型只生成"模板复制品" | 纯 Python ~30 行 | 🟡 |
| 5 | **CATH/TS50/T500 恢复率** | 标准逆折叠基准上的恢复率 | 基础模型泛化性 | CATH 仓库已集成；TS50/T500 需下载 | 🟢 |

### 2.2 结构评估指标（AlphaFold2 / ESMFold 预测）

> 论文中**最核心的筛选体系**，通过预测 3D 结构并与 WT 模板对比来筛选。

| # | 指标 | 定义 | 论文阈值 | 仓库状态 |
|:--:|------|------|----------|:--:|
| 6 | **整体 pLDDT** | AlphaFold2 每残基置信度的全局平均（0-100） | ≥ 均值 − 1σ | 🟢 需 ESMFold / AlphaFold2 |
| 7 | **σ(ΔpLDDT)** | `std(pLDDT_AP − pLDDT_WT)`，局部结构差异波动性 | ≤ 均值 + 1σ | 🟡 numpy 计算，依赖 #6 |
| 8 | **count(|ΔpLDDT| > 10)** | 单残基 pLDDT 差异 > 10 的位点数量 | ≤ 均值 + 1σ | 🟢 predict_structures.py 已实现 |
| 9 | **RMSD** | `√(1/N · Σ‖r_AP − r_WT‖²)`，CA 原子坐标均方根偏差 | < 3.0Å (KmAgo 主论文) / ≤ 2.5Å (KmAgo 补充材料) / < 1.0Å (PfAgo) | 🟢 tmtools (Python 库) |
| 10 | **TM-score** | 0-1，全局折叠相似性（>0.5 同折叠，>0.9 几乎相同） | > 0.9 (KmAgo) / > 0.97 (PfAgo，Suppl: > 0.9) | 🟢 tmtools (Python 库)（与 #9 同一次调用） |

**筛选逻辑链**：整体 pLDDT → σ(ΔpLDDT) → count(|ΔpLDDT| > 10) → 残基级精细对比 → RMSD + TM-score 确认。三步 pLDDT 初筛约 70% 通过（淘汰 ~30/100），残基级精细对比后精选 27 条（27%）。论文三步初筛淘汰 ~30 条，Fig. 2b 残基级审图后再精选至 27 条。

注：补充材料 (Suppl. Section 6.1) 指出 RMSD 也用于预筛，而非仅筛后确认；补充材料 RMSD 阈值 ≤ 2.5Å（主论文 < 3.0Å）。

### 2.3 序列分析指标

| # | 指标 | 定义 | 论文用途/阈值 | 仓库状态 |
|:--:|------|------|--------------|:--:|
| 11 | **序列一致性 vs WT** | 与模板野生型序列的氨基酸一致比例 | 50-70%——够新但不能太远 | 🟡 纯 Python ~10行 |
| 12 | **序列一致性 vs NCBI NR** | 与 NCBI 非冗余数据库中所有天然蛋白的最高相似度 | < 40%——不是已有蛋白变体 | 🟢 BLAST+ (conda) |
| 13 | **催化基序保留** | PIWI 结构域催化四联体 DEDD/DEDH 四个残基是否全部正确 | 代码自设底线（论文仅事后检验 27 条精选序列，非筛选步骤）| 🟡 纯 Python ~15行 |
| 14 | **保守性评分 R_seq** | `log₂20 + Σ p_n·log₂(p_n)`，基于 Shannon 熵 | 与 pAgo 家族天然分布一致 | 🟢 MUSCLE + Python |
| 15 | **系统发育树归属** | 与 694 个 WT pAgo 蛋白的进化关系 | 位于正确 long-A 进化分支 | 🟢 MUSCLE + IQ-TREE |
| 16 | **t-SNE 嵌入分布** | 高维序列特征 2D 投影 | 从 WT 向 pAgo 家族景观扩散（不坍缩） | 🟡 sklearn |
| 17 | **筛选通过率** | 通过 pLDDT + RMSD + TM 三级筛选的比例 | ≥ 原方法的 70%（改进后对比用） | 🟡 依赖 #6-10 |

### 2.4 高级分析指标（可选）

| # | 指标 | 工具 | 说明 | 仓库状态 |
|:--:|------|------|------|:--:|
| 18 | **静电表面** | Chimera / ChimeraX `coulombic` | 切割位点负电 + 底物结合区正电 | 🔴 GUI 软件 |
| 19 | **MD 结合自由能** | GROMACS + CHARMM36m + AlphaFold3 | 催化剂-底物复合物稳定性 | 🔴 专业软件 |
| 20 | **Spearman 相关性** (突变效应) | `run_pt.py` 内置 | 预测突变得分 vs ProteinGym 实验数据 | 🔵 |

### 2.5 效率对比指标（方法改进论证用）

| # | 指标 | 测量方式 | 仓库状态 |
|:--:|------|----------|:--:|
| 21 | 训练损失收敛速度 | 相同步数的 loss 值 / 达到相同 loss 的步数 | 🔵 `run_pt.py` 自动 |
| 22 | 恢复率 vs 步数曲线 | 更快达到更高恢复率 | 🔵 `run_pt.py` 自动 |
| 23 | 每 epoch 训练时间 | `time.time()` | 🟡 |
| 24 | GPU 显存占用 | `torch.cuda.max_memory_allocated()` | 🟡 |
| 25 | 可学习参数量 | `sum(p.numel())` | 🟡 |
| 26 | 单条序列生成时间 | 计时 `sample()` 调用 | 🟡 |
| 27 | 扩散步数 vs 质量 | `sampling_timesteps` 5→100 的恢复率变化 | 🟡 |

---

## 三、评估流程

### 3.1 一站式 Pipeline

```
0. 生成序列
   └── python protein_DIFF/inference.py ...
       输出: FASTA + predict.csv (含恢复率)

1. 结构预测
   └── ESMFold (pip install esm) 或 AlphaFold2
       输出: 每条序列的 PDB + 逐残基 pLDDT

2. 三级筛选
   ├── 整体 pLDDT 过滤 (去掉 < μ−σ 的)
   ├── σ(ΔpLDDT) 过滤 (去掉 > μ+σ 的)
   ├── count(|ΔpLDDT| > 10) 过滤 (去掉 > μ+σ 的)
   └── 残基级精细对比（⚠ 代码未实现 — 论文为人工审图 Fig. 2b,
       将 ~70 条精选至 27 条，无可自动化阈值）

3. 结构确认
   └── tmtools: RMSD + TM-score
       标准: RMSD < 3.0Å, TM-score > 0.9

4. 序列分析
   ├── 序列一致性 (vs WT, vs NCBI NR, 成对)
   ├── 催化基序保留 (DEDD/DEDH, 必须全部正确)
   ├── 保守性评分 (MUSCLE + R_seq)
   └── 系统发育树 (IQ-TREE)

5. (可选) 高级分析
   ├── t-SNE 嵌入可视化
   ├── 静电表面 (Chimera 手动)
   ├── MD 模拟 (GROMACS, 与专业人士合作)
   └── 突变效应评估 (run_pt.py 内置, 需 ProteinGym)
```

### 3.2 快速启动命令

```bash
# Step 0: 生成 100 条序列
python protein_DIFF/inference.py \
    --ckpt ckpt/Jun_5_ago_dataset=CATH_result_lr=0.0005_wd=0.0_dp=0.08_hidden=256_noisy_type=uniform_embed_ss=False_88935.pt \
    --target_protein dataset/Ago/process/AGO_050_model_3_ptm.pt \
    --target_protein_dir dataset/Ago/process/ \
    --gen_num 100 --output_dir result/predict

# Step 1: 结构预测 (ESMFold 批量)
# 需自行编写批量脚本，参照 dataset/predict_structure.py

# Step 2-4: 筛选 + 分析 (需自行编写，逻辑见第四节代码片段)

# Step 3: RMSD/TM-score
# RMSD/TM-score 通过 tmtools Python 库计算（metrics_structure.py），无需 US-align。

# Step 4: 系统发育树
conda install -c bioconda muscle iqtree
muscle -align combined.fasta -output aligned.fasta
iqtree -s aligned.fasta -m BLOSUM62 -B 1500
```

---

## 四、仓库代码对照与集成方案

### 4.1 立即可用（🔵 = 仓库已实现）

| 指标 | 代码位置 | 使用方式 |
|------|----------|----------|
| 恢复率 | `run_pt.py:seq_recovery()` (L375) | 训练时自动计算；`inference.py` 输出到 FASTA 头部 |
| 困惑度 | `run_pt.py:Trainer.train()` (L530) | 自动输出到 CSV + PNG |
| 训练曲线（loss/恢复率/困惑度） | `run_pt.py:Trainer.train()` | `result/{protein}/figure/*.png` + CSV |
| 消融实验 | 复用 `run_pt.py` | 不带 `--target_protein_dir` 参数即可 |
| Spearman 相关性 | `run_pt.py:compute_single_site_corr_score_all()` | 需 ProteinGym 数据集 |
| 突变效应评估 | `dataset/pdbbind_eval.py` | 单点/多点突变，需 ProteinGym |

### 4.2 需自行编写（🟡 = 纯 Python，无外部依赖）

```python
# ---- 催化基序检查 (~15 行) ----
def check_catalytic_motif(seq, positions, expected):
    """positions: [526,561,595,712] (0-indexed KmAgo DEDD)"""
    return all(seq[p] == e for p, e in zip(positions, expected) if p < len(seq))

# ---- 序列一致性 (~10 行) ----
def seq_identity(s1, s2):
    return sum(a == b for a, b in zip(s1, s2)) / len(s1)

# ---- σ(ΔpLDDT) + count(|ΔpLDDT| > 10) (依赖 #6 pLDDT 数组) ----
import numpy as np
delta = ap_plddt[:len(wt)] - wt_plddt[:len(ap)]
sigma_delta = np.std(delta)
large_diffs = int(np.sum(np.abs(delta) > 10))：仅计 AP pLDDT 超过 WT 的位点

# ---- 效率指标 ----
import time, torch
t_start = time.time(); trainer.train(); elapsed = time.time() - t_start
gpu_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
n_params = sum(p.numel() for p in model.parameters())
t_infer = time.time(); model.sample(data); t_infer = time.time() - t_infer
```

### 4.3 可集成的外部工具（🟢 = pip/conda 一条命令）

| 工具 | 安装 | 覆盖指标 | 集成方式 |
|------|------|----------|----------|
| **ESMFold** | `pip install esm` | pLDDT (#6) | `import esm` Python API |
| **tmtools** | `pip install tmtools` | RMSD (#9), TM-score (#10) | `from tmtools import tm_align` |
| **BLAST+** | `conda install -c bioconda blast` | NCBI NR 一致性 (#12) | `subprocess.run(['blastp',...])` |
| **MUSCLE v5** | `conda install -c bioconda muscle` | 保守性评分 (#14) | `subprocess.run(['muscle',...])` |
| **IQ-TREE** | `conda install -c bioconda iqtree` | 系统发育树 (#15) | `subprocess.run(['iqtree',...])` |
| **scikit-learn** | `pip install scikit-learn` | t-SNE (#16) | `from sklearn.manifold import TSNE` |

tmtools 集成示例（替代 US-align）：

```python
from tmtools import tm_align

def compute_rmsd_tm(coords1, coords2, seq1, seq2):
    """coords: (N, 3) CA coordinates; seq: string. 使用 tmtools 替代 US-align."""
    result = tm_align(coords1, coords2, seq1, seq2)
    rmsd = result.rmsd
    tm = max(result.tm_norm_chain1, result.tm_norm_chain2)
    return rmsd, tm
```

### 4.4 暂不集成的特殊软件（🔴）

| 指标 | 工具 | 原因 |
|------|------|------|
| 静电表面 | Chimera / ChimeraX | GUI 软件，建议手动打开 PDB 执行 `coulombic` |
| MD 模拟 | GROMACS + CHARMM36m | 专业计算、耗时长、论文仅用于解释而非筛选 |
| vs ProteinMPNN | ProteinMPNN 独立仓库 | 需单独搭环境 + 预训练权重，建议手动对比一次 |

---

## 五、方法改进论证

> 当对 CPDiffusion 进行改进（网络结构、训练策略、噪声调度等），需从两个维度论证。

### 5.1 两个维度

| 维度 | 回答的问题 | 核心指标 |
|------|-----------|----------|
| **A: 证明改进了** | 训练/推理是否更有效？ | #21-27 效率指标 + #1 恢复率提升 |
| **B: 保证不退化** | 生成序列质量是否持平原方法？ | #1 恢复率、#13 催化基序、#6-10 结构指标、#11 序列一致性 |

### 5.2 质量保证底线（维度 B 不可妥协的）

| 指标 | 最低标准 |
|------|----------|
| 恢复率 | ≥ 原方法 |
| 催化基序保留 | 代码自设 100%（论文未列入筛选，仅 Fig.5c 事后检验）|
| 整体 pLDDT | 均值不显著低于原方法 |
| RMSD 均值 | ≤ 原方法 ± 0.5Å |
| TM-score 均值 | ≥ 原方法 − 0.05 |
| 序列一致性 vs WT | 保持 50-70% 范围 |
| 筛选通过率 | ≥ 原方法的 70% |
| t-SNE 嵌入分布 | 不坍缩（与原方法生成序列在 2D 空间中有重叠） |

### 5.3 实验矩阵

| 编号 | 实验 | 对比 | 期望 | 依赖指标 |
|:--:|------|------|------|:--:|
| E1 | 训练曲线 | O vs N 的 loss/恢复率/困惑度 | N 收敛更快或最终更好 | #21 #22 #1 #2 |
| E2 | 最终恢复率 | O vs N, ≥3 个随机种子, ±std | N ≥ O | #1 |
| E3 | 催化基序 | O vs N, 100 条序列 | **100%** | #13 |
| E4 | pLDDT 三级筛选 | O vs N, 100 条序列 | N 通过率 ≥ O × 0.7 | #6 #7 #8 |
| E5 | RMSD/TM 分布 | O vs N, 筛选后序列 | N ≈ O（无显著差异） | #9 #10 |
| E6 | 序列多样性 | O vs N 的成对一致性分布 | 不坍缩 | #4 #11 |
| E7 | 消融实验 | N 去掉改进 vs N 完整 | 改进组件有正向贡献 | #1 #13 #4 |
| E8 | vs ProteinMPNN | O vs N vs ProteinMPNN | N > O > MPNN | #1 #13 |
| E9 | 推理速度 | O vs N, 100 条序列 | N 更快（如适用） | #26 |
| E10 | 参数/显存 | O vs N | N 更省（如适用） | #24 #25 |

> O = Original (原 CPDiffusion), N = New (改进后 CPFlow)

### 5.4 优先级

| 优先级 | 必须做 | 建议做 | 加分项 |
|:--:|--------|--------|--------|
| **P0** | E2 (恢复率), E3 (催化基序), E7 (消融) | — | — |
| **P1** | — | E1 (训练曲线), E4 (pLDDT), E5 (RMSD/TM) | E6 (多样性) |
| **P2** | — | — | E8 (vs MPNN), E9-10 (效率) |

### 5.5 最少可行验证 (MVV)

```
1. 恢复率 ≥ 原方法 (逆折叠能力)
2. 催化基序保留 (生物学底线：催化残基错任意一个蛋白即失活，代码设 100%)
3. 消融实验 (改进组件有独立贡献)
```

三项通过 → 方法层面有效 + 核心能力无损 → 可进入全面的结构评估。

---

## 六、评估脚本目录结构

```
protein_DIFF/eval/                         ← ✅ Phase 1-6 + Gap Fill 全部实现
├── __init__.py                            # 包初始化 + 论文指标对应
├── full_pipeline.py                       # 一键运行
├── run_foldseek_parallel.sh               # FoldSeek 并行 (from ReQFlow)
├── metrics_sequence.py                    # 🔵🟡 催化基序, 一致性, 多样性, t-SNE
├── metrics_aa_properties.py               # 🟡 AA 属性保留 (Supp Fig. S4)
├── metrics_efficiency.py                  # 🔵🟡 参数, 时间, 显存
├── metrics_structure.py                   # 🟢 TM-score/RMSD/CA-CA/SS (tmtools>=0.3.0)
├── predict_structures.py                  # 🟢 ESMFold/AF2(MSA+模板+5模型) + pLDDT 三级筛选
├── metrics_phylogeny.py                   # 🟢 保守性 + 系统发育树 (MUSCLE + IQ-TREE)
├── metrics_novelty.py                     # 🟢 BLAST NCBI NR + FoldSeek PDB
└── metrics_spearman.py                    # 🟢 突变效应 Spearman (Supp Data §4)
```

---

## 七、附录：论文干实验指标分类速查

| 类别 | 包含指标 | 用途 |
|------|----------|------|
| 模型质量 | 恢复率、困惑度、交叉熵、序列多样性 | 训练与泛化性 |
| 结构筛选 | pLDDT, σ(ΔpLDDT), count(|ΔpLDDT| > 10), RMSD, TM-score | 三级筛选 + 确认 |
| 序列分析 | 序列一致性(vs WT, vs NR, 成对)、保守性评分、催化基序、WebLogo | 新颖性 + 功能保留 |
| 进化分析 | 系统发育树、t-SNE | 进化归属 + 空间探索 |
| 高级分析 | 静电表面、MD 模拟 | 机理解释（非必需） |
| 基线对比 | vs ProteinMPNN (恢复率、催化基序)、消融实验 | 方法优越性 |
