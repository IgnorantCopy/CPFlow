# ReQFlow 评估代码分析与 CPFlow 复用方案

> ReQFlow repo: `https://github.com/AngxiaoYue/ReQFlow/tree/master/analysis`
> 对照文档: `docs/01_metrics_reference.md`

---

## 一、ReQFlow 评估体系概览

### 1.1 三大核心指标

```
┌─────────────────────────────────────────────────────────────┐
│                  ReQFlow 评估三支柱                          │
│                                                             │
│  ① Designability (可设计性)                                  │
│     → ESMFold 自洽性: 结构→序列→结构, 比较 TM-score 和 RMSD  │
│     标准: min_rmsd < 2Å → "可设计的"                         │
│                                                             │
│  ② Diversity (多样性)                                       │
│     → 同长度生成结构之间的成对 TM-score 均值                  │
│     TM-score 越低 → 多样性越高                               │
│                                                             │
│  ③ Novelty (新颖性)                                         │
│     → FoldSeek 在 PDB 数据库中搜索相似结构                   │
│     标准: max TM-score < 0.5 → "新颖的"                     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 辅助指标

- 二级结构组成 (helix/strand/coil 比例)
- 回转半径 (Radius of gyration)
- CA-CA 键几何 (偏差率、有效性、碰撞)
- 推理时间统计

### 1.3 代码结构

```
analysis/
├── all_metric_calculation.py   # 主入口 (~700 行)，编排全流程
├── metrics.py                  # 纯函数：TM-score, RMSD, 二级结构, CA-CA 几何
├── utils.py                    # PDB 写入工具
├── run_foldseek_parallel.sh    # GNU parallel 并行 FoldSeek 搜索
└── README.md
```

**设计优点**：
- 关注点分离清晰：`metrics.py` 纯函数 → `all_metric_calculation.py` 编排
- Pipeline 模式：`clean → file_generate → designability → diversity → novelty → report`
- 并行化：FoldSeek 用 GNU parallel 多核并行
- 结构化输出：`Metrics.txt` 汇总 + 多个 CSV 中间文件

---

## 二、逐文件分析：CPFlow 可复用性

### 2.1 `metrics.py` — ⭐⭐⭐ 高度可复用

| 函数 | 功能 | CPFlow 复用 | 说明 |
|------|------|:--:|------|
| `calc_tm_score()` | 用 `tmtools` 库计算 TM-score | ✅ 直接复用 | 替代 US-align，Python 原生，无需 subprocess |
| `calc_mdtraj_metrics()` | 二级结构比例 + 回转半径 | ✅ 直接复用 | 需要 `pip install mdtraj` |
| `calc_ca_ca_metrics()` | CA-CA 键几何验证 | ✅ 直接复用 | 检测不合理键长和原子碰撞 |
| `calc_aligned_rmsd()` | 对齐后的 RMSD | ✅ 直接复用 | 替代 US-align 的 RMSD 计算 |

> **对 CPFlow 的改进**：可以不用 US-align，直接用 `tmtools` Python 库算 TM-score 和 RMSD，集成更简单。

### 2.2 `all_metric_calculation.py` — ⭐⭐ 结构可借鉴，逻辑需适配

| 模块 | 功能 | CPFlow 复用 | 说明 |
|------|------|:--:|------|
| `file_generate()` | 解析推理输出目录，收集 PDB 路径 | ❌ 需适配 | CPFlow 输出 FASTA，目录结构不同 |
| `designability_calculate()` | 自洽性评估统计 | ⚠️ 部分复用 | CPFlow 需要先 AlphaFold2 预测结构再做自洽性 |
| `diversity_calculate()` | 成对 TM-score 多样性 | ✅ 直接复用 | 生成结构的 pairwise TM |
| `run_foldseek()` | 调用 FoldSeek | ✅ 直接复用 | Bash 脚本接口匹配 |
| `foldseek_calculate()` | 解析 FoldSeek 结果 | ✅ 直接复用 | 读取 summary CSV |
| `calc_additional_metrics()` | 二级结构 + CA-CA | ✅ 直接复用 | 只需 PDB 文件即可 |
| `plot_time()` | 推理时间可视化 | ⚠️ 需适配 | 需 CPFlow 记录时间 |
| `clean_folder()` | 清除非必要文件 | ❌ 不需要 | CPFlow 输出结构不同 |

### 2.3 `run_foldseek_parallel.sh` — ⭐⭐⭐ 完全可复用

- 直接用 GNU parallel 对 PDB 列表并行运行 FoldSeek
- 接口：输入 PDB 列表 + designable 列表 + 输出目录
- CPFlow 只需准备好 PDB 列表文件即可

### 2.4 `utils.py` — ⭐ 依赖 ReQFlow 内部模块

- `write_prot_to_pdb()` 依赖 `data.protein` 和 `openfold`
- 功能简单（写 PDB），CPFlow 用 `biotite` 或 ESMFold 的 PDB 输出替代即可

---

## 三、CPFlow vs ReQFlow 评估适配对照

### 3.1 核心差异

| 维度 | ReQFlow | CPFlow |
|------|---------|--------|
| **生成内容** | 3D 骨架坐标 (N,CA,C) | 氨基酸序列 (FASTA) |
| **自洽性（可设计性）** | 结构→ESMFold序列→ESMFold结构→比较TM-score | 序列→AlphaFold2结构→比较 pLDDT/RMSD/TM |
| **多样性** | 生成结构之间的 pairwise TM-score | ✅ 相同，预测结构之间的 pairwise TM |
| **新颖性** | FoldSeek 搜索 PDB | ✅ 相同，预测结构→PDB 搜索 |

### 3.2 自洽性评估的适配

ReQFlow 的自洽性（`scTM`）路径：

```
生成的 3D 结构
    → ESMFold inverse folding → 氨基酸序列
    → ESMFold forward folding → 预测 3D 结构
    → TM-align 比较 → scTM = TM-score(原始结构, 预测结构)
    → 标准: scTM > 0.5, min_rmsd < 2Å
```

CPFlow 的自洽性（论文方法）路径：

```
生成的 FASTA 序列
    → AlphaFold2 预测 → 3D 结构
    → 与 WT 模板比较 → RMSD, TM-score, pLDDT
    → 三级筛选 → 标准: RMSD < 3Å, TM > 0.9
```

> CPFlow 不需要 ESMFold inverse folding 这一步，因为**它本来就是生成序列的模型**。直接用 AlphaFold2 预测结构→与 WT 比较即可。

### 3.3 直接可复用的部分

```
ReQFlow analysis/                     CPFlow 复用方式
─────────────────────                 ─────────────────
metrics.py
  ├── calc_tm_score()          →  ✅  替代 US-align, 用于:
  │                                   • 生成序列预测结构的 pairwise TM (多样性)
  │                                   • 预测结构 vs WT 模板的 TM (结构确认)
  │
  ├── calc_mdtraj_metrics()    →  ✅  预测结构的二级结构验证
  │                                   • helix/strand/coil 比例是否合理
  │                                   • 回转半径是否正常
  │
  └── calc_ca_ca_metrics()     →  ✅  预测结构的几何验证
                                      • 键长是否合理
                                      • 是否存在原子碰撞

run_foldseek_parallel.sh       →  ✅  完全复用
                                      • 只需将生成的 PDB 列表传入

all_metric_calculation.py
  └── diversity_calculate()    →  ✅  逻辑复用
  └── foldseek_calculate()     →  ✅  逻辑复用
  └── calc_additional_metrics()→  ✅  逻辑复用
```

---

## 四、建议的 CPFlow 评估集成方案 (已实现)

### 4.1 当前目录结构

```
protein_DIFF/eval/                          (11 files, ~3100 lines)
├── __init__.py                    # 包初始化
├── full_pipeline.py               # 一键运行
├── run_foldseek_parallel.sh       # FoldSeek 并行 (直接从 ReQFlow 复制)
├── metrics_sequence.py            # Phase 1: 催化基序, 序列一致性, 多样性, t-SNE
├── metrics_aa_properties.py       # Phase 1b: AA 属性保留
├── metrics_efficiency.py          # Phase 2: 参数, 时间, 显存
├── metrics_structure.py           # Phase 3: TM-score/RMSD/CA-CA/SS
├── predict_structures.py          # Phase 4: ESMFold/AF2(MSA+模板+5模型) + pLDDT
├── metrics_phylogeny.py           # Phase 5: MUSCLE + IQ-TREE + R_seq
├── metrics_novelty.py             # Phase 6: BLAST + FoldSeek
└── metrics_spearman.py            # Gap fill: 突变效应 Spearman
```

### 4.2 四个审计轮次修复的关键项

| 轮次 | 修复项 |
|:--:|------|
| 第1轮 | B1/B2/C2/E3/D2 不改理由说明 |
| 第2轮 | A1 函数恢复, D1 路径修正, B3/B4 配置, C1 PfAgo 位点, E1 WT 自检, E2 TM 归一 |
| 第3轮 | C1 撤回假位点, B3 注释修正 |
| 第4轮 | B3 `--templates` 真加, C1 CLI 提示实做, docstring 清理 |

### 4.2 可直接复制的代码

#### metrics.py（从 ReQFlow 直接移植，改 import 路径）

```python
# protein_DIFF/eval/metrics.py
# 移植自 ReQFlow analysis/metrics.py
# 改动: 去掉 from data import utils as du 依赖, 改为自包含

import numpy as np
from tmtools import tm_align

def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    """计算 TM-score, 返回 (tm_norm_chain1, tm_norm_chain2)"""
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2

def calc_mdtraj_metrics(pdb_path):
    """二级结构 + 回转半径"""
    import mdtraj as md
    traj = md.load(pdb_path)
    pdb_ss = md.compute_dssp(traj, simplified=True)
    return {
        'coil_percent': np.mean(pdb_ss == 'C'),
        'helix_percent': np.mean(pdb_ss == 'H'),
        'strand_percent': np.mean(pdb_ss == 'E'),
        'non_coil_percent': np.mean((pdb_ss == 'H') | (pdb_ss == 'E')),
        'radius_of_gyration': md.compute_rg(traj)[0],
    }

def calc_ca_ca_metrics(ca_pos, bond_tol=0.1, clash_tol=1.0):
    """CA-CA 键几何验证"""
    ca_pos = ca_pos * 10  # nm → Å
    ca_bond_dists = np.linalg.norm(
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - 3.8))  # 标准 CA-CA 距离 3.8Å
    ca_ca_valid = np.mean(ca_bond_dists < (3.8 + bond_tol))
    ca_ca_dists2d = np.linalg.norm(
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=1) > 0)]
    return {
        'ca_ca_deviation': ca_ca_dev,
        'ca_ca_valid_percent': ca_ca_valid,
        'num_ca_ca_clashes': np.sum(inter_dists < clash_tol),
    }
```

#### run_foldseek_parallel.sh（直接复制，无需改动）

```bash
# 从 ReQFlow 直接复制到 protein_DIFF/eval/run_foldseek_parallel.sh
```

### 4.3 新增的 CPFlow 特有模块

```python
# protein_DIFF/eval/metrics_diversity.py
# 结构层面多样性: 生成序列→AlphaFold2预测结构→pairwise TM-score

from protein_DIFF.eval.metrics import calc_tm_score
import itertools
import numpy as np

def structure_diversity(pdb_paths, seqs):
    """计算生成结构之间的 pairwise TM-score 均值"""
    # 加载所有结构的 CA 坐标
    coords = [_load_ca_coords(p) for p in pdb_paths]

    pairwise_tm = []
    for (c1, s1), (c2, s2) in itertools.combinations(list(zip(coords, seqs)), 2):
        tm1, tm2 = calc_tm_score(c1, c2, s1, s2)
        pairwise_tm.append(max(tm1, tm2))

    return np.mean(pairwise_tm)  # 越低 = 多样性越高


# protein_DIFF/eval/metrics_novelty.py
# FoldSeek 新颖性: 生成结构 → PDB 数据库搜索 → max TM-score

def structural_novelty(pdb_paths, foldseek_summary_csv):
    """解析 FoldSeek 结果, 统计新颖性"""
    import pandas as pd
    df = pd.read_csv(foldseek_summary_csv)
    return {
        'max_tm_mean': df['Max TM-score'].mean(),
        'max_tm_std': df['Max TM-score'].std(),
        'novelty_ratio': (df['Max TM-score'] < 0.5).mean(),  # TM < 0.5 视为新颖
    }
```

### 4.4 集成后的完整评估流程

```
0. CPFlow 推理 → FASTA 序列
      │
      ▼
1. AlphaFold2 / ESMFold 结构预测 → PDB
      │
      ▼
2. CPFlow 现有评估
   ├── metrics_sequence.py   → 催化基序, 序列一致性, 序列层面多样性
   ├── metrics_structure.py  → pLDDT, σ(ΔpLDDT), count(ΔpLDDT>10)
   └── metrics_training.py   → 训练曲线
      │
      ▼
3. 🆕 ReQFlow 移植评估
   ├── metrics.py            → TM-score, RMSD, 二级结构, CA-CA 几何
   ├── metrics_diversity.py  → 结构层面 pairwise TM 多样性
   └── metrics_novelty.py    → FoldSeek PDB 搜索新颖性
      │
      ▼
4. full_pipeline.py → 统一报告
```

---

## 五、对比：ReQFlow 的三支柱 vs CPFlow 论文的评估体系

| 评估维度 | ReQFlow 方法 | CPFlow 论文方法 | 互补性 |
|----------|-------------|----------------|--------|
| **可设计性** | scTM (自洽性 TM-score) | pLDDT + RMSD + TM-score vs WT | ReQFlow 自洽性也可用于 CPFlow: 生成序列→结构→ESMFold→结构→比较 |
| **多样性** | Pairwise TM-score (结构层面) | 成对序列一致性 (序列层面) | **互补**：CPFlow 没做结构层面多样性 |
| **新颖性** | FoldSeek vs PDB | 序列一致性 vs NCBI NR | **互补**：FoldSeek 是结构层面新颖性 |
| **结构几何** | CA-CA 键长、碰撞、二级结构 | 无 | **补充**：CPFlow 没做结构几何验证 |

> **关键发现**：CPFlow 论文缺少"结构层面多样性"和"结构几何验证"，这两项 ReQFlow 的代码可以直接补上。

---

## 六、需要新增安装的依赖

| 工具 | 用途 | 安装 |
|------|------|------|
| `tmtools` | Python 原生 TM-score 计算（替代 US-align） | `pip install tmtools` |
| `mdtraj` | 二级结构解析 + 回转半径 | `pip install mdtraj` |
| `FoldSeek` | 结构新颖性 PDB 搜索 | `conda install -c bioconda foldseek` |
| `GNU parallel` | FoldSeek 并行化 | `apt install parallel` |

> `tmtools` 替代 US-align 的好处：纯 Python API，无需 subprocess，更易集成。
