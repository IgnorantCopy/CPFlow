# CPFlow 评估资源与耗时估算

> 基准：100 条序列 × ~770 AA (KmAgo)，单机环境
> 标注 🟢 可在普通笔记本运行，🟡 需要 GPU 或工作站，🔴 需要集群或长时间等待

---

## 总览

| Phase | 文件 | 硬件 | 100 条耗时 | 内存 | 显存 |
|:--:|------|:--:|:--:|:--:|:--:|
| 1 | `metrics_sequence.py` | 🟢 CPU | < 1s | ~10 MB | — |
| 1b | `metrics_aa_properties.py` | 🟢 CPU | < 1s | ~10 MB | — |
| 2 | `metrics_efficiency.py` | 🟢 CPU | < 1s | ~5 MB | — |
| 3 | `metrics_structure.py` | 🟢 CPU | 5-10 min | ~500 MB | — |
| 4 (ESMFold) | `predict_structures.py` | 🟡 GPU | 1-4 h | ~2 GB | 8-16 GB |
| 4 (AlphaFold2) | `predict_structures.py` | 🔴 GPU | 16-50 h | ~4 GB | 16-24 GB |
| 4 (pLDDT filter) | `predict_structures.py filter` | 🟢 CPU | < 1s | ~10 MB | — |
| 5 (MUSCLE) | `metrics_phylogeny.py` | 🟢 CPU | 1-5 min | ~2 GB | — |
| 5 (IQ-TREE) | `metrics_phylogeny.py` | 🟡 CPU | 5-30 min | ~4 GB | — |
| 6 (BLAST) | `metrics_novelty.py blast` | 🟢 CPU | 5-10 min* | ~50 MB | — |
| 6 (FoldSeek) | `metrics_novelty.py foldseek` | 🟡 CPU | 10-30 min | ~2 GB | — |
| Spearman | `metrics_spearman.py` | 🟡 GPU | 10-30 min | ~2 GB | 4-8 GB |
| Pipeline | `full_pipeline.py` | 取决于启用的 Phase | 2-50 h | — | — |

> *BLAST 受 NCBI 速率限制，含 2s/条的 sleep。无 rate limit 纯查询时间 < 1 min。

---

## 逐项详细

### Phase 1: `metrics_sequence.py` — 🟢 CPU, < 1s

| 资源 | 估算 |
|------|------|
| **CPU** | 任意，单核 |
| **内存** | ~10 MB（100 条 × 770 AA 字符串） |
| **磁盘** | 输入 CSV/FASTA ~200 KB |
| **耗时** | < 1s |
| **瓶颈** | 成对一致性 O(n²)：100 条 = 4950 对，每对比较 770 字符 → ~3.8M 字符比较 |

**1000 条序列时**：~50 万对比较 → ~5s，仍可接受。

---

### Phase 1b: `metrics_aa_properties.py` — 🟢 CPU, < 1s

与 Phase 1 相同量级。每序列逐位分类 770 个 AA 到 4 个属性组。

---

### Phase 2: `metrics_efficiency.py` — 🟢 CPU, < 1s

仅解析训练 CSV 文件（~100 行），纯数值计算。

---

### Phase 3: `metrics_structure.py` — 🟢 CPU, 5-10 min

| 资源 | 估算 |
|------|------|
| **CPU** | 单核即可，tmtools 内部是 C++ |
| **内存** | ~500 MB（加载 100 个 PDB + TM-align 中间矩阵） |
| **磁盘** | 100 个 PDB × ~800 KB = ~80 MB |
| **耗时** | 5-10 min |

耗时分解：

| 子步骤 | 单条耗时 | ×100 总耗时 |
|--------|:--:|:--:|
| `load_ca_from_pdb` | < 0.1s | < 10s |
| `tmtools.tm_align` (vs WT) | 1-3s | 2-5 min |
| `calc_ca_ca_metrics` | < 0.1s | < 10s |
| `calc_secondary_structure` (mdtraj DSSP) | 0.5-1s | 1-2 min |

**瓶颈**：`tm_align` 的 C++ 实现本身已很快（TM-align 是领域最快的比对算法）。100 条序列约 2-5 分钟。

---

### Phase 4: `predict_structures.py` — 🟡/🔴 GPU

#### 4a. ESMFold（推荐初筛）

| 资源 | 估算 |
|------|------|
| **GPU** | RTX 3090/4090 (24 GB) 或 A100 (40/80 GB) |
| **显存** | 8-16 GB（770 AA；可通过 `--chunk_size 64` 降到 ~8 GB） |
| **内存** | ~2 GB |
| **磁盘** | 100 个 PDB × ~800 KB = ~80 MB |
| **耗时（单条）** | RTX 3090: 2-5 min；A100: 30s-2 min |
| **耗时（100 条）** | RTX 3090: 3-8 h；A100: 1-4 h |

> 注意：ESMFold 不支持 batch inference，需逐条运行。如果 OOM，调 `--chunk_size` 降低轴向注意力内存。

#### 4b. AlphaFold2 via ColabFold（论文级精度）

| 资源 | 估算 |
|------|------|
| **GPU** | A100 (40/80 GB) 推荐，V100 (32 GB) 勉强 |
| **显存** | 16-24 GB（5 模型并行时更多） |
| **内存** | ~4 GB |
| **磁盘** | 100 × 5 模型 × ~2 MB = ~1 GB |
| **耗时（单条）** | ~10-30 min（含 MSA 服务器查询 + 5 模型推理） |
| **耗时（100 条）** | 16-50 h |

> ColabFold 瓶颈在 MSA 服务器查询（网络延迟）和 5 个模型串行。**实际使用建议只对筛选后剩余的 ~30 条序列跑 AlphaFold2**，而非全部 100 条。

#### 4c. pLDDT 三级筛选

纯 numpy 操作，< 1s。

---

### Phase 5: `metrics_phylogeny.py` — 🟢/🟡 CPU

| 资源 | 估算 |
|------|------|
| **CPU** | MUSCLE: 单核；IQ-TREE: 多核（`-T AUTO`） |
| **内存** | MUSCLE: ~1-2 GB；IQ-TREE: ~2-4 GB |
| **磁盘** | MSA 文件 ~1 MB；树文件 ~100 KB |

#### MUSCLE（多序列比对）

| 序列数 × 长度 | 耗时 |
|:--|:--|
| 101 × 770 AA（WT + 100 生成） | 1-3 min |
| 795 × 770 AA（+694 天然 pAgo） | 5-15 min |

#### IQ-TREE（系统发育树）

| 参数 | 耗时 |
|:--|:--|
| 101 条，BLOSUM62，1500 bootstrap | 5-15 min |
| 795 条，同上 | 30-90 min |

> bootstrap 是主要耗时来源。如果只做快速测试，可降到 `-B 100`。

---

### Phase 6: `metrics_novelty.py`

#### 6a. BLAST（在线 NCBI NR）

| 资源 | 估算 |
|------|------|
| **CPU** | 任意 |
| **内存** | ~50 MB |
| **磁盘** | XML 结果 ~100 KB/条 |

| 序列数 | 耗时 | 说明 |
|:--|:--|------|
| 1 条 | 10-30s | 网络延迟 + NCBI 处理 |
| 3-5 条 | 1-3 min | 含 2s/条的 sleep（rate limit） |
| 100 条 | 5-10 min | 受 NCBI rate limit 约束 |

> **NCBI 未认证用户速率限制**：约 1 req/2s。如果被 throttle，需等待更久或注册 API key。建议首次用 `--max_seqs 3` 测试连通性。

#### 6b. FoldSeek（离线 PDB 搜索）

| 资源 | 估算 |
|------|------|
| **CPU** | 多核（GNU parallel 默认 50% 核心） |
| **内存** | ~2 GB |
| **磁盘** | PDB 数据库 ~50 GB（一次性下载） |

| 结构数 | 耗时 |
|:--|:--|
| 30 条（筛选后 designable） | 5-10 min |
| 100 条 | 10-30 min |

> FoldSeek 的 `--exhaustive-search` 是全库比对，耗时与 PDB 数据库大小（~200K 条目）成正比。并行度由 `run_foldseek_parallel.sh` 的 `JOBS` 变量控制。

---

### Spearman: `metrics_spearman.py` — 🟡 GPU

| 资源 | 估算 |
|------|------|
| **GPU** | 需要（加载完整 CPDiffusion 模型 + 推理） |
| **显存** | 4-8 GB（模型 ~400 万参数 + graph data） |
| **内存** | ~2 GB |
| **耗时** | 每个 ProteinGym 蛋白 ~2-5 min；全部 ~28 个蛋白 ~30-60 min |

> 每个蛋白需要 10 次 forward pass 取平均 + 对每个单点突变计算交叉熵。ProteinGym 数据集约 28 个蛋白质，总计约数千个突变。

---

## 推荐执行策略

### 最小验证（< 10 min，笔记本可跑）

```bash
# 只跑 Phase 1 + 1b + 2（零依赖，秒级）
python protein_DIFF/eval/metrics_sequence.py --csv result/predict/predict.csv --wt_fasta wt.fasta
python protein_DIFF/eval/metrics_aa_properties.py --csv result/predict/predict.csv --wt_fasta wt.fasta
```

### 快速全评估（~2-4 h，单 GPU 工作站）

```bash
# ESMFold + 全部 Phase（跳过 AlphaFold2 和 FoldSeek）
python protein_DIFF/eval/full_pipeline.py \
    --csv result/predict/predict.csv \
    --wt_fasta wt.fasta \
    --engine esmfold \
    --skip_blast --skip_foldseek
```

### 论文级全评估（~20-60 h，A100 集群）

```bash
# AlphaFold2 + 全部 Phase
python protein_DIFF/eval/full_pipeline.py \
    --csv result/predict/predict.csv \
    --wt_fasta wt.fasta \
    --engine alphafold
```
