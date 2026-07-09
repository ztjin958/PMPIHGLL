# STITCH 大肠杆菌（E. coli）数据集

本目录为 **STITCH** 数据库在 **大肠杆菌** 上的代谢物–蛋白互作及代谢物/蛋白关联网络，附带 ChemGPT 与蛋白大模型嵌入。本仓库中数据集键：`stitch_ecoli_400`、`stitch_ecoli_700`（区别主要在 PMI 正样本阈值与标签列名，节点规模见下表）。

---

## 数据是什么

| 类型 | 说明 |
|------|------|
| **PMI** | 代谢物（`CIDs*`）与蛋白（`511145.b*`）的关联；含大量负样本 |
| **400 / 700** | 两套 PMI 列表：同一批 **39585** 条边、**29** 个代谢物、**1365** 个蛋白；正样本数不同（400 约 **3996**，700 约 **2319**） |
| **m_m / p_p** | 代谢物–代谢物、蛋白–蛋白 关联强度（STITCH combined score 等） |
| **特征** | `meta_ChemGPT-19M.npy`、`protein_large_model.npy` 与索引文件**逐行对齐** |

---

## PMI 文件

| 文件 | 标签列 | 行数（约） | 正样本（约） |
|------|--------|------------|--------------|
| `m_p_links_400.csv` | `target` | 39585 | 3996 |
| `m_p_links_700.csv` | `score` | 39585 | 2319 |
| `*_processed.csv` | 同左 | 同左 | +约 1% 负翻正 |

- **列**：`meta, protein, target` 或 `meta, protein, score`
- **排列**：行顺序任意；prepare 按行建 `edges.csv` 并重编号
- **400 vs 700**：节点集合相同，**正/负划分不同**（置信度阈值不同），用于不同难负样本设定

---

## 嵌入与索引（必须对齐）

### `meta.smi` ↔ `meta_ChemGPT-19M.npy`

- **29 行** SMILES + CID；**npy** 形状 `(29, 256)`
- **第 i 行 smi** ↔ **npy[i]**

### `matched_protein_sequences.csv` ↔ `protein_large_model.npy`

- **1364 行**蛋白（CSV 行数）；**npy** `(1364, 1024)`
- PMI 中可能出现 **1365** 个 unique protein：若有蛋白不在 CSV/npy 中，prepare 会对该节点用**随机向量**并打印 `Not in protein_large_model`

---

## 边文件

### `m_m_links.tsv`

- Tab 三列：`CID1  CID2  weight`（无表头）
- 用于构建代谢物相似图 → `meta_edge.edgelist`（整数节点 id）

### `p_p_links.tsv`

- Tab 三列：`511145.b*  511145.b*  weight`
- → `protein_edge.edgelist`

### `meta_edge.edgelist` / `protein_edge.edgelist`

- 空格分隔：`src dst weight`（**已映射**为 PMI 子图中的 0…N-1 编号）
- 若已存在，prepare 默认**不覆盖**（与 `Prepare.py` 行为一致）

### `edges.csv`

- prepare 写出：`meta_idx, protein_idx, 0/1`

---

## 文件关系示意

```text
m_p_links_400.csv 或 m_p_links_700.csv
        │
        ├─► 定义 meta/protein 节点表 + 监督边
        │
meta.smi ──────────► meta_ChemGPT-19M.npy
m_m_links.tsv ─────► meta_edge.edgelist ──► 代谢物超图 / 相似矩阵

matched_protein_sequences.csv ──► protein_large_model.npy
p_p_links.tsv ──────────────────► protein_edge.edgelist ──► 蛋白超图
```

---

## 本仓库用法

- `datasets_processed.py`：`stitch_ecoli_400` / `stitch_ecoli_700`
- 本地 CV 默认：`m_p_links_400.csv` / `m_p_links_700.csv`（非 processed）
- 与 `piazza/` 合并：`merge_stitch_piazza_ecoli.py` → `stitch_piazza_ecoli_400|700/`