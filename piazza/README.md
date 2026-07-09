# Piazza 数据集（大肠杆菌相关互作）

本目录为 **Piazza** 来源的 **代谢物–蛋白相互作用（PMI / m–p）** 及辅助图与预计算特征，供本仓库 `prepare_core` / `main_*` 构图与训练使用。物种背景与 STITCH 大肠杆菌节点命名一致（蛋白 ID 形如 `511145.b****`）。

---

## 数据是什么

| 类型 | 说明 |
|------|------|
| **PMI 监督边** | 代谢物（PubChem 风格 CID）与蛋白是否相互作用/关联；大量负样本 + 少量正样本 |
| **代谢物–代谢物边** | `m_m_links.tsv`，STITCH/Piazza 风格的代谢物相似或关联分数 |
| **蛋白–蛋白边** | `p_p_links.tsv`，蛋白互作或关联分数 |
| **节点特征** | 代谢物 ChemGPT 嵌入、蛋白大模型嵌入（与下方索引文件**行序对齐**） |
| **序列** | `511145.protein.sequences.v12.0.fa` 为原始 FASTA（辅助来源，训练管线主要用 `matched_protein_sequences.csv`） |

规模（原始 `m_p_links.csv`，约）：**17352** 条边，**18** 个代谢物，**964** 个蛋白，正样本约 **765**（`score` 为 `1`/`True`）。

---

## 文件格式与排列规则

### 1. `m_p_links.csv`（及 `m_p_links_processed.csv`）

- **列**：`meta, protein, score`
- **meta**：代谢物 ID，如 `CIDs00091493`
- **protein**：`511145.b0523` 等形式
- **score**：`0`/`1` 或 `False`/`True`（正/负标签）
- **行顺序**：无特殊要求；`prepare_core` 会按 CSV 行生成 `edges.csv`（内部再映射为 0…N-1 的节点编号）
- **processed**：`process_m_p_links_sample_neg.py` 从负样本中随机翻转约 1% 为正，正样本数略增；列名与行数与原始相同

### 2. `meta.smi` ↔ `meta_ChemGPT-19M.npy`

- **meta.smi**：每行 `SMILES CID`，**最后一列为 CID**，与 `m_p_links` 的 `meta` 一致  
  例：`CC(=O)... CIDs00091493`
- **meta_ChemGPT-19M.npy**：形状 `(18, 256)`，**第 i 行**对应 `meta.smi` **第 i 行**（按文件从上到下的顺序，从 0 起）
- **对齐方式**（代码）：`df_meta[df_meta.iloc[:,1] == cid].index[0]` 取行号再索引 npy

### 3. `matched_protein_sequences.csv` ↔ `protein_large_model.npy`

- **列**：`Protein_ID, Sequence`
- **行顺序**：第 i 行（不含表头）对应 `protein_large_model.npy` 的 **第 i 行**
- **npy**：形状 `(964, 1024)`
- **Protein_ID** 与 `m_p_links.protein`、`p_p_links.tsv` 中蛋白名一致

### 4. `m_m_links.tsv` / `p_p_links.tsv`

- **三列**（Tab 分隔）：`node1  node2  weight`（无表头）
- 节点名为 **原始字符串 ID**（CID 或蛋白 ID），不是 0-based 整数
- `prepare_core` 读取后写入 `meta_edge.edgelist` / `protein_edge.edgelist`（映射为 PMI 中出现的节点编号）

### 5. `meta_edge.edgelist` / `protein_edge.edgelist`

- 由 prepare 根据 `m_m_links` / `p_p_links` 与 **当前 PMI 中的 meta/protein 字典** 生成
- 格式：`src dst weight`（空格分隔，**已是整数节点编号**）

### 6. `edges.csv`

- prepare 生成：`meta_idx, protein_idx, label`（0-based，对应当前 PMI 子图的代谢物/蛋白列表）

### 7. 其它

- `m_p_links_piazza_in_stitch_nodes_not_in_pmi_*.csv`：与 `stitch_ecoli` 对照筛选的辅助结果（常为空或极少行）
- `_check_overlap.py` / `filter_piazza_vs_stitch.py`：与 STITCH 节点/边重叠分析脚本，非训练必需

---

## 各文件关系（数据流）

```text
m_p_links.csv ──────────────┐
                            ├──► edges.csv（训练标签边，整数节点）
meta.smi ──► meta_ChemGPT-19M.npy ──► 代谢物节点特征
m_m_links.tsv ──► meta_edge.edgelist ──► 代谢物相似图 / 超图

matched_protein_sequences.csv ──► protein_large_model.npy ──► 蛋白节点特征
p_p_links.tsv ──► protein_edge.edgelist ──► 蛋白相似图 / 超图
```

- **PMI 文件**决定有哪些代谢物、蛋白进入子图及正负样本。
- **边文件**（m_m、p_p）可在更大节点集上定义，prepare 时仅保留能映射到 PMI 节点字典的边。
- 本仓库配置键：`datasets_processed.py` 中 `piazza` → 默认 `m_p_links_processed.csv`；本地 CV 默认 `m_p_links.csv`。

---

## 与 STITCH 合并

跨源实验使用 `stitch_piazza_ecoli_400/` 等合并目录时，嵌入需按合并后的 `meta.smi` / `matched_protein_sequences.csv` **重新拼接**（见 `merge_stitch_piazza_ecoli.py`），不可只复制 STITCH 的 npy。