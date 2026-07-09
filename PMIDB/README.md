# PMIDB 数据集

**PMIDB**（Protein–Metabolite Interaction Database）在本仓库中仅包含 **人类（human）** 子目录：`PMIDB/human/`。内容为 KEGG 风格代谢物与 UniProt/Ensembl 蛋白 ID 的互作及人类蛋白网络特征。

---

## 目录结构

```text
PMIDB/
└── human/          ← 训练与 prepare 使用的数据根目录（datasets_processed: pmidb_human）
```

---

## 数据是什么（human）

| 类型 | 说明 |
|------|------|
| **PMI** | 代谢物（表中列名为 **KEGG**，值为 `CIDs*`）与蛋白（**Uniprot_KB_id**，如 `9606.ENSP00000499020`） |
| **规模** | 约 **14035** 条边，**4** 个代谢物，**6308** 个蛋白，正样本约 **3829**（`interaction` 为 1/True） |
| **代谢物–代谢物** | `m_m_links.tsv` **为空**（无 m–m 边） |
| **蛋白–蛋白** | 大规模 `p_p_links`（可分片存储） |
| **特征** | 代谢物 ChemGPT `(4, 384)`；蛋白 `(6308, 1024)` |

---

## PMI：`m_p_links.csv`

- **列**：`KEGG, Uniprot_KB_id, interaction`（**不是** `meta/protein`，但语义相同）
- `prepare_core` 使用 **前两列** 作为代谢物、蛋白 ID，第三列为标签
- `m_p_links_processed.csv`：经 `process_m_p_links_sample_neg.py` 处理，格式相同

---

## 嵌入对齐

### `meta.smi` ↔ `meta_ChemGPT-19M.npy`

- **4 行**；npy **`(4, 384)`**（维度与 STITCH/Piazza 的 256 不同，以本目录为准）
- 第 **i** 行 smi ↔ npy[i]；CID 与 PMI 的 `KEGG` 列一致

### `matched_protein_sequences.csv` ↔ `protein_large_model.npy`

- **6308 行** `Protein_ID, Sequence`
- **npy** `(6308, 1024)`，**行序与 CSV 一致**（从第 0 条序列对应第 0 行嵌入）

---

## 蛋白互作边 `p_p_links`

| 文件 | 说明 |
|------|------|
| `p_p_links.tsv` | 完整合并文件（约 162MB），Tab 三列：`蛋白1  蛋白2  weight` |
| `p_p_links_part1.tsv` … `part5.tsv` | 分片；**按 part 编号顺序拼接**即等于完整 `p_p_links.tsv` |

`prepare_core` 若缺少 `p_p_links.tsv`，会自动从 `p_p_links_part*.tsv` 合并生成。

### `m_m_links.tsv` / `meta_edge.edgelist`

- **空文件**：无代谢物–代谢物边；prepare 时代谢物相似图 `embeddings_meta` 可能全部为 fallback，**ChemGPT 大模型特征仍来自 npy**

### `protein_edge.edgelist`

- 由 `p_p_links` 映射到 PMI 蛋白编号后生成（体积大，可已预生成）

### `edges.csv`

- prepare 写出：整数 `meta_idx, protein_idx, label`

---

## 文件关系（human）

```text
m_p_links.csv
    ├─► KEGG 列 → 代谢物节点（仅 4 个）
    └─► Uniprot_KB_id → 蛋白节点（6308）

meta.smi ──► meta_ChemGPT-19M.npy
m_m_links.tsv（空）──► 无 m–m 结构边

matched_protein_sequences.csv ──► protein_large_model.npy
p_p_links.tsv（或 part1–5）──► protein_edge.edgelist
```

---

## 本仓库用法

- 配置键：`pmidb_human` → 目录 `PMIDB/human`
- 入口：`Prepare_pmidb_human_processed.py` / `main_pmidb_human_processed.py`（默认 processed CSV）
- OpenMP：若报错可设 `KMP_DUPLICATE_LIB_OK=TRUE`（`prepare_core` 已默认尝试设置）