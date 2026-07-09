# STITCH 酿酒酵母（S. cerevisiae）数据集

本目录为 **STITCH** 在 **酿酒酵母** 上的代谢物–蛋白互作及关联网络，结构与 `stitch_ecoli/` 相同。数据集键：`stitch_yeast_400`、`stitch_yeast_700`。

---

## 数据是什么

| 类型 | 说明 |
|------|------|
| **PMI** | 代谢物 `CIDs*` 与酵母蛋白（`4932.Y*` 基因名格式） |
| **400 / 700** | 两套 PMI：**47150** 条边、**41** 个代谢物、**1150** 个蛋白；正样本约 **4065**（400）与 **1879**（700） |
| **m_m / p_p** | 代谢物、蛋白关联边（STITCH 分数） |
| **特征** | ChemGPT `(41, 256)`，蛋白嵌入 `(1150, 1024)` |

---

## PMI 文件

| 文件 | 标签列 | 行数 | 正样本（约） |
|------|--------|------|--------------|
| `m_p_links_400.csv` | `score` | 47150 | 4065 |
| `m_p_links_700.csv` | `score` | 47150 | 1879 |
| `*_processed.csv` | 同左 | 同左 | 略增（负采样翻转） |

- **列**：`meta, protein, score`
- **蛋白 ID 示例**：`4932.YLL060C`（与 `matched_protein_sequences.csv` 的 `Protein_ID` 一致）

---

## 排列与对齐（与 stitch_ecoli 相同规则）

### `meta.smi` ↔ `meta_ChemGPT-19M.npy`

- **41 行**；npy 第 **i** 行对应 smi 第 **i** 行（CID 在最后一列）

### `matched_protein_sequences.csv` ↔ `protein_large_model.npy`

- **1150 行** ↔ `(1150, 1024)`，**行序一致**

### `m_m_links.tsv` / `p_p_links.tsv`

- Tab 三列，**原始字符串 ID**，无表头
- prepare 映射为 PMI 子图内的整数 id 后写入 `*_edge.edgelist`

### `edges.csv`

- prepare 生成：`meta_idx, protein_idx, label`

---

## 文件关系

```text
m_p_links_400.csv 或 m_p_links_700.csv  →  节点集 + PMI 监督
meta.smi + meta_ChemGPT-19M.npy         →  代谢物特征
m_m_links.tsv                           →  meta_edge.edgelist
matched_protein_sequences.csv
  + protein_large_model.npy             →  蛋白特征
p_p_links.tsv                           →  protein_edge.edgelist
```

---

## 本仓库用法

- `datasets_processed.py`：`stitch_yeast_400` / `stitch_yeast_700`
- 本地 CV：`main_local_stitch_yeast_*_{protein,meta}.py`，默认原始 CSV