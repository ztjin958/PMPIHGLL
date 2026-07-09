合并数据集 piazza (m_p_links.csv) + PMIDB/ecoil (m_p_links.csv)
- m_p_links_merged.csv / m_p_links.csv: 列 meta, protein, target, source
- PMIDB 源 CSV 列映射: KEGG->meta, Uniprot_KB_id->protein, interaction->target
- m_m_links.tsv / p_p_links.tsv: 三列 node1\tnode2\tscore；来自两侧目录已有 TSV 融合后按 PMI 节点过滤
- source 取值: pmidb_ecoil, piazza, 或二者用分号连接
