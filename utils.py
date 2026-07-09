import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

def edgelist_to_matrix(l,file_path):
    # 读取edgelist文件
    # 假设文件格式：node1 node2 weight
    if os.path.getsize(file_path) == 0:
        return None
    df = pd.read_csv(file_path, sep='\s+', header=None, names=['node1', 'node2', 'weight'])
    # if len(df)>5e5:
    #     df = df.sample(n=int(5e5), random_state=42)
    # 创建零矩阵
    matrix = np.zeros((l, l))

    # 填充矩阵
    for _, row in df.iterrows():
        i, j, w = int(row['node1']), int(row['node2']), row['weight']
        if i < 0 or j < 0 or i >= l or j >= l:
            continue
        matrix[i][j] = w / 1000
        # 如果是无向图，取消下面一行的注释
        matrix[j][i] = w/1000
    np.fill_diagonal(matrix,val=1)
    return matrix



# 将N*M列的超图矩阵 转换为2*H列的超图矩阵
def convert_adjacency_matrix(adj_matrix):
    list = []
    for adj in adj_matrix:
        if isinstance(adj, np.ndarray):
            adj = torch.from_numpy(adj)

            # 确保输入是torch张量并移动到指定设备
        if not isinstance(adj, torch.Tensor):
            raise ValueError("输入必须是numpy数组或torch张量")
        adj = adj.to(device)
        # 获取矩阵的大小
        n = adj.shape[1]

        # 初始化两个空列表
        rows = []
        cols = []

        # 遍历每一列
        for col in range(n):
            # 找到该列中值为1的行索引
            row_indices = torch.where(adj[:, col] == 1)[0]
            # 添加列索引和对应的行索引到结果列表
            rows.extend(row_indices.tolist())
            cols.extend([col] * len(row_indices))

        # 构造最终的 2×M 矩阵
        result_matrix = torch.tensor([rows, cols], dtype=torch.int64, device=device)
        list.append(result_matrix)
    return list


def convert_hypergraph_to_pyg(H_list, eps=1e-8, device=None):
    """
    将 N×M 加权超图关联矩阵列表转为 PyG 格式（H5：支持 is_probH 等非零权）。

    返回:
        hyperedge_index_list: 每个元素 shape [2, num_incidences]
        incidence_weight_list: 每条关联 (node, hyperedge) 的权重，shape [num_incidences]
        hyperedge_weight_list: 每条超边标量权（列非零权均值），shape [num_hyperedges]
    """
    if device is None:
        device = globals().get("device", "cpu")

    hyperedge_index_list = []
    incidence_weight_list = []
    hyperedge_weight_list = []

    for H in H_list:
        if isinstance(H, np.ndarray):
            H_t = torch.from_numpy(H.astype(np.float32))
        elif isinstance(H, torch.Tensor):
            H_t = H.float()
        else:
            raise ValueError("H 必须是 numpy 或 torch 张量")
        H_t = H_t.to(device)
        n_nodes, n_edges = H_t.shape

        rows, cols, vals = [], [], []
        col_means = torch.zeros(n_edges, dtype=torch.float32, device=device)

        for col in range(n_edges):
            col_vec = H_t[:, col]
            nz_mask = col_vec > eps
            if not nz_mask.any():
                continue
            nz_idx = torch.where(nz_mask)[0]
            nz_val = col_vec[nz_idx]
            col_means[col] = nz_val.mean()
            for r, v in zip(nz_idx.tolist(), nz_val.tolist()):
                rows.append(r)
                cols.append(col)
                vals.append(v)

        if len(rows) == 0:
            hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            incidence_weight = torch.zeros(0, dtype=torch.float32, device=device)
            hyperedge_weight = torch.zeros(n_edges, dtype=torch.float32, device=device)
        else:
            hyperedge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
            incidence_weight = torch.tensor(vals, dtype=torch.float32, device=device)
            hyperedge_weight = col_means

        hyperedge_index_list.append(hyperedge_index)
        incidence_weight_list.append(incidence_weight)
        hyperedge_weight_list.append(hyperedge_weight)

    return hyperedge_index_list, incidence_weight_list, hyperedge_weight_list


def compute_distance_matrix(X, metric='cosine', p=3):
    """
    计算节点特征矩阵的距离矩阵
    参数:
        X: numpy array, shape (N, d), 节点特征矩阵
        metric: 距离度量方法，支持 'euclidean', 'manhattan', 'minkowski', 'cosine'
        p: 闵可夫斯基距离的阶数，仅在 metric='minkowski' 时使用
    返回:
        dist_matrix: numpy array, shape (N, N), 距离矩阵
    """
    N = X.shape[0]

    if metric == 'euclidean':
        # 欧几里得距离：||x-y||^2 = ||x||^2 + ||y||^2 - 2x·y
        dot_product = np.dot(X, X.T)
        square_norm = np.sum(X ** 2, axis=1)
        dist_matrix = square_norm.reshape(-1, 1) + square_norm
        dist_matrix = np.sqrt(np.maximum(dist_matrix - 2 * dot_product, 0))

    elif metric == 'manhattan':
        # 曼哈顿距离：sum(|x_i - y_i|)
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            dist_matrix[i] = np.sum(np.abs(X - X[i]), axis=1)

    elif metric == 'minkowski':
        # 闵可夫斯基距离：(sum(|x_i - y_i|^p))^(1/p)
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            dist_matrix[i] = np.power(np.sum(np.power(np.abs(X - X[i]), p), axis=1), 1 / p)

    elif metric == 'cosine':
        # 余弦距离：1 - cosine_similarity
        norms = np.sqrt(np.sum(X ** 2, axis=1))
        if np.any(norms == 0):
            raise ValueError("特征矩阵中存在零向量")
        normalized_X = X / norms[:, np.newaxis]
        cosine_similarity = np.dot(normalized_X, normalized_X.T)
        dist_matrix = 1 - cosine_similarity
        dist_matrix = np.maximum(dist_matrix, 0)  # 避免数值误差导致负值

    elif metric == 'gaussian':
        """
        計算輸入矩陣 X 的高斯核相似度矩陣。

        參數:
            X (array-like): 形狀為 (N, P) 的輸入矩陣，N 是樣本數，P 是特徵數
            sigma (float): 高斯核的帶寬參數，默認為 1.0

        返回:
            array: 形狀為 (N, N) 的相似度矩陣，元素為 X 中每對樣本的高斯相似度
        """
        sigma = 1.0
        # 將輸入轉換為 NumPy 陣列
        X = np.array(X)
        N = X.shape[0]

        # 計算所有樣本對的平方歐幾里得距離
        # 使用廣播計算：||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i.dot(x_j)
        squared_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)  # ||x_i||^2
        dot_product = np.dot(X, X.T)  # x_i.dot(x_j)
        squared_dist = squared_norm + squared_norm.T - 2 * dot_product  # ||x_i - x_j||^2

        # 計算高斯核相似度
        similarity_matrix = np.exp(-squared_dist / (2 * sigma ** 2))
        return 1-similarity_matrix
    elif metric == 'Jaccard':
        """
            計算輸入矩陣 X 的 Jaccard 相似度矩陣，假設 X 是二值數據。

            參數:
                X (array-like): 形狀為 (N, P) 的二值矩陣，N 是樣本數，P 是特徵數

            返回:
                array: 形狀為 (N, N) 的 Jaccard 相似度矩陣
            """
        X = np.array(X, dtype=bool)  # 確保輸入是二值數據
        N = X.shape[0]

        # 計算交集和並集
        intersection = np.dot(X, X.T)  # |A ∩ B|
        sum_X = np.sum(X, axis=1).reshape(-1, 1)
        union = sum_X + sum_X.T - intersection  # |A ∪ B|

        # 避免除以 0
        similarity_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float),
                                      where=union != 0)
        return 1 - similarity_matrix
    else:
        raise ValueError("不支持的距离度量方法，仅支持 'euclidean', 'manhattan', 'minkowski', 'cosine'")

    # 确保对角线为 0
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix

def cos_dis(X):
    """
    计算余弦相似度矩阵（1 - cosine similarity 即为距离）
    :param X: N x d 的节点特征矩阵
    :return: N x N 的距离矩阵
    """
    cos_sim = cosine_similarity(X)
    return 1 - cos_sim  # 返回的是距离矩阵


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    """
    根据距离矩阵构造超图的顶点-边矩阵
    :param dis_mat: 节点之间的距离矩阵
    :param k_neig: 每个节点的 k 最近邻
    :param is_probH: 是否构造概率型的顶点-边矩阵（默认为二值矩阵）
    :param m_prob: 控制概率型矩阵的尺度
    :return: N x M 的超图顶点-边矩阵
    """
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))

    for center_idx in range(n_obj):
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.argsort(dis_vec)[:k_neig]  # 选择最近的 k 个邻居
        avg_dis = np.mean(dis_vec)

        for node_idx in nearest_idx:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0  # 二值化
    return H


def hyperedge_concat(*H_list):
    """
    合并多个超图矩阵
    :param H_list: 包含多个超图矩阵的列表
    :return: 合并后的超图矩阵
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            if H is None:
                H = h
            else:
                if isinstance(h, np.ndarray):  # 对于普通的矩阵
                    H = np.hstack((H, h))
                else:  # 如果是列表类型，逐个拼接
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def construct_H_with_KNN(X, K_neigs=[8], metric = 'cosine', split_diff_scale=True, is_probH=False, m_prob=1):
    """
    从原始节点特征矩阵构造多尺度超图的顶点-边矩阵
    :param X: N x d 的节点特征矩阵
    :param K_neigs: 一个整数列表，表示每个超图的邻居数量
    :param split_diff_scale: 是否按不同尺度分开构造超边矩阵
    :param is_probH: 是否构造概率型的顶点-边矩阵
    :param m_prob: 控制概率的超参数
    :return: 超图的顶点-边矩阵，可能是一个合并后的矩阵或多个矩阵列表
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])  # 确保输入是二维矩阵

    if isinstance(K_neigs, int):
        K_neigs = [K_neigs]  # 如果只有一个邻居数量，则将其转换为列表

    dis_mat = compute_distance_matrix(X,metric)  # 计算距离矩阵（余弦距离）

    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)  # 构造超图
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)  # 合并超图矩阵
        else:
            H.append(H_tmp)  # 如果需要分开，保存为列表
    # min_val = np.min(dis_mat)
    # max_val = np.max(dis_mat)
    # normalized_arr = (dis_mat - min_val) / (max_val - min_val)
    # np.fill_diagonal(normalized_arr, 1)
    # result = np.sum(normalized_arr * (H[0] == 1), axis=0)
    # result = torch.tensor(result, dtype=torch.float32).to(device)
    return H



# 根据蛋白质的SEQ 获得对应的特征 长度343
def find_amino_acid(x):
    return ('B' in x) | ('O' in x) | ('J' in x) | ('U' in x) | ('X' in x) | ('Z' in x)


# encode amino acid sequence using CT
def CT(sequence):
    classMap = {'G': '1', 'A': '1', 'V': '1', 'L': '2', 'I': '2', 'F': '2', 'P': '2',
                'Y': '3', 'M': '3', 'T': '3', 'S': '3', 'H': '4', 'N': '4', 'Q': '4', 'W': '4',
                'R': '5', 'K': '5', 'D': '6', 'E': '6', 'C': '7'}

    seq = ''.join([classMap[x] for x in sequence])
    length = len(seq)
    coding = np.zeros(343, dtype=np.int64)
    for i in range(length - 2):
        index = int(seq[i]) + (int(seq[i + 1]) - 1) * 7 + (int(seq[i + 2]) - 1) * 49 - 1
        coding[index] = coding[index] + 1
    return coding


# 序列CT编码
def sequence_CT(gene_entry_seq):
    # ambiguous_index = gene_entry_seq.loc[gene_entry_seq[1].apply(find_amino_acid)].index
    # gene_entry_seq.drop(ambiguous_index, axis=0, inplace=True)
    # gene_entry_seq.index = range(len(gene_entry_seq))
    # print("after filtering:", gene_entry_seq.shape)
    # print("encode amino acid sequence using CT...")
    import re
    def remove_non_standard_aa(sequence):
        return re.sub(r'[BOXJUZ]', '', sequence)

    CT_list = []
    for seq in gene_entry_seq[1].values:
        CT_list.append(CT(remove_non_standard_aa(seq)))
    gene_entry_seq[1] = CT_list

    return gene_entry_seq



def balance_tensor(tensor):
    # 1. 找到 0 和 1 的索引
    zero_indices = (tensor == 0).nonzero(as_tuple=True)[0]  # 0元素索引
    one_indices = (tensor == 1).nonzero(as_tuple=True)[0]  # 1元素索引

    # 2. 计算需要保留的 0 元素数量
    num_ones = one_indices.shape[0]
    num_zeros = zero_indices.shape[0]
    print(f"0元素个数: {num_zeros}, 1元素个数: {num_ones}")

    if num_zeros > num_ones:
        # 3. 随机选择要删除的 0 元素索引
        perm = torch.randperm(num_zeros)  # 随机打乱 0 元素的索引
        delete_count = num_zeros - num_ones  # 需要删除的 0 数量
        delete_indices = zero_indices[perm[:delete_count]]  # 随机选择要删除的索引

        # 4. 构造掩码，标记保留的元素
        mask = torch.ones_like(tensor, dtype=torch.bool)  # 默认全保留
        mask[delete_indices] = False  # 将要删除的索引标记为 False

        # 5. 使用掩码保留元素，保持原始顺序
        balanced_tensor = tensor[mask]
        # print(f"随机删除了 {len(delete_indices)} 个 0 元素")

        return balanced_tensor, delete_indices
    else:
        print("0元素数量已经小于等于1元素数量，无需删除")
        return tensor, torch.tensor([])




def load_edge_index(file_path, sample_size=None):
    # 读取文件
    df = pd.read_csv(file_path, sep=" ", header=None, names=["src", "tgt", "weight"])

    # 检查行数
    total_edges = len(df)
    # print(f"文件总边数: {total_edges}")
    sampled_df = df
    # 随机抽取 500,000 行
    if sample_size:
        if total_edges <= sample_size:
            sampled_df = df
        else:
            sampled_df = df.sample(n=sample_size, random_state=42)  # random_state 确保可重复性

    # 提取边索引和权重
    edge_index = sampled_df[["src", "tgt"]].astype(int).values  # 转换为整数数组
    edge_weight = sampled_df["weight"].astype(float) / 1000  # 转换为浮点数并除以 1000

    # 转换为 PyTorch 张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # (2, num_edges)
    edge_weight = torch.tensor(edge_weight.values, dtype=torch.float32)  # (num_edges,)

    # print(f"抽样后边数: {edge_index.shape[1]}")
    return edge_index, edge_weight



def process_data(arr, r, fold):
    """
    R 轮 RUS × fold 折指标汇总。
    - R>1：每轮 RUS 先对 fold 折求均值，再对 R 个均值求 mean±std（与 main.py 一致）。
    - R=1：对 fold 折直接 mean±std（不能对 1 个 RUS 均值用 ddof=1，否则 std=nan）。
    """
    data = np.asarray(arr, dtype=float)
    expected = r * fold
    if data.size != expected:
        raise ValueError(
            f"process_data: 需要 {expected} 个分数 (R={r}, fold={fold})，当前 {data.size}"
        )

    if r == 1:
        final_mean = float(np.mean(data))
        final_std = float(np.std(data, ddof=1)) if fold > 1 else 0.0
    else:
        groups = data.reshape(r, fold)
        group_means = np.mean(groups, axis=1)
        final_mean = float(np.mean(group_means))
        final_std = float(np.std(group_means, ddof=1)) if r > 1 else 0.0

    return f"{final_mean:.5f} ± {final_std:.3f}"


def summarize_rus_cv_metrics(scores, R: int, n_splits: int) -> dict:
    """
    与 process_data / main.py 一致：
    R>1：每轮 RUS 的 fold 均值 → 对 R 轮求 mean±std；
    R=1：在 n_splits 折上直接 mean±std（勿对单个 RUS 均值用 ddof=1）。
    """
    data = np.asarray(scores, dtype=float)
    formatted = process_data(scores, R, n_splits)
    parts = formatted.split("±")
    final_mean = float(parts[0].strip())
    final_std = float(parts[1].strip())

    out: dict = {
        "R": R,
        "n_splits": n_splits,
        "n_scores": int(data.size),
        "mean": final_mean,
        "std": final_std,
        "formatted": formatted,
        "per_fold_scores": [float(x) for x in data],
    }
    if R == 1:
        out["aggregation"] = "mean_std_over_folds"
        out["per_rus_fold_means"] = None
    else:
        out["aggregation"] = "mean_std_over_rus_means"
        groups = data.reshape(R, n_splits)
        out["per_rus_fold_means"] = [float(x) for x in np.mean(groups, axis=1)]
    return out


