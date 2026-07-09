import warnings
warnings.filterwarnings('ignore')
import torch
from torch import nn
from torch_geometric.nn import HypergraphConv
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

class protein_HGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(protein_HGNN, self).__init__()

        self.conv1 = HypergraphConv(in_channels, hidden_channels, dropout=0.3)
        self.pb1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, out_channels, dropout=0.3)


    def forward(self, x, hyperedge_index, hyperedge_weight=None, hyperedge_attr=None):
        x = self.conv1(x, hyperedge_index, hyperedge_weight=hyperedge_weight, hyperedge_attr=hyperedge_attr)
        x = self.pb1(x)
        x = self.conv2(x, hyperedge_index, hyperedge_weight=hyperedge_weight)

        return x

class meta_HGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HypergraphConv(in_channels, hidden_channels, dropout=0.3)
        self.pb1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, out_channels, dropout=0.3)


    def forward(self, x, hyperedge_index, hyperedge_weight=None, hyperedge_attr=None):
        x = self.conv1(x, hyperedge_index, hyperedge_weight=hyperedge_weight, hyperedge_attr=hyperedge_attr)
        x = self.pb1(x)
        x = self.conv2(x, hyperedge_index, hyperedge_weight=hyperedge_weight)

        return x

class HGCN(torch.nn.Module):
    def __init__(self, protein_dim, meta_dim):
        super(HGCN, self).__init__()
        self.protein_conv_1 = protein_HGNN(protein_dim, protein_dim*2, protein_dim)
        self.protein_conv_2 = protein_HGNN(protein_dim, protein_dim*2, protein_dim)


        self.meta_conv_1 = meta_HGNN(meta_dim, meta_dim*2, meta_dim)
        self.meta_conv_2 = meta_HGNN(meta_dim, meta_dim*2, meta_dim)

        self.p12 = nn.Parameter(torch.tensor(0.0))
        self.p13 = nn.Parameter(torch.tensor(0.0))
        self.p23 = nn.Parameter(torch.tensor(0.0))
        self.m12 = nn.Parameter(torch.tensor(0.0))
        self.m13 = nn.Parameter(torch.tensor(0.0))
        self.m23 = nn.Parameter(torch.tensor(0.0))

    def hypergraph_contrastive_loss(self, z1, z2, temperature=0.7):
        """
        计算超图对比学习中的 InfoNCE 损失。

        参数:
        - z1, z2: 来自不同超图视图的嵌入表示，shape 均为 [n_samples, n_features]
        - temperature: 温度参数，调节相似度分布的平滑度

        返回:
        - loss: 标量损失值
        """
        # 对嵌入进行 L2 归一化，使得 cosine 相似度计算更稳定
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        # 计算相似度矩阵：两两样本间的点乘相似度除以 temperature
        sim_matrix = torch.mm(z1_norm, z2_norm.t()) / temperature  # [n_samples, n_samples]

        # 每个样本正样本对为其在另一视图中的对应样本，其索引相同
        target = torch.arange(z1.shape[0]).to(z1.device)

        # 分别从两个方向计算交叉熵损失：
        # 1. 以 z1 为 anchor，z2 为正样本和负样本
        loss_12 = F.cross_entropy(sim_matrix, target)
        # 2. 以 z2 为 anchor，z1 为正样本和负样本
        loss_21 = F.cross_entropy(sim_matrix.t(), target)

        # 最终损失为两个方向损失的平均
        loss = (loss_12 + loss_21) / 2.0
        return loss


    def forward(self, data):

        x_protein_knn_sim, hyperedge_protein_knn_sim_index = data["protein_dis_knn_from_sim_data"].x, data["protein_dis_knn_from_sim_data"].hyperedge_index

        x_meta_knn_sim, hyperedge_meta_knn_sim_index = data["meta_dis_knn_from_sim_data"].x, data["meta_dis_knn_from_sim_data"].hyperedge_index



        protein_conv_2 = self.protein_conv_1(x_protein_knn_sim, hyperedge_protein_knn_sim_index[0])
        protein_conv_3 = self.protein_conv_2(x_protein_knn_sim, hyperedge_protein_knn_sim_index[1])

        meta_conv_2 = self.meta_conv_1(x_meta_knn_sim, hyperedge_meta_knn_sim_index[0])
        meta_conv_3 = self.meta_conv_2(x_meta_knn_sim, hyperedge_meta_knn_sim_index[1])


        loss_p1 = self.hypergraph_contrastive_loss(protein_conv_2,protein_conv_3)
        loss_m2 = self.hypergraph_contrastive_loss(meta_conv_2,meta_conv_3)
        protein = torch.stack((protein_conv_3, protein_conv_2), dim=0)
        meta = torch.stack((meta_conv_3, meta_conv_2), dim=0)
        loss = torch.exp(-self.p12) * loss_p1 + self.p12 + torch.exp(-self.m13) * loss_m2 + self.m13

        return protein.unsqueeze(0),meta.unsqueeze(0),x_protein_knn_sim.unsqueeze(0).unsqueeze(0),x_meta_knn_sim.unsqueeze(0).unsqueeze(0),loss

class RowChannelAttentionWithReduction(nn.Module):
    def __init__(self, channels, reduction=16):

        super(RowChannelAttentionWithReduction, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(channels, 1)
        self.fc2 = nn.Linear(1, channels)
        self.conv_reduce = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        try:
            batch_size, channels, height, width = x.size()
        except:
            x = x.unsqueeze(0)
            batch_size, channels, height, width = x.size()

        x_reshaped = x.reshape(batch_size * height, channels, 1, width)

        avg_pooled = self.avg_pool(x_reshaped)  # 形状: (batch_size * height, channels, 1, 1)
        avg_pooled = avg_pooled.view(batch_size * height, channels)  # 形状: (batch_size * height, channels)

        attention = F.relu(self.fc1(avg_pooled))  # 第一层全连接 + ReLU 激活
        attention = torch.sigmoid(self.fc2(attention))  # 第二层全连接 + Sigmoid 激活

        attention = attention.view(batch_size * height, channels, 1, 1)

        x_attended = x_reshaped * attention

        x_attended = x_attended.view(batch_size, channels, height, width).squeeze(0).permute(1,0,2)

        x_reduced = self.conv_reduce(x_attended)  # 形状: (batch_size, 1, height, width)

        return x_reduced.permute(1,0,2).unsqueeze(0)

class Model(torch.nn.Module):
    def __init__(self, protein_dim, meta_dim):
        super(Model, self).__init__()
        self.HC = HGCN(protein_dim,meta_dim)
        self.att_protein = RowChannelAttentionWithReduction(2)
        self.att_meta = RowChannelAttentionWithReduction(2)

        self.encoder = nn.Sequential(
            nn.Linear((protein_dim+meta_dim), (protein_dim+meta_dim)//4),
            nn.BatchNorm1d((protein_dim+meta_dim)//4),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear((protein_dim+meta_dim)//4, (protein_dim+meta_dim)//8),
            nn.BatchNorm1d((protein_dim+meta_dim)//8),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear((protein_dim+meta_dim)//8, 1)
        )


    def forward(self, data, index):
        protein, meta, protein_o,meta_o,loss = self.HC(data)

        protein = self.att_protein(protein)
        meta = self.att_meta(meta)

        protein = protein+protein_o
        meta = meta+meta_o

        results_rows = []
        for i in range(len(index[0])):
            results_rows.append(torch.concat((protein[:,:,index[0][i],:],meta[:,:,index[1][i],:]),dim=2))
        x = torch.stack(results_rows, dim=2).to(device).squeeze()
        # 单条 MPI 时 squeeze 会去掉 batch 维，BatchNorm1d 需要 [N, C]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.encoder(x)

        return x, loss
