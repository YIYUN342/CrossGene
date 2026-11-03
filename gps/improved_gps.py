"""
GPS 核心模块的 PyTorch 实现
GatedGCN + DualBranchAttention + LayerScale + Stochastic Depth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralityEncoding(nn.Module):
    def __init__(self, max_degree, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_degree, d_model)

    def forward(self, distances):
        centrality = (distances.abs() == 1).float().sum(-1)
        centrality = centrality.long().clamp(0, self.embedding.num_embeddings - 1)
        return self.embedding(centrality)


class LayerScale(nn.Module):
    def __init__(self, d_model, init_value=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model) * init_value)

    def forward(self, x):
        return x * self.scale


class GatedGCN(nn.Module):
    """边门控图卷积 — GPS 的核心局部消息传递层"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.edge_gate = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.edge_msg = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.res_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_attr):
        B, N, D = x.shape
        # edge_attr: [B, N, N, 1]


        xi = x.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        xj = x.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        edge_repr = torch.cat([xi, edge_attr, xj], dim=-1)  # [B, N, N, 2D+1]

        gate = self.edge_gate(edge_repr)        # [B, N, N, 1]
        message = self.edge_msg(edge_repr)       # [B, N, N, D]
        message = gate * message

        out = message.sum(dim=2)                 # [B, N, D]
        out = out + self.res_proj(x)
        return self.dropout(out)


class SpatialBiasEncoding(nn.Module):
    def __init__(self, d_sp_enc=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_sp_enc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_sp_enc, num_heads),
        )

    def forward(self, distances):
        dist = distances.float().clamp(-1, 50)
        bias = self.net(dist)
        return bias.permute(0, 3, 1, 2)


class DualBranchAttention(nn.Module):
    """特征自注意力 + 空间交叉注意力 → 门控融合 (from v3)"""
    def __init__(self, d_model, num_heads, d_sp_enc=128, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Branch 1: Feature Self-Attention


        self.wq1 = nn.Linear(d_model, d_model)
        self.wk1 = nn.Linear(d_model, d_model)
        self.wv1 = nn.Linear(d_model, d_model)
        self.spatial_bias = SpatialBiasEncoding(d_sp_enc, num_heads, dropout)

        # Branch 2: Spatial Cross-Attention


        self.wq2 = nn.Linear(d_model, d_model)
        self.spatial_enc = nn.Sequential(
            nn.Linear(1, d_sp_enc), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_sp_enc, d_model),
        )
        self.wk2 = nn.Linear(d_model, d_model)
        self.wv2 = nn.Linear(d_model, d_model)

        # Gate


        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, B):
        return x.view(B, -1, self.num_heads, self.depth).transpose(1, 2)

    def forward(self, x, min_distance_matrix, mask=None):
        B, N, _ = x.shape
        scale = self.depth ** -0.5

        # Branch 1: Feature Self-Attention + spatial bias


        q1, k1, v1 = self.wq1(x), self.wk1(x), self.wv1(x)
        q1, k1, v1 = [self.split_heads(t, B) for t in (q1, k1, v1)]
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        attn1 = attn1 + self.spatial_bias(min_distance_matrix)
        if mask is not None: attn1 = attn1.masked_fill(mask.unsqueeze(1), -1e9)
        attn1_w = F.softmax(attn1, dim=-1); attn1_w = self.dropout(attn1_w)
        feat_out = torch.matmul(attn1_w, v1)
        feat_out = feat_out.transpose(1, 2).contiguous().view(B, N, self.d_model)

        # Branch 2: Spatial Cross-Attention


        q2 = self.wq2(x)
        spat_feat = self.spatial_enc(min_distance_matrix)
        k2 = self.wk2(spat_feat.mean(dim=2)); v2 = self.wv2(spat_feat.mean(dim=2))
        q2, k2, v2 = [self.split_heads(t, B) for t in (q2, k2, v2)]
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale
        if mask is not None: attn2 = attn2.masked_fill(mask.unsqueeze(1), -1e9)
        attn2_w = F.softmax(attn2, dim=-1); attn2_w = self.dropout(attn2_w)
        spat_out = torch.matmul(attn2_w, v2)
        spat_out = spat_out.transpose(1, 2).contiguous().view(B, N, self.d_model)

        # Gate fusion


        gate = self.gate(torch.cat([feat_out, spat_out], dim=-1))
        fused = gate * feat_out + (1 - gate) * spat_out
        return self.out_proj(fused), (attn1_w, attn2_w, gate)


class GPSBlock(nn.Module):
    """GPS Block: GatedGCN(局部) + DualBranchAttention(全局) + FFN"""
    def __init__(self, d_model, num_heads, dff, d_sp_enc=128, dropout=0.1, layer_idx=0, drop_path=0.0):
        super().__init__()
        self.gcn = GatedGCN(d_model, dropout)
        self.attn = DualBranchAttention(d_model, num_heads, d_sp_enc, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ls1 = LayerScale(d_model)
        self.ls2 = LayerScale(d_model)
        self.ls3 = LayerScale(d_model)
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        self.do3 = nn.Dropout(dropout)
        self.drop_path = drop_path

    def forward(self, x, min_distance_matrix, mask=None, training=True):
        # Edge attr: normalize distances to [0,1]


        edge_attr = min_distance_matrix.float().clamp(-1, 20) / 20.0
        edge_attr = torch.where(min_distance_matrix < 0, torch.ones_like(edge_attr) * 0.5, edge_attr)

        # 1. GatedGCN (局部消息传递)


        residual = x
        x = residual + self.do1(self.ls1(self.gcn(self.norm1(x), edge_attr)))

        # 2. DualBranchAttention (全局)


        residual = x
        x = residual + self.do2(self.ls2(self.attn(self.norm2(x), min_distance_matrix, mask)[0]))

        # 3. FFN


        residual = x
        x = residual + self.do3(self.ls3(self.ffn(self.norm3(x))))
        return x
