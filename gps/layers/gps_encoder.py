# -*- coding: utf-8 -*-
"""
GPS (General, Powerful, Scalable) Graph Encoder Block

每个 GPS Block 包含三个处理分支:
  1. GatedGCNv2 (局部消息传递): 使用空间距离作为边特征门控
  2. Multi-Head Self-Attention (全局交互): 捕获子图内长程依赖
  3. FFN + LayerScale + Stochastic Depth

参考文献:
  - Rampasek et al., "Recipe for a General, Powerful, Scalable Graph Transformer", NeurIPS 2022
  - Bresson & Laurent, "Residual Gated Graph ConvNets", 2018
"""

import tensorflow as tf


# GatedGCNv2 — 边特征门控图卷积


class GatedGCNLayer(tf.keras.layers.Layer):
    """
    边特征门控图卷积层 (GatedGCNv2).

    输入: 节点特征 x [B, N, D], 边特征 e [B, N, N, De]
    过程:
      - 对每条边 (i,j) 计算门控值: gate = sigmoid(W_g * [x_i || e_ij || x_j])
      - 消息传递: message = gate * ReLU(W_m * [x_i || e_ij || x_j])
      - 聚合: x_i' = sum_j message_ij + W_r * x_i (残差)
    输出: [B, N, D]
    """

    def __init__(self, d_model, d_edge=1, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_edge = d_edge
        self.dropout_rate = dropout

    def build(self, input_shape):
        d = self.d_model
        de = self.d_edge
        concat_dim = d * 2 + de

        # Edge gate network


        self.edge_gate_net = tf.keras.Sequential([
            tf.keras.layers.Dense(d, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ], name='edge_gate_net')

        # Message network


        self.edge_message_net = tf.keras.Sequential([
            tf.keras.layers.Dense(d * 2, activation='relu'),
            tf.keras.layers.Dense(d, activation='relu'),
        ], name='edge_message_net')

        # Residual projection


        self.residual_proj = tf.keras.layers.Dense(d, name='gcn_residual')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.built = True

    def call(self, x, edge_attr, training=False):
        """
        Args:
            x: 节点特征 [B, N, D]
            edge_attr: 边特征 [B, N, N, De] (如空间距离)
        Returns:
            更新后的节点特征 [B, N, D]
        """
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        D = self.d_model

        # 扩展节点特征以构造边表示


        xi = tf.tile(x[:, :, tf.newaxis, :], [1, 1, N, 1])  # [B, N, N, D]
        xj = tf.tile(x[:, tf.newaxis, :, :], [1, N, 1, 1])  # [B, N, N, D]

        # 边表示 = [x_i || edge_attr || x_j]


        edge_repr = tf.concat([xi, edge_attr, xj], axis=-1)  # [B, N, N, D*2+De]

        # 门控值: e_ij ∈ [0, 1]


        gate = self.edge_gate_net(edge_repr)  # [B, N, N, 1]

        # 消息


        message = self.edge_message_net(edge_repr)  # [B, N, N, D]
        message = gate * message  # 门控消息

        # 聚合: sum over neighbors


        out = tf.reduce_sum(message, axis=2)  # [B, N, D]

        # 残差连接


        residual = self.residual_proj(x)
        out = out + residual
        out = self.dropout(out, training=training)

        return out


# GPS Block — 局部 GatedGCN + 全局 Transformer + FFN


class GPSBlock(tf.keras.layers.Layer):
    """
    GPS 编码器块: 组合局部图卷积和全局注意力.

    结构:
      x = x + GatedGCN(LN1(x), edge_attr)     ← 局部消息传递
      x = x + MHA(LN2(x), spatial_bias)        ← 全局自注意力
      x = x + FFN(LN3(x))                      ← 前馈网络
    每步均有 LayerScale 和可选的 Stochastic Depth.
    """

    def __init__(self, d_model, num_heads, dff, dropout=0.1,
                 d_sp_enc=128, sp_enc_activation='relu',
                 drop_path_rate=0.0, max_distance=4.0, mpnn_type='gatedgcn', **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
        self.drop_path_rate = drop_path_rate
        self.max_distance = max_distance

        # 可学习边温度 (closer nodes = stronger edges)


        self.edge_temperature = self.add_weight(
            name='edge_temp', shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            constraint=lambda t: tf.clip_by_value(t, 0.1, 10.0)
        )

        # 局部: GatedGCN


        self.gated_gcn = GatedGCNLayer(
            d_model=d_model, d_edge=1, dropout=dropout, name=f'{kwargs.get("name", "gps")}_gcn'
        )

        # 全局: Multi-Head Self-Attention with spatial bias


        from .multiHeadAttention import MultiHeadAttention
        self.mha = MultiHeadAttention(d_model, num_heads, d_sp_enc, sp_enc_activation)

        # FFN


        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
        ], name='gps_ffn')

        # Layer Normalization


        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout


        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

        # LayerScale (可学习缩放)


        self.ls1 = self.add_weight(
            name='ls_gcn', shape=(d_model,),
            initializer=tf.keras.initializers.Constant(0.1), trainable=True
        )
        self.ls2 = self.add_weight(
            name='ls_attn', shape=(d_model,),
            initializer=tf.keras.initializers.Constant(0.1), trainable=True
        )
        self.ls3 = self.add_weight(
            name='ls_ffn', shape=(d_model,),
            initializer=tf.keras.initializers.Constant(0.1), trainable=True
        )

    def call(self, x, training=False, mask=None, min_distance_matrix=None):
        """
        Args:
            x: 节点特征 [B, N, D]
            mask: 注意力 mask [B, 1, N, N]
            min_distance_matrix: 空间距离矩阵 [B, N, N], 用作 GCN 边特征
        Returns:
            更新后的节点特征 [B, N, D]
        """
        # 预处理边特征用于 GatedGCN: [B, N, N] -> [B, N, N, 1]


        if min_distance_matrix is not None:
            edge_attr = tf.expand_dims(
                tf.cast(min_distance_matrix, x.dtype), axis=-1
            )
            # 反向距离编码: 越近的节点边权重越大, padding -> 0


            max_d = tf.cast(self.max_distance, edge_attr.dtype)
            edge_attr = tf.where(
                edge_attr < 0,
                tf.zeros_like(edge_attr),                          # padding -> 0
                1.0 - tf.minimum(edge_attr, max_d) / max_d         # 动态归一化: 最近→1, 最远→0
            )
            # 可学习温度缩放


            temp = tf.cast(self.edge_temperature, edge_attr.dtype)
            edge_attr = edge_attr * temp
            # 用 mask 显式清零 padding 边


            if mask is not None:
                edge_mask = 1.0 - tf.squeeze(mask, axis=1)  # 1=valid, 0=padding
                edge_attr = edge_attr * tf.expand_dims(edge_mask, axis=-1)
        else:
            # 无边特征: 使用全 1 矩阵


            B = tf.shape(x)[0]
            N = tf.shape(x)[1]
            edge_attr = tf.ones((B, N, N, 1), dtype=x.dtype)

        # 1. 局部: GatedGCN


        residual = x
        x_norm = self.ln1(x)
        x_gcn = self.gated_gcn(x_norm, edge_attr, training=training)
        x_gcn = self.dropout1(x_gcn, training=training)
        x = residual + x_gcn * tf.cast(self.ls1, x_gcn.dtype)

        if training and self.drop_path_rate > 0:
            x = stochastic_depth(x, self.drop_path_rate)

        # 2. 全局: Multi-Head Self-Attention


        residual = x
        x_norm = self.ln2(x)
        attn_out, _ = self.mha(x_norm, min_distance_matrix, mask)
        attn_out = self.dropout2(attn_out, training=training)
        x = residual + attn_out * tf.cast(self.ls2, attn_out.dtype)

        if training and self.drop_path_rate > 0:
            x = stochastic_depth(x, self.drop_path_rate)

        # 3. FFN


        residual = x
        x_norm = self.ln3(x)
        ffn_out = self.ffn(x_norm)
        ffn_out = self.dropout3(ffn_out, training=training)
        x = residual + ffn_out * tf.cast(self.ls3, ffn_out.dtype)

        if training and self.drop_path_rate > 0:
            x = stochastic_depth(x, self.drop_path_rate)

        return x


# 多尺度融合 (与 improved_graphormer 相同)


class MultiScaleFusion(tf.keras.layers.Layer):
    """
    多尺度融合层: 收集所有层的输出，可学习加权融合.
    """

    def __init__(self, n_layers, d_model, **kwargs):
        super().__init__(**kwargs)
        self.layer_weights = self.add_weight(
            name='ms_fusion_weights',
            shape=(n_layers,),
            initializer=tf.keras.initializers.Constant(1.0 / n_layers),
            trainable=True,
            constraint=lambda w: tf.nn.softmax(w, axis=0)
        )

    def call(self, layer_outputs):
        stacked = tf.stack(layer_outputs, axis=0)  # [L, *input_dims]
        ndim = len(stacked.shape) - 1  # exclude layer dim
        reshape_target = (-1,) + (1,) * ndim  # dynamic broadcast shape
        weights = tf.cast(
            tf.reshape(self.layer_weights, reshape_target), stacked.dtype
        )
        weighted = stacked * weights
        return tf.reduce_sum(weighted, axis=0)


# Stochastic Depth


def stochastic_depth(x, drop_prob):
    """随机深度 (DropPath)."""
    if drop_prob <= 0.0:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    return (x / keep_prob) * binary_tensor
