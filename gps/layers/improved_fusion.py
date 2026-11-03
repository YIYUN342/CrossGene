# -*- coding: utf-8 -*-
"""
改进版注意力融合 — 通道多样性正则 + 残差门控

改进点:
  1. 通道注意力权重 L2 正则: 防止单一通道主导
  2. 残差门控: 学习保留多少原始信息
  3. 多头通道注意力: 多个注意力头捕获不同融合模式
"""

import tensorflow as tf


class ImprovedAttentionFusion(tf.keras.layers.Layer):
    """改进的多通道注意力融合层。"""

    def __init__(self, d_model, n_channels, num_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_channels = n_channels
        self.num_heads = num_heads

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 多头通道注意力


        self.wq = tf.keras.layers.Dense(num_heads * n_channels * d_model // num_heads)
        self.wk = tf.keras.layers.Dense(num_heads * n_channels * d_model // num_heads)
        self.wv = tf.keras.layers.Dense(num_heads * n_channels * d_model // num_heads)

        # 残差门控


        self.gate = tf.keras.layers.Dense(d_model, activation='sigmoid', name='fusion_gate')
        self.residual_proj = tf.keras.layers.Dense(d_model, name='residual_proj')

        self.dense = tf.keras.layers.Dense(d_model)

    def build(self, input_shape):
        if self.built:
            return
        B = input_shape[0] if input_shape[0] is not None else 1
        expanded_shape = (B, 1, input_shape[-1])
        proj_shape = (B, self.d_model)
        self.wq.build(expanded_shape)
        self.wk.build(expanded_shape)
        self.wv.build(expanded_shape)
        self.gate.build(proj_shape)
        self.residual_proj.build(proj_shape)
        self.dense.build(proj_shape)
        self.layernorm.build(proj_shape)
        super().build(input_shape)

    def split_heads_multi(self, x, batch_size):
        # x: [B, 1, n_channels * d_model]


        x = tf.reshape(x, (batch_size, self.n_channels, self.d_model))
        # 多头上采样: [B, n_channels, D] → [B, n_heads, n_channels, D//n_heads]


        x = tf.reshape(x, (batch_size, self.n_channels, self.num_heads, -1))
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [B, n_heads, n_channels, D//n_heads]
        return x

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x_expanded = tf.expand_dims(x, axis=1)  # [B, 1, n_channels * D]

        # 残差连接: 用平均池化作为残差


        x_reshaped = tf.reshape(x, (batch_size, self.n_channels, self.d_model))
        residual = self.residual_proj(tf.reduce_mean(x_reshaped, axis=1))

        # 多头通道注意力


        q = self.wq(x_expanded)
        k = self.wk(x_expanded)
        v = self.wv(x_expanded)

        q = self.split_heads_multi(q, batch_size)  # [B, n_heads, n_channels, depth]
        k = self.split_heads_multi(k, batch_size)
        v = self.split_heads_multi(v, batch_size)

        # 注意力计算


        matmul_qk = tf.matmul(q, k, transpose_b=True)  # [B, n_heads, n_channels, n_channels]
        dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
        scaled_logits = matmul_qk * tf.math.rsqrt(dk)

        attention_weights = tf.nn.softmax(scaled_logits, axis=-1)  # [B, n_heads, n_channels, n_channels]

        # 通道注意力正则: 防止集中到单一通道


        avg_weights = tf.reduce_mean(attention_weights, axis=1)  # [B, n_channels, n_channels]
        w_dtype = attention_weights.dtype
        # 熵正则: 如果所有注意力在一个通道 → 熵低 → 惩罚


        ent_reg = -tf.reduce_mean(
            tf.reduce_sum(avg_weights * tf.math.log(avg_weights + tf.cast(1e-3, w_dtype)), axis=-1)
        )
        # 期望均匀分布 → 高熵; 实际太低 → 惩罚


        target_ent = tf.cast(tf.math.log(tf.cast(self.n_channels, w_dtype)) * 0.5, w_dtype)
        entropy_loss = tf.maximum(tf.cast(0.0, w_dtype), target_ent - ent_reg)
        self.add_loss(tf.cast(0.01, w_dtype) * entropy_loss)

        # 加权求和


        v_flat = tf.reshape(v, (batch_size, self.num_heads, self.n_channels, -1))
        scaled_attention = tf.matmul(attention_weights, v_flat)  # [B, n_heads, n_channels, depth]
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, self.n_channels, -1))
        fused = self.dense(tf.reduce_mean(concat_attention, axis=1))

        # 残差门控 (混合精度兼容: 确保 gate 和 residual dtype 一致)


        gate_val = self.gate(fused)
        one = tf.constant(1.0, dtype=gate_val.dtype)
        output = gate_val * fused + (one - gate_val) * tf.cast(residual, gate_val.dtype)

        return self.layernorm(output), attention_weights
