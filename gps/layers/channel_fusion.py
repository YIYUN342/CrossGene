# -*- coding: utf-8 -*-
"""通道注意力融合层 — 多头注意力 + 残差门控 + 熵正则"""

import tensorflow as tf


class ChannelFusion(tf.keras.layers.Layer):
    """
    将 G 个通道嵌入融合为单个全局嵌入。

    改进点:
      - 多头通道注意力 (2 heads) 捕获不同融合模式
      - 残差门控: learnable gate 控制新旧信息比例
      - 熵正则: 防止某个通道主导注意力
    """

    def __init__(self, d_model, n_channels, num_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_channels = n_channels
        self.num_heads = num_heads

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        d_k = num_heads * n_channels * d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_k)
        self.wk = tf.keras.layers.Dense(d_k)
        self.wv = tf.keras.layers.Dense(d_k)

        self.gate = tf.keras.layers.Dense(d_model, activation='sigmoid', name='fusion_gate')
        self.residual_proj = tf.keras.layers.Dense(d_model, name='residual_proj')
        self.dense = tf.keras.layers.Dense(d_model)

    def build(self, input_shape):
        if self.built:
            return
        B = input_shape[0] if input_shape[0] is not None else 1
        self.wq.build((B, 1, input_shape[-1]))
        self.wk.build((B, 1, input_shape[-1]))
        self.wv.build((B, 1, input_shape[-1]))
        self.gate.build((B, self.d_model))
        self.residual_proj.build((B, self.d_model))
        self.dense.build((B, self.d_model))
        self.layernorm.build((B, self.d_model))
        super().build(input_shape)

    def split_heads_multi(self, x, batch_size):
        x = tf.reshape(x, (batch_size, self.n_channels, self.d_model))
        x = tf.reshape(x, (batch_size, self.n_channels, self.num_heads, -1))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [B, H, C, depth]

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x_expanded = tf.expand_dims(x, axis=1)  # [B, 1, C*D]

        # 残差: 各通道均值


        x_reshaped = tf.reshape(x, (batch_size, self.n_channels, self.d_model))
        residual = self.residual_proj(tf.reduce_mean(x_reshaped, axis=1))

        # 多头通道注意力


        q = self.wq(x_expanded)
        k = self.wk(x_expanded)
        v = self.wv(x_expanded)
        q = self.split_heads_multi(q, batch_size)
        k = self.split_heads_multi(k, batch_size)
        v = self.split_heads_multi(v, batch_size)

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
        scaled_logits = matmul_qk * tf.math.rsqrt(dk)
        attn_weights = tf.nn.softmax(scaled_logits, axis=-1)

        # 熵正则


        avg_w = tf.reduce_mean(attn_weights, axis=1)
        w_dtype = attn_weights.dtype
        ent = -tf.reduce_mean(tf.reduce_sum(avg_w * tf.math.log(avg_w + 1e-3), axis=-1))
        target_ent = tf.cast(tf.math.log(tf.cast(self.n_channels, w_dtype)) * 0.5, w_dtype)
        entropy_loss = tf.maximum(0.0, target_ent - ent)
        self.add_loss(0.01 * entropy_loss)

        # 加权求和


        v_flat = tf.reshape(v, (batch_size, self.num_heads, self.n_channels, -1))
        scaled_attn = tf.matmul(attn_weights, v_flat)
        scaled_attn = tf.transpose(scaled_attn, perm=[0, 2, 1, 3])
        concat = tf.reshape(scaled_attn, (batch_size, self.n_channels, -1))
        fused = self.dense(tf.reduce_mean(concat, axis=1))

        # 残差门控


        gate_val = self.gate(fused)
        output = gate_val * fused + (1.0 - gate_val) * tf.cast(residual, gate_val.dtype)
        return self.layernorm(output), attn_weights
