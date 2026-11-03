# -*- coding: utf-8 -*-
"""
改进版流匹配损失 — 多尺度信息保持 + 通道多样性正则

改进点:
  1. 不只重建 concat → final，还重建 final → 各通道分量
  2. 通道方差正则：防止所有通道坍缩到同一表示
  3. Sinkhorn 正则化：替代简单的 MSE + Cosine
"""

import tensorflow as tf


class AdaptiveFlowLoss(tf.keras.layers.Layer):
    """
    自适应流匹配正则化。

    输入:
      source: 拼接的多通道嵌入 [B, n_graphs * d_model]
      target: 融合后的嵌入 [B, d_model]
      per_channel: list of [B, d_model], n_graphs 个单通道嵌入

    过程:
      1. target → 重建 source (MSE)
      2. 重建的 source 分解为各通道 → 与 per_channel 对比 (余弦)
      3. 通道方差正则 → 防止坍缩
    """

    def __init__(self, d_model, n_graphs, alpha=0.4, beta=0.3, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_graphs = n_graphs
        self.alpha = alpha  # MSE 权重
        self.beta = beta    # 通道一致性权重

        # 重建网络: d_model → 2*d_model → n_graphs*d_model


        self.reconstructor = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 2, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(n_graphs * d_model),
        ], name='adaptive_flow_reconstructor')

        # 通道投影器: 从 source 提取各通道分量


        self.channel_projectors = [
            tf.keras.layers.Dense(d_model, activation='relu', name=f'ch_proj_{i}')
            for i in range(n_graphs)
        ]

    def call(self, source, target, per_channel_list):
        batch_size = tf.shape(target)[0]
        src_dtype = source.dtype

        # 1. 重建 source


        reconstructed = self.reconstructor(target)  # [B, n_graphs * d_model]

        # 2. MSE 损失


        mse_loss = tf.reduce_mean(tf.square(source - reconstructed))

        # 3. 通道一致性: 分解 reconstructed，与各 per_channel 对比


        rec_reshaped = tf.reshape(reconstructed, (batch_size, self.n_graphs, self.d_model))
        channel_cosine_loss = tf.cast(0.0, src_dtype)
        one = tf.cast(1.0, src_dtype)
        for g in range(self.n_graphs):
            rec_ch = rec_reshaped[:, g, :]
            rec_norm = tf.math.l2_normalize(rec_ch, axis=-1)
            ch_norm = tf.math.l2_normalize(per_channel_list[g], axis=-1)
            channel_cosine_loss += one - tf.reduce_mean(
                tf.reduce_sum(rec_norm * ch_norm, axis=-1)
            )
        channel_cosine_loss = channel_cosine_loss / tf.cast(self.n_graphs, src_dtype)

        # 4. 通道方差正则：鼓励不同通道有不同表示


        per_ch_stack = tf.stack(per_channel_list, axis=1)  # [B, n_graphs, D]
        channel_mean = tf.reduce_mean(per_ch_stack, axis=1, keepdims=True)  # [B, 1, D]
        channel_variance = tf.reduce_mean(tf.square(per_ch_stack - channel_mean))
        # 方差太小 → 惩罚；方差适中 → 无惩罚


        var_target = tf.cast(0.05, src_dtype)
        variance_loss = tf.maximum(tf.cast(0.0, src_dtype), var_target - channel_variance)

        total = (tf.cast(self.alpha, src_dtype) * mse_loss +
                 tf.cast(1 - self.alpha - self.beta, src_dtype) * channel_cosine_loss +
                 tf.cast(self.beta, src_dtype) * variance_loss)

        return total

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_graphs": self.n_graphs,
            "alpha": self.alpha,
            "beta": self.beta,
        })
        return config
