# -*- coding: utf-8 -*-
"""
自适应流匹配正则化损失

三部分:
  1. 从融合嵌入重建拼接嵌入 (MSE)
  2. 通道级余弦一致性
  3. 通道方差正则 (防坍缩)
"""

import tensorflow as tf


class AdaptiveFlowLoss(tf.keras.layers.Layer):
    def __init__(self, d_model, n_graphs, alpha=0.4, beta=0.3, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_graphs = n_graphs
        self.alpha = alpha
        self.beta = beta

        self.reconstructor = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 2, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(n_graphs * d_model),
        ], name='flow_reconstructor')

        self.channel_projectors = [
            tf.keras.layers.Dense(d_model, activation='relu', name=f'ch_proj_{i}')
            for i in range(n_graphs)
        ]

    def call(self, source, target, per_channel_list):
        batch_size = tf.shape(target)[0]
        dtype = source.dtype

        # 1. 重建


        reconstructed = self.reconstructor(target)
        mse_loss = tf.reduce_mean(tf.square(source - reconstructed))

        # 2. 通道一致性


        rec_reshaped = tf.reshape(reconstructed, (batch_size, self.n_graphs, self.d_model))
        channel_cosine = tf.cast(0.0, dtype)
        for g in range(self.n_graphs):
            rc = tf.math.l2_normalize(rec_reshaped[:, g, :], axis=-1)
            pc = tf.math.l2_normalize(per_channel_list[g], axis=-1)
            channel_cosine += 1.0 - tf.reduce_mean(tf.reduce_sum(rc * pc, axis=-1))
        channel_cosine /= tf.cast(self.n_graphs, dtype)

        # 3. 通道方差正则


        per_ch = tf.stack(per_channel_list, axis=1)
        ch_mean = tf.reduce_mean(per_ch, axis=1, keepdims=True)
        ch_var = tf.reduce_mean(tf.square(per_ch - ch_mean))
        var_target = tf.cast(0.05, dtype)
        variance_loss = tf.maximum(0.0, var_target - ch_var)

        total = (tf.cast(self.alpha, dtype) * mse_loss +
                 tf.cast(1 - self.alpha - self.beta, dtype) * channel_cosine +
                 tf.cast(self.beta, dtype) * variance_loss)
        return total

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'n_graphs': self.n_graphs,
                       'alpha': self.alpha, 'beta': self.beta})
        return config
