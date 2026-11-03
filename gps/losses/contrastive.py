# -*- coding: utf-8 -*-
"""
自适应对比学习损失

特点:
  - 可学习温度参数 τ (0.5 → 0.05)
  - Hard Negative Mining: 高相似度负样本自动加权
  - 多视图 InfoNCE
"""

import tensorflow as tf


class AdaptiveContrastiveLoss(tf.keras.layers.Layer):
    def __init__(self, init_temperature=0.5, min_temperature=0.05, **kwargs):
        super().__init__(**kwargs)
        self.temperature = self.add_weight(
            name='contrast_temperature',
            shape=(),
            initializer=tf.keras.initializers.Constant(init_temperature),
            trainable=True,
            constraint=lambda t: tf.clip_by_value(t, min_temperature, 1.0))
        self.min_temperature = min_temperature
        self.init_temperature = init_temperature

    def call(self, embeddings_list, training=True):
        n_views = len(embeddings_list)
        if n_views < 2:
            return tf.constant(0.0)

        # 展平可能的 3D 输入


        flat_embs = []
        for e in embeddings_list:
            if len(e.shape) > 2:
                e = tf.reduce_mean(e, axis=list(range(1, len(e.shape) - 1)))
            flat_embs.append(e)

        norms = [tf.math.l2_normalize(e, axis=-1) for e in flat_embs]
        total_loss = 0.0
        pair_count = 0

        for i in range(n_views):
            for j in range(i + 1, n_views):
                sim = tf.matmul(norms[i], norms[j], transpose_b=True)
                sim = sim / tf.cast(self.temperature, sim.dtype)
                batch_size = tf.shape(norms[i])[0]

                # Hard Negative Mining


                sim_dtype = sim.dtype
                pos_mask = tf.cast(tf.eye(batch_size), sim_dtype)
                neg_mask = 1.0 - pos_mask
                neg_sim = sim * neg_mask
                neg_weights = tf.nn.softmax(
                    tf.cast(neg_sim, tf.float32) / tf.cast(self.temperature, tf.float32), axis=-1)
                weighted_sim = sim + tf.cast(neg_weights, sim_dtype) * 0.1

                labels = tf.range(batch_size)
                loss_i = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=tf.cast(weighted_sim, tf.float32)))
                loss_j = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=tf.cast(tf.transpose(weighted_sim), tf.float32)))
                total_loss += loss_i + loss_j
                pair_count += 1

        return tf.cast(total_loss, tf.float32) / tf.cast(pair_count * 2, tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({'init_temperature': self.init_temperature,
                       'min_temperature': self.min_temperature})
        return config
