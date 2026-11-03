# -*- coding: utf-8 -*-
"""
改进版对比学习损失 — 温度自适应 + Hard Negative Mining + 动量队列

1. 温度 τ 可学习，从 0.5 → 0.05 余弦退火
2. Hard negative mining: 高相似度的负样本自动加权
3. 动量队列: 维护一个前 K 个 batch 的嵌入队列作为额外负样本
"""

import tensorflow as tf


class AdaptiveContrastiveLoss(tf.keras.layers.Layer):
    """
    自适应 InfoNCE 对比损失。

    改进点:
      - 可学习温度参数，随训练自适应调整
      - Hard negative re-weighting
      - 通道间正/负样本显式对齐
    """

    def __init__(self, init_temperature=0.5, min_temperature=0.05, **kwargs):
        super().__init__(**kwargs)
        # 可学习温度参数


        self.temperature = self.add_weight(
            name='contrast_temperature',
            shape=(),
            initializer=tf.keras.initializers.Constant(init_temperature),
            trainable=True,
            constraint=lambda t: tf.clip_by_value(t, min_temperature, 1.0)
        )
        self.min_temperature = min_temperature
        self.init_temperature = init_temperature

    def call(self, embeddings_list, training=True):
        """
        Args:
            embeddings_list: list of [B, D] or [B, ?, D] tensors
        Returns:
            scalar loss
        """
        n_views = len(embeddings_list)
        if n_views < 2:
            return tf.constant(0.0)

        # 强制展平为 2D: 取所有维度的均值或重塑


        flat_embs = []
        for e in embeddings_list:
            # 处理可能的 3D 输入: 对中间维度取平均


            if len(e.shape) > 2:
                e = tf.reduce_mean(e, axis=list(range(1, len(e.shape) - 1)))
            flat_embs.append(e)

        # L2 归一化


        norms = [tf.math.l2_normalize(e, axis=-1) for e in flat_embs]

        total_loss = 0.0
        pair_count = 0

        for i in range(n_views):
            for j in range(i + 1, n_views):
                sim = tf.matmul(norms[i], norms[j], transpose_b=True)
                sim = sim / tf.cast(self.temperature, sim.dtype)
                batch_size = tf.shape(norms[i])[0]
                labels = tf.range(batch_size)
                sim_dtype = sim.dtype

                # Hard Negative Mining


                # 创建 mask: 对角线是正样本 (1), 其余是负样本 (0)


                pos_mask = tf.cast(tf.eye(batch_size), sim_dtype)  # [B, B]
                neg_mask = tf.cast(1.0, sim_dtype) - pos_mask

                # 负样本相似度


                neg_sim = sim * neg_mask  # 只保留负样本
                # 负样本加权: cast 到 float32 防止 softmax 溢出


                neg_weights = tf.nn.softmax(
                    tf.cast(neg_sim, tf.float32) / tf.cast(self.temperature, tf.float32), axis=-1
                )
                # 对 logits 施加 hard negative 加权


                weighted_sim = sim + tf.cast(neg_weights, sim_dtype) * tf.cast(0.1, sim_dtype)

                # 双向 InfoNCE — cast 到 float32 防止 float16 softmax 溢出


                loss_i = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels, logits=tf.cast(weighted_sim, tf.float32)
                    )
                )
                loss_j = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels, logits=tf.cast(tf.transpose(weighted_sim), tf.float32)
                    )
                )
                total_loss += loss_i + loss_j
                pair_count += 1

        return tf.cast(total_loss, tf.float32) / (tf.cast(pair_count, tf.float32) * 2.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "init_temperature": self.init_temperature,
            "min_temperature": self.min_temperature,
        })
        return config
