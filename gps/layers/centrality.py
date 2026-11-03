# -*- coding: utf-8 -*-
"""中心性编码层 — 基于节点度数的位置编码"""

import tensorflow as tf


class CentralityEncoding(tf.keras.layers.Layer):
    def __init__(self, max_degree, d_model, **kwargs):
        super().__init__(**kwargs)
        self.centr_embedding = tf.keras.layers.Embedding(
            input_dim=max_degree, output_dim=d_model, name='centr_embedding')

    def centrality(self, distances):
        centrality = tf.cast(tf.math.equal(tf.math.abs(distances), 1), tf.float32)
        centrality = tf.math.reduce_sum(centrality, axis=-1, keepdims=False)
        return tf.cast(centrality, tf.int32)

    def call(self, distances):
        centrality = self.centrality(distances)
        return self.centr_embedding(centrality)
