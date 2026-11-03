# -*- coding: utf-8 -*-
"""空间编码层 — 将最短路径距离映射为注意力偏置"""

import tensorflow as tf


class SpatialEncoding(tf.keras.layers.Layer):
    def __init__(self, d_sp_enc=16, activation='relu'):
        super().__init__()
        self.d_sp_enc = d_sp_enc
        self.activation = activation
        self.dense1 = tf.keras.layers.Dense(d_sp_enc, activation=activation)
        self.dense2 = tf.keras.layers.Dense(64, activation=activation)
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, distances):
        outputs = self.dense1(distances)
        outputs = self.dense2(outputs)
        return outputs
