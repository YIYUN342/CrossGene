# -*- coding: utf-8 -*-
"""多头自注意力 + scaled_dot_product_attention（含空间偏置）"""

import tensorflow as tf
from .spatial import SpatialEncoding


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_sp_enc, sp_enc_activation):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_sp_enc = d_sp_enc
        self.spatial_encoding = SpatialEncoding(d_sp_enc, sp_enc_activation)
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, min_distance_matrix, mask):
        batch_size = tf.shape(q)[0]
        spatial_bias = self.spatial_encoding(min_distance_matrix)

        q = self.wq(q)
        k = self.wk(spatial_bias)
        v = self.wv(spatial_bias)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, self.num_heads, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


def scaled_dot_product_attention(q, k, v, num_heads, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
    scaled_attention_logits = matmul_qk * tf.math.rsqrt(dk)

    if mask is not None:
        mask_tiled = tf.tile(mask, [1, num_heads, 1, 1])
        mask_value = tf.cast(-10000.0, scaled_attention_logits.dtype)
        scaled_attention_logits += (tf.cast(mask_tiled, scaled_attention_logits.dtype) * mask_value)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=3)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights
