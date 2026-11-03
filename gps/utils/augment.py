# -*- coding: utf-8 -*-
"""图数据增强 — 特征噪声 + 空间掩码"""

import tensorflow as tf


def edge_dropout_feature(feature, drop_rate=0.1, training=True):
    """节点特征加高斯噪声 + Feature Dropout"""
    if not training or drop_rate <= 0:
        return feature
    dtype = feature.dtype
    noise = tf.cast(tf.random.normal(tf.shape(feature), mean=0.0, stddev=drop_rate * 0.1), dtype)
    feature = feature + noise
    mask = tf.cast(tf.random.uniform(tf.shape(feature), dtype=dtype) > drop_rate, dtype)
    feature = feature * mask / tf.cast(1.0 - drop_rate + 1e-8, dtype)
    return feature


def spatial_dropout(spatial_matrix, drop_rate=0.15, training=True):
    """随机将部分空间距离设为不可达 (-1)"""
    if not training or drop_rate <= 0:
        return spatial_matrix
    dtype = spatial_matrix.dtype
    mask = tf.cast(tf.random.uniform(tf.shape(spatial_matrix), dtype=dtype) > drop_rate, dtype)
    augmented = tf.where(mask > 0.5, spatial_matrix, tf.ones_like(spatial_matrix) * tf.cast(-1.0, dtype))
    return augmented
