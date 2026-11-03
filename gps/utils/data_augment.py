# -*- coding: utf-8 -*-
"""
图数据增强模块 — 提升模型鲁棒性

1. 边随机丢弃 (Edge Dropout): 训练时随机屏蔽部分邻接边
2. 特征噪声 (Feature Noise): 训练时对节点特征加高斯噪声
3. 子图扰动: 对 spatial_matrix 进行随机 dropout
"""

import tensorflow as tf


def edge_dropout_feature(feature, drop_rate=0.1, training=True):
    """
    对节点特征加噪声 + 随机屏蔽特征维度。

    Args:
        feature: [B, n_neighbors, feature_dim]
        drop_rate: 噪声/屏蔽比例
        training: 是否训练模式
    Returns:
        augmented feature (同 shape)
    """
    if not training or drop_rate <= 0:
        return feature

    feat_dtype = feature.dtype

    # 1. 高斯噪声 (匹配输入 dtype)


    noise = tf.cast(
        tf.random.normal(tf.shape(feature), mean=0.0, stddev=drop_rate * 0.1),
        feat_dtype
    )
    feature = feature + noise

    # 2. Feature dropout: 随机屏蔽特征维度


    mask = tf.cast(
        tf.random.uniform(tf.shape(feature), dtype=feat_dtype) > drop_rate,
        feat_dtype
    )
    # 保持非零均值: 用 mask 缩放


    feature = feature * mask / tf.cast(1.0 - drop_rate + 1e-8, feat_dtype)

    return feature


def spatial_dropout(spatial_matrix, drop_rate=0.15, training=True):
    """
    对空间距离矩阵进行随机扰动。

    Args:
        spatial_matrix: [B, multi_hop, n_neighbors, n_neighbors] or [B, n_neighbors, n_neighbors]
        drop_rate: 扰动比例
        training: 是否训练模式
    Returns:
        augmented spatial matrix
    """
    if not training or drop_rate <= 0:
        return spatial_matrix

    sp_dtype = spatial_matrix.dtype

    # 随机将部分距离设为 -1 (不可达)


    mask = tf.cast(
        tf.random.uniform(tf.shape(spatial_matrix), dtype=sp_dtype) > drop_rate,
        sp_dtype
    )
    # 被 mask 掉的位置设为 -1 (padding)


    augmented = tf.where(mask > 0.5, spatial_matrix, tf.ones_like(spatial_matrix) * tf.cast(-1.0, sp_dtype))
    return augmented


def neighbor_dropout(node_neighbor, n_nodes, drop_rate=0.1, training=True):
    """
    随机替换部分邻居节点 (类似边缘扰动)。

    Args:
        node_neighbor: 邻居索引矩阵
        n_nodes: 总节点数
        drop_rate: 替换比例
        training: 是否训练模式
    Returns:
        augmented neighbor indices
    """
    if not training or drop_rate <= 0:
        return node_neighbor

    shape = tf.shape(node_neighbor)
    # 随机 mask


    mask = tf.cast(
        tf.random.uniform(shape) > drop_rate,
        node_neighbor.dtype
    )
    # 随机新邻居


    random_neighbors = tf.random.uniform(shape, minval=0, maxval=n_nodes, dtype=node_neighbor.dtype)
    # 混合


    augmented = node_neighbor * mask + random_neighbors * (1 - mask)
    return augmented
