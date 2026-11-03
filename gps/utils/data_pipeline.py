# -*- coding: utf-8 -*-
"""
GPU 优化数据流水线 — tf.data.Dataset 构建

关键优化:
  - cache(): 将数据缓存到内存，避免重复 I/O
  - shuffle(): 打乱训练数据
  - batch(): 批量加载
  - prefetch(tf.data.AUTOTUNE): GPU 计算时 CPU 异步预取下一批
"""

import tensorflow as tf


def create_gpu_dataset(node_ids, labels, batch_size, shuffle=True, cache=True):
    """
    创建 GPU 优化的数据流水线。

    Args:
        node_ids: numpy array, shape [N, 1]
        labels: numpy array, shape [N, 1] or [N,]
        batch_size: int
        shuffle: bool, 是否打乱
        cache: bool, 是否缓存到内存

    Returns:
        tf.data.Dataset
    """
    # 确保是 float32 (标签)


    labels = labels.astype('float32')

    ds = tf.data.Dataset.from_tensor_slices((node_ids, labels))

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(node_ids), 10000))

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
