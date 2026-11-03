# -*- coding: utf-8 -*-
"""tf.data GPU 优化数据流水线"""

import tensorflow as tf


def create_gpu_dataset(node_ids, labels, batch_size, shuffle=True, cache=True):
    """创建带 cache / prefetch 的 tf.data.Dataset"""
    labels = labels.astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((node_ids, labels))
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(node_ids), 10000))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
