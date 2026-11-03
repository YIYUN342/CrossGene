import tensorflow as tf


class CentralityEncoding(tf.keras.layers.Layer):
    def __init__(self, max_degree, d_model, **kwargs):
        super(CentralityEncoding, self).__init__(**kwargs)
        self.centr_embedding = tf.keras.layers.Embedding(
            input_dim=max_degree, output_dim=d_model, name='centr_embedding')

    def centrality(self, distances):
        # 计算度中心性: 统计距离为 1 的邻居数


        centrality = tf.cast(tf.math.equal(tf.math.abs(distances), 1), tf.float32)
        centrality = tf.math.reduce_sum(centrality, axis=-1, keepdims=False)
        return tf.cast(centrality, tf.int32)

    def call(self, distances):
        centrality = self.centrality(distances)
        centrality_encoding = self.centr_embedding(centrality)
        # 遵循混合精度策略，不强制 cast 到 float32


        return centrality_encoding
