import tensorflow as tf
from .spatialEncoding import SpatialEncoding


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_sp_enc, sp_enc_activation):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_sp_enc = d_sp_enc
        self.sp_enc_activation = sp_enc_activation
        self.spatial_encoding = SpatialEncoding(self.d_sp_enc, self.sp_enc_activation)

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, min_distance_matrix, mask):
        batch_size = tf.shape(q)[0]

        spatial_encoding_bias = self.spatial_encoding(min_distance_matrix)

        q = self.wq(q)
        k = self.wk(spatial_encoding_bias)
        v = self.wv(spatial_encoding_bias)

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
    # 混合精度兼容: 使用输入自身的 dtype 计算 scaling


    dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
    scaling = tf.math.rsqrt(dk)
    scaled_attention_logits = matmul_qk * scaling

    if mask is not None:
        mask_tiled = tf.tile(mask, [1, num_heads, 1, 1])
        # float16 安全值: -10000.0 (float16 max ≈ 65504)


        mask_value = tf.cast(-10000.0, scaled_attention_logits.dtype)
        scaled_attention_logits += (tf.cast(mask_tiled, scaled_attention_logits.dtype) * mask_value)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=3)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights
