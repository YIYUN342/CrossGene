import tensorflow as tf

class WeightedBinaryCrossEntropy(object):
    def __init__(self, pos_ratio):
        self.pos = pos_ratio
        self.neg = 1.0 - pos_ratio
        self.sample_weights = tf.cast(self.neg / self.pos, tf.float32)

    def get_loss(self, y_true, y_pred):
        y = tf.cast(y_true, y_pred.dtype)

        epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = tf.math.log(y_pred / (1 - y_pred))  ##做解码

        cost = tf.nn.weighted_cross_entropy_with_logits(y, y_pred, tf.cast(self.sample_weights, y_pred.dtype))

        return tf.reduce_mean(cost * tf.cast(self.pos, y_pred.dtype), axis=-1)