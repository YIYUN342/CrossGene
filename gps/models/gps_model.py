# -*- coding: utf-8 -*-
"""
GPS: 基于 GPS 编码器的图表示学习模型

架构:
  1. 子图提取 (GPU tf.constant 预加载)
  2. 数据增强 (特征噪声 + 空间掩码)
  3. 中心性编码 + 特征投影
  4. GPS 编码器堆栈 (GatedGCN + MHA + FFN) × n_layers
  5. 多尺度融合 (可学习加权)
  6. 通道注意力融合 (多头 + 残差门控 + 熵正则)
  7. 辅助损失: 自适应对比学习 + 流匹配
  8. 分类: Dense(1, sigmoid)
"""

import sklearn.metrics as m
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, roc_curve

from ..losses import WeightedBinaryCrossEntropy
from ..losses.contrastive import AdaptiveContrastiveLoss
from ..losses.flow_matching import AdaptiveFlowLoss
from ..callbacks import KGCNMetric
from ..layers import CentralityEncoding, GPSBlock, MultiScaleFusion, ChannelFusion
from ..utils.augment import edge_dropout_feature, spatial_dropout
from ..utils.pipeline import create_gpu_dataset

tf.random.set_seed(42)


class GPSModel(Model):
    """Subclassed Model — 混合精度 + XLA + GPU 全优化"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config
        self._contrast_weight = getattr(config, 'contrast_weight', 0.1)
        self._flow_weight = getattr(config, 'flow_weight', 0.05)
        self._label_smoothing = getattr(config, 'label_smoothing', 0.05)
        self._aug_noise = getattr(config, 'aug_noise', 0.1)
        self._aug_spatial = getattr(config, 'aug_spatial', 0.15)

        self.n_graphs = config.n_graphs
        self.n_neighbors = config.n_neighbors
        self.n_layers = config.n_layers
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.feature_dim = int(config.node_feature.shape[-1])
        self.multi_hop = int(config.spatial_matrix.shape[2])
        self.max_degree = config.max_degree
        self.dff = config.dff
        self.dropout = config.dropout
        self.d_sp_enc = config.d_sp_enc

        # GPU 数据预加载


        self._nf = tf.constant(config.node_feature, dtype=tf.float32, name='nf')
        self._nn = tf.constant(config.node_neighbor, dtype=tf.int32, name='nn')
        self._dm = tf.constant(config.distance_matrix, dtype=tf.float32, name='dm')
        self._sm = tf.constant(config.spatial_matrix, dtype=tf.float32, name='sm')

        # 层


        self.centr_layer = CentralityEncoding(self.max_degree, self.d_model, name='centr')
        self.gps_blocks = [
            GPSBlock(self.d_model, self.num_heads, self.dff, self.dropout,
                     self.d_sp_enc, config.sp_enc_activation,
                     drop_path_rate=0.1 * (i / max(1, self.n_layers - 1)),
                     name=f'gps_block_{i}')
            for i in range(self.n_layers)
        ]
        self.ms_fusion = MultiScaleFusion(self.n_layers, self.d_model, name='ms_fusion')
        self.input_proj = Dense(self.d_model, activation=config.top_activation, name='input_proj')
        self.fusion_layer = ChannelFusion(self.d_model, self.n_graphs, num_heads=2, name='fusion')
        self.classifier = Dense(1, activation='sigmoid', dtype='float32', name='classifier')

        # 辅助损失


        self.contrastive_fn = AdaptiveContrastiveLoss(
            init_temperature=getattr(config, 'contrast_temperature', 0.5),
            min_temperature=0.05, name='contrast')
        self.flow_fn = AdaptiveFlowLoss(
            d_model=self.d_model, n_graphs=self.n_graphs, alpha=0.4, beta=0.3, name='flow')
        self.proj_head = tf.keras.Sequential([
            Dense(self.d_model, activation='relu'),
            Dense(self.d_model // 2),
        ], name='proj_head')

        self.contrast_metric = tf.keras.metrics.Mean(name='contrast_loss')
        self.flow_metric = tf.keras.metrics.Mean(name='flow_loss')
        self.temp_metric = tf.keras.metrics.Mean(name='contrast_temp')
        self._loss_fn_obj = WeightedBinaryCrossEntropy(config.loss_mul)

    def _get_sub_info(self, node_id):
        if len(node_id.shape) == 1:
            node_id = tf.expand_dims(node_id, axis=-1)
        nn = tf.squeeze(tf.gather(self._nn, node_id), 1)
        nf = tf.gather(self._nf, nn)
        dist = tf.gather(self._dm, nn)
        if len(dist.shape) == 3:
            dist = tf.expand_dims(dist, axis=-1)
        spatial = tf.squeeze(tf.gather(self._sm, node_id), 1)
        return nf, dist, spatial

    def call(self, inputs, training=False):
        node_id = inputs
        B = tf.shape(node_id)[0]
        G = self.n_graphs
        K = self.n_neighbors
        D = self.d_model
        FD = self.feature_dim
        MH = self.multi_hop

        # 1. 子图提取 + 数据增强


        node_feature, distance, spatial = self._get_sub_info(node_id)

        if training:
            node_feature = edge_dropout_feature(node_feature, self._aug_noise, True)
            spatial = spatial_dropout(spatial, self._aug_spatial, True)

        # 2. 并行通道: [B, G, ...] → [B*G, ...]


        all_feat = tf.reshape(node_feature, (B * G, K, FD))
        all_dist = tf.reshape(distance, (B * G, K, 1))
        all_spatial = tf.reshape(spatial, (B * G, MH, K, K))

        # 3. 中心性编码 + 输入投影


        centr_enc = self.centr_layer(all_dist)
        x = tf.cast(all_feat, centr_enc.dtype)
        x = x * tf.cast(tf.math.sqrt(tf.cast(D, tf.float32)), x.dtype) + centr_enc
        if training:
            x = tf.nn.dropout(x, rate=self.dropout)
        x = self.input_proj(x)

        mask_hop0 = all_spatial[:, 0, :, :]
        padding_mask = tf.cast(tf.equal(mask_hop0, -1), x.dtype)

        # 4. GPS 编码器堆栈 + 多尺度融合


        layer_outputs = []
        for n in range(self.n_layers):
            attn_mask = padding_mask[:, tf.newaxis, :, :]
            x = self.gps_blocks[n](x, training=training, mask=attn_mask,
                                   min_distance_matrix=mask_hop0)
            layer_outputs.append(x[:, 0, :])  # CLS token

        fused_all = tf.add_n(layer_outputs) / tf.cast(len(layer_outputs), layer_outputs[0].dtype)

        # 5. 通道融合


        fused_per_channel = tf.reshape(fused_all, (B, G, D))
        concat_out = tf.reshape(fused_per_channel, (B, G * D))
        final_emb, _ = self.fusion_layer(concat_out)

        # 6. 辅助损失 (训练时)


        if training:
            emb_list = [fused_per_channel[:, g, :] for g in range(G)]
            proj = [self.proj_head(e) for e in emb_list]
            contrast_loss = self.contrastive_fn(proj, training=True)
            self.add_loss(contrast_loss * tf.cast(self._contrast_weight, contrast_loss.dtype))
            self.contrast_metric.update_state(contrast_loss)
            self.temp_metric.update_state(self.contrastive_fn.temperature)

            flow_loss = self.flow_fn(concat_out, final_emb, emb_list)
            self.add_loss(flow_loss * tf.cast(self._flow_weight, flow_loss.dtype))
            self.flow_metric.update_state(flow_loss)

        # 7. 分类


        return self.classifier(final_emb)

    @tf.function(jit_compile=False)
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            bce_loss = self.compute_loss(y=tf.cast(y, tf.float32), y_pred=y_pred)
            reg_loss = tf.add_n([tf.cast(l, tf.float32) for l in self.losses]) if self.losses else 0.0
            total_loss = bce_loss + reg_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(total_loss)
            elif metric.name == 'bce_loss':
                metric.update_state(bce_loss)
            elif metric.name == 'reg_loss':
                metric.update_state(reg_loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function(jit_compile=False)
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        total_loss = self.compute_loss(y=tf.cast(y, tf.float32), y_pred=y_pred)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(total_loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def compute_loss(self, y, y_pred):
        s = tf.cast(self._label_smoothing, tf.float32)
        if s > 0.0:
            y = y * (1.0 - s) + 0.5 * s
        return self._loss_fn_obj.get_loss(y, y_pred)


class GPS:
    """GPS 模型工厂 — 训练、评估、保存/加载"""

    def __init__(self, config):
        self.config = config
        self.callbacks = []
        self.built_model = None

    def build(self):
        return self.build_model()

    def build_model(self):
        config = self.config
        model = GPSModel(config, name='GPS')
        model.compile(optimizer=config.optimizer, metrics=['acc'], jit_compile=False)
        self.built_model = model
        return model

    def fit(self, x_train, y_train, x_val, y_val):
        self.callbacks = []
        self.callbacks.append(KGCNMetric(x_train, y_train, x_val, y_val, self.config.dataset))
        self.callbacks.append(ReduceLROnPlateau(
            monitor='val_auc', factor=0.5, patience=5,
            verbose=1, mode='max', epsilon=0.0001, cooldown=0, min_lr=0))

        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
        if 'modelcheckpoint' in self.config.callbacks_to_add:
            self.callbacks.append(ModelCheckpoint(
                filepath=f'{self.config.checkpoint_dir}/{self.config.exp_name}.weights.h5',
                monitor='val_auc', save_best_only=True, save_weights_only=True,
                mode='max', verbose=1))
        if 'earlystopping' in self.config.callbacks_to_add:
            self.callbacks.append(EarlyStopping(
                monitor='val_auc', mode='max', patience=20, verbose=1))

        print(f'  GPS | 标签平滑={getattr(self.config, "label_smoothing", 0.05)} | '
              f'对比权重={getattr(self.config, "contrast_weight", 0.1)} | '
              f'流匹配权重={getattr(self.config, "flow_weight", 0.05)}')

        train_ds = create_gpu_dataset(x_train, y_train, self.config.batch_size, shuffle=True)
        val_ds = create_gpu_dataset(x_val, y_val, self.config.batch_size, shuffle=False)

        self.built_model.fit(
            train_ds, epochs=self.config.n_epoch,
            validation_data=val_ds, callbacks=self.callbacks, verbose=2)

    def predict(self, x):
        return self.built_model.predict(x).flatten()

    def load_best_model(self):
        path = f'{self.config.checkpoint_dir}/{self.config.exp_name}.weights.h5'
        self.built_model.load_weights(path)

    def score(self, x, y, threshold=0.5):
        y_true = y.flatten()
        y_pred = self.built_model.predict(x).flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_pred)
        p, r, t = precision_recall_curve(y_true=y_true, y_score=y_pred)
        aupr = m.auc(r, p)
        y_pred_bin = [1 if prob >= threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred_bin)
        prec = precision_score(y_true=y_true, y_pred=y_pred_bin)
        rec = recall_score(y_true=y_true, y_pred=y_pred_bin)
        f1 = f1_score(y_true=y_true, y_pred=y_pred_bin)
        return auc, acc, prec, rec, f1, aupr, fpr.tolist(), tpr.tolist()
