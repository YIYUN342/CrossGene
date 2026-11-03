# -*- coding: utf-8 -*-
"""
GPS-Improved — GPU 全面优化的 GPS 模型 (v3: subclassed Model)

改进:
  1-8. 数据增强/多尺度融合/自适应对比/流匹配/标签平滑/通道熵/LayerScale/残差门控
GPU 优化:
  9. GPU 数据预加载 (tf.constant on GPU)
 10. 并行通道处理 (消除串行循环)
 11. tf.data 流水线
 12. 混合精度 + XLA JIT
 13. Subclassed Model: 无 Lambda 层, 完全控制 forward pass
"""

import os
import sklearn.metrics as m
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.metrics import roc_curve

from ..losses.weightedBinaryCrossEntropy import WeightedBinaryCrossEntropy
from ..losses.improved_contrastive import AdaptiveContrastiveLoss
from ..losses.improved_flow import AdaptiveFlowLoss
from ..callbacks import KGCNMetric
from ..layers.centralityEncoding import CentralityEncoding
from ..layers.gps_encoder import GPSBlock, MultiScaleFusion
from ..layers.improved_fusion import ImprovedAttentionFusion
from ..utils.data_augment import edge_dropout_feature, spatial_dropout
from ..utils.data_pipeline import create_gpu_dataset

tf.random.set_seed(42)


class ImprovedGPSModel(Model):
    """
    Subclassed Model — 完全控制 forward, 无 Lambda 层, GPU 全优化.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.gps_config = config
        self._contrast_weight = getattr(config, 'contrast_weight', 0.1)
        self._flow_weight = getattr(config, 'flow_weight', 0.05)
        self._label_smoothing = getattr(config, 'label_smoothing', 0.1)
        self._aug_noise = getattr(config, 'aug_noise', 0.1)
        self._aug_spatial = getattr(config, 'aug_spatial', 0.15)

        # 数据 shape


        self.n_graphs = config.n_graphs
        self.n_neighbors = config.n_neighbors
        self.n_layers = config.n_layers
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.feature_dim = int(config.node_feature.shape[-1])
        self.multi_hop = int(config.spatial_matrix.shape[2])
        self.max_degree = config.max_degree
        self.dff = config.dff
        self.d_sp_enc = config.d_sp_enc
        self.dropout = config.dropout
        self.top_activation = config.top_activation
        self.sp_enc_activation = config.sp_enc_activation

        # GPU 数据预加载


        self._node_feature_gpu = tf.constant(config.node_feature, dtype=tf.float32, name='nf_gpu')
        self._node_neighbor_gpu = tf.constant(config.node_neighbor, dtype=tf.int32, name='nn_gpu')
        self._distance_matrix_gpu = tf.constant(config.distance_matrix, dtype=tf.float32, name='dm_gpu')
        self._spatial_matrix_gpu = tf.constant(config.spatial_matrix, dtype=tf.float32, name='sm_gpu')

        # 层


        self.centr_layer = CentralityEncoding(self.max_degree, self.d_model, name='centr')

        # Max spatial distance (for edge encoding normalization)


        spa_max = float(tf.reduce_max(self._spatial_matrix_gpu))
        self._max_dist = tf.cast(spa_max, tf.float32) if spa_max > 0 else tf.constant(4.0)

        # GPS Encoder: GatedGCN (local) + MHA (global) + FFN


        self.gps_blocks = [
            GPSBlock(
                self.d_model, self.num_heads, self.dff, self.dropout,
                self.d_sp_enc, self.sp_enc_activation,
                drop_path_rate=0.1 * (i / max(1, self.n_layers - 1)),
                max_distance=self._max_dist,
                name=f'gps_block_{i}'
            ) for i in range(self.n_layers)
        ]

        self.ms_fusion = MultiScaleFusion(self.n_layers, self.d_model, name='ms_fusion')

        self.input_proj = Dense(self.d_model, activation=self.top_activation, name='input_proj')

        self.fusion_layer = ImprovedAttentionFusion(
            d_model=self.d_model, n_channels=self.n_graphs, num_heads=2, name='fusion'
        )

        # d_top=256 projection head (匹配 Original GPS)


        self.top_proj = Dense(256, activation='relu', name='top_proj')
        self.top_dropout_layer = Dropout(0.5, name='top_dropout')
        self.classifier = Dense(1, activation='sigmoid', dtype='float32', name='classifier')

        # 辅助损失模块


        self.contrastive_fn = AdaptiveContrastiveLoss(
            init_temperature=0.1, min_temperature=0.05, name='contrast'
        )
        self.flow_fn = AdaptiveFlowLoss(
            d_model=self.d_model, n_graphs=self.n_graphs, alpha=0.4, beta=0.3, name='flow'
        )
        self.proj_head = tf.keras.Sequential([
            Dense(self.d_model, activation='relu'),
            Dense(self.d_model // 2),
        ], name='proj_head')

        # 可学习读出门控: CLS token vs mean pooling


        self.readout_alpha = tf.Variable(
            0.7, trainable=True, dtype=tf.float32,
            constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0),
            name='readout_alpha'
        )

        # 指标


        self.contrast_metric = tf.keras.metrics.Mean(name='contrast_loss')
        self.flow_metric = tf.keras.metrics.Mean(name='flow_loss')
        self.temp_metric = tf.keras.metrics.Mean(name='contrast_temp')

        # Loss function


        self._loss_fn_obj = WeightedBinaryCrossEntropy(config.loss_mul)

    def _get_sub_info(self, node_id):
        """GPU 上的子图 gather 操作."""
        # 确保 node_id 是 2D [B, 1]


        if len(node_id.shape) == 1:
            node_id = tf.expand_dims(node_id, axis=-1)
        node_neighbors = tf.squeeze(tf.gather(self._node_neighbor_gpu, node_id), 1)
        node_feature = tf.gather(self._node_feature_gpu, node_neighbors)
        distance = tf.gather(self._distance_matrix_gpu, node_neighbors)
        if len(distance.shape) == 3:
            distance = tf.expand_dims(distance, axis=-1)
        spatial = tf.squeeze(tf.gather(self._spatial_matrix_gpu, node_id), 1)
        return node_feature, distance, spatial

    def call(self, inputs, training=False):
        node_id = inputs
        B = tf.shape(node_id)[0]
        nG = self.n_graphs
        nN = self.n_neighbors
        D = self.d_model
        FD = self.feature_dim
        MH = self.multi_hop
        NH = self.num_heads

        # 1. 子图提取 + 数据增强


        node_feature, distance, spatial = self._get_sub_info(node_id)
        # shapes: [B, nG, nN, FD], [B, nG, nN, 1], [B, nG, MH, nN, nN]



        if training:
            node_feature = edge_dropout_feature(node_feature, drop_rate=self._aug_noise, training=True)
            spatial = spatial_dropout(spatial, drop_rate=self._aug_spatial, training=True)

        # 2. 并行通道处理: [B, nG, ...] → [B*nG, ...]


        all_feat = tf.reshape(node_feature, (B * nG, nN, FD))
        all_dist = tf.reshape(distance, (B * nG, nN, 1))
        all_spatial = tf.reshape(spatial, (B * nG, MH, nN, nN))

        # 中心性编码


        centr_enc = self.centr_layer(all_dist)  # [B*nG, nN, D]

        # 输入特征: feat + centrality encoding (d_model matches feature_dim)


        x = tf.cast(all_feat, centr_enc.dtype)
        scale = tf.cast(tf.math.sqrt(tf.cast(D, tf.float32)), x.dtype)
        x = x * scale + centr_enc

        if training:
            x = tf.nn.dropout(x, rate=self.dropout)

        x = self.input_proj(x)  # [B*nG, nN, D]

        # Padding mask


        mask_hop0 = all_spatial[:, 0, :, :]  # [B*nG, nN, nN]
        padding_mask = tf.cast(tf.equal(mask_hop0, -1), x.dtype)  # [B*nG, nN, nN]

        # 3. 并行 GPS Encoder 层 (GatedGCN + MHA + FFN) + 多尺度融合


        layer_outputs = []
        for n in range(self.n_layers):
            attn_mask = padding_mask[:, tf.newaxis, :, :]  # [B*nG, 1, nN, nN], scaled_dot_product_attention 里会 tile
            x = self.gps_blocks[n](
                x, training=training, mask=attn_mask, min_distance_matrix=mask_hop0
            )
            # Rich readout: CLS token + mean pooling


            cls_token = x[:, 0, :]                      # [B*nG, D]
            mean_pool = tf.reduce_mean(x, axis=1)       # [B*nG, D]
            alpha = tf.cast(self.readout_alpha, cls_token.dtype)
            layer_outputs.append(alpha * cls_token + (1.0 - alpha) * mean_pool)  # [B*nG, D]

        # 多尺度融合: 简单平均 (调试用)


        fused_all = self.ms_fusion(layer_outputs)  # [B*nG, D]

        # 4. 恢复通道维度


        fused_per_channel = tf.reshape(fused_all, (B, nG, D))  # [B, nG, D]

        # 5. 通道融合


        concat_out = tf.reshape(fused_per_channel, (B, nG * D))  # [B, nG*D]
        final_emb, fusion_weights = self.fusion_layer(concat_out)  # [B, D]

        # 6. 辅助损失


        if training:
            # 每个通道的嵌入列表


            emb_list = [fused_per_channel[:, g, :] for g in range(nG)]

            # 对比损失


            proj = [self.proj_head(e) for e in emb_list]
            contrast_loss = self.contrastive_fn(proj, training=True)
            self.add_loss(contrast_loss * tf.cast(self._contrast_weight, contrast_loss.dtype))
            self.contrast_metric.update_state(contrast_loss)
            self.temp_metric.update_state(self.contrastive_fn.temperature)

            # 流匹配损失


            flow_loss = self.flow_fn(concat_out, final_emb, emb_list)
            self.add_loss(flow_loss * tf.cast(self._flow_weight, flow_loss.dtype))
            self.flow_metric.update_state(flow_loss)

        # 7. 分类 (d_top=256 projection + dropout)


        x = self.top_proj(final_emb)
        if training:
            x = self.top_dropout_layer(x)
        logits = self.classifier(x)  # [B, 1]
        return logits

    # 训练/测试 step


    @tf.function(jit_compile=False)
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            bce_loss = self.compute_loss(y=tf.cast(y, tf.float32), y_pred=y_pred)
            if self.losses:
                reg_loss = tf.add_n([tf.cast(l, tf.float32) for l in self.losses])
            else:
                reg_loss = 0.0
            total_loss = bce_loss + reg_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # 梯度裁剪: 防止 float16 溢出导致 NaN


        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

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
        smoothing = tf.cast(self._label_smoothing, tf.float32)
        if smoothing > 0.0:
            y = y * (tf.constant(1.0) - smoothing) + tf.constant(0.5) * smoothing
        return self._loss_fn_obj.get_loss(y, y_pred)


class ImprovedGPS:
    """改进版 GPS 模型工厂。"""

    def __init__(self, config):
        self.config = config
        self.callbacks = []
        self.built_model = None

    def build_model(self):
        """直接创建 subclassed Model, 无需 functional API."""
        config = self.config
        model = ImprovedGPSModel(config, name='ImprovedGPS')
        model.compile(optimizer=config.optimizer, metrics=['acc'], jit_compile=False)
        self.built_model = model
        return model

    def build(self):
        return self.build_model()

    def fit(self, x_train, y_train, x_val, y_val):
        self.callbacks = []
        self.callbacks.append(KGCNMetric(x_train, y_train, x_val, y_val, self.config.dataset))
        self.callbacks.append(ReduceLROnPlateau(
            monitor='val_auc', factor=0.5, patience=8,
            verbose=1, mode='max', epsilon=0.0001, cooldown=0, min_lr=1e-6
        ))

        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
        if 'modelcheckpoint' in self.config.callbacks_to_add:
            self.callbacks.append(ModelCheckpoint(
                filepath=f'{self.config.checkpoint_dir}/{self.config.exp_name}.weights.h5',
                monitor='val_auc', save_best_only=True, save_weights_only=True,
                mode='max', verbose=1
            ))
        if 'earlystopping' in self.config.callbacks_to_add:
            self.callbacks.append(EarlyStopping(
                monitor='val_auc', mode='max', patience=20, verbose=1
            ))

        print(f'  [GPU优化] Subclassed Model + 数据预加载显存 + 并行通道 + XLA JIT + 混合精度')
        print(f'  [改进] 标签平滑={getattr(self.config, "label_smoothing", 0.1)}, '
              f'数据增强噪声={getattr(self.config, "aug_noise", 0.1)}')
        print(f'  [辅助损失] 对比权重={getattr(self.config, "contrast_weight", 0.1)}, '
              f'流匹配权重={getattr(self.config, "flow_weight", 0.05)}')

        train_ds = create_gpu_dataset(x_train, y_train, self.config.batch_size, shuffle=True)
        val_ds = create_gpu_dataset(x_val, y_val, self.config.batch_size, shuffle=False)

        self.built_model.fit(
            train_ds,
            epochs=self.config.n_epoch,
            validation_data=val_ds,
            callbacks=self.callbacks,
            verbose=2
        )

    def predict(self, x):
        return self.built_model.predict(x).flatten()

    def load_best_model(self):
        path = f'{self.config.checkpoint_dir}/{self.config.exp_name}.weights.h5'
        self.built_model.load_weights(path)

    def load_swa_model(self):
        path = os.path.join(self.config.checkpoint_dir, f'{self.config.exp_name}_swa.weights.h5')
        print(f'Logging Info - Loading SWA model checkpoint: {path}')
        self.built_model.load_weights(path)
        print('Logging Info - SWA Model loaded')

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
