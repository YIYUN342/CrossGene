# -*- coding: utf-8 -*-
"""GPS Model Configuration"""
import os as _os

# 目录 (AutoDL server auto-detection)


_SERVER_DATA = '/root/autodl-tmp/dataset/'
_SERVER_PDATA = '/root/autodl-tmp/pdata/'

if _os.path.exists(_SERVER_DATA):
    RAW_DATA_DIR = _SERVER_DATA
    PROCESSED_DATA_DIR = _SERVER_PDATA
else:
    RAW_DATA_DIR = _os.getcwd() + '/data/'
    PROCESSED_DATA_DIR = _os.getcwd() + '/pdata/'

LOG_DIR = _os.getcwd() + '/log'
MODEL_SAVED_DIR = _os.getcwd() + '/ckpt'

H5_FILE = {
    'CPDB': _os.path.join(RAW_DATA_DIR, 'CPDB_multiomics.h5'),
}

SPATIAL_TEMPLATE = '{dataset}_method_{strategy}_channel_{n_channel}_neighbor_{n_neighbor}_spatial.npy'
SUBGRAPHA_TEMPLATE = '{dataset}_method_{strategy}_channel_{n_channel}_neighbor_{n_neighbor}_subgraphs.npy'
PERFORMANCE_LOG = 'GPS_performance'


class ImprovedModelConfig:
    """GPS 模型配置 (best: AUC=0.854, AUPR=0.718 on CPDB)"""

    def __init__(self):
        # 基础配置


        self.n_layers = 3           # GPS 编码器层数
        self.d_model = 64           # 隐藏维度
        self.l2_weight = 1e-7       # L2 正则化
        self.lr = 0.002             # 学习率
        self.n_epoch = 150          # 训练轮数
        self.dff = 128              # FFN 中间维度
        self.max_degree = 1322      # 最大度数 (中心性编码)
        self.n_neighbors = 8        # 每个子图邻居数
        self.n_graphs = 6           # 子图通道数
        self.num_heads = 4          # 注意力头数
        self.concat_n_layers = self.n_layers
        self.dropout = 0.3          # Dropout 率
        self.loss_mul = 0.35        # 正类 BCE 权重
        self.d_sp_enc = 128         # 空间编码维度
        self.batch_size = 64        # 批次大小
        self.top_dropout = 0.5      # 分类头 Dropout
        self.d_top = 256            # 分类头投影维度
        self.optimizer = 'adam'
        self.model_head = 'cls_mean'  # CLS + MeanPool 门控读出
        self.sp_enc_activation = "relu"
        self.top_activation = "relu"

        # 辅助损失


        self.contrast_weight = 0.1       # 对比损失权重
        self.contrast_temperature = 0.1  # 对比损失初始温度
        self.flow_weight = 0.05          # 流匹配损失权重

        # 正则化


        self.label_smoothing = 0.05      # 标签平滑
        self.aug_noise = 0.1             # 特征噪声强度
        self.aug_spatial = 0.15          # 空间掩码比例

        # 数据 (运行时注入)


        self.distance_matrix = None
        self.node_feature = None
        self.node_neighbor = None
        self.spatial_matrix = None
        self.training = None

        # 训练


        self.exp_name = None
        self.model_name = None
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_auc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1
        self.early_stopping_monitor = 'val_auc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 20
        self.early_stopping_verbose = 1
        self.K_Fold = 1
        self.callbacks_to_add = None
        self.swa_start = 3
        self.dataset = None
