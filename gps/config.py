# -*- coding: utf-8 -*-
"""GPS 项目配置"""

import os

# 数据路径


_SERVER_DATA = '/root/autodl-tmp/dataset/'
_SERVER_PDATA = '/root/autodl-tmp/pdata/'
_SERVER_SP = '/root/autodl-tmp/sp/'

if os.path.exists(_SERVER_DATA):
    # AutoDL 服务器环境


    RAW_DATA_DIR = _SERVER_DATA
    PROCESSED_DATA_DIR = _SERVER_PDATA
    SP_DIR = _SERVER_SP
else:
    # 本地开发环境


    RAW_DATA_DIR = os.getcwd() + '/dataset/networks/'
    PROCESSED_DATA_DIR = os.getcwd() + '/pdata/'
    SP_DIR = os.getcwd() + '/sp/'

LOG_DIR = os.getcwd() + '/log/'
MODEL_SAVED_DIR = os.getcwd() + '/ckpt/'

# 数据集


DATASETS = ['CPDB', 'PCNET', 'MULTINET', 'IREF', 'IREF_2015', 'STRINGdb', 'LTG', 'MTG']

H5_FILE = {ds: os.path.join(RAW_DATA_DIR, f'{ds}_multiomics.h5') for ds in DATASETS}

# 预处理文件模板


SPATIAL_TEMPLATE = '{dataset}_method_{strategy}_channel_{n_channel}_neighbor_{n_neighbor}_spatial.npy'
SUBGRAPH_TEMPLATE = '{dataset}_method_{strategy}_channel_{n_channel}_neighbor_{n_neighbor}_subgraphs.npy'
ADJ_TEMPLATE = '{dataset}_adj.npy'
FEATURE_TEMPLATE = '{dataset}_feature.npy'
SHORT_PATH = '{dataset}_sp.npy'

RESULT_LOG = {ds: f'{ds}_gps_result.txt' for ds in DATASETS}
PERFORMANCE_LOG = 'GPS_performance'


class ModelConfig:
    """GPS 模型超参数"""

    def __init__(self):
        # 网络结构


        self.n_layers = 3          # GPS 编码器层数
        self.d_model = 64          # 隐藏维度
        self.num_heads = 4         # 注意力头数
        self.dff = 128             # FFN 中间维度
        self.d_sp_enc = 128        # 空间编码维度
        self.sp_enc_activation = 'relu'
        self.top_activation = 'relu'
        self.model_head = 'average'

        # 子图采样


        self.n_graphs = 6          # 随机游走视图数
        self.n_neighbors = 8       # 每个视图的邻居数
        self.max_degree = 1322     # 最大度 (用于中心性编码 embedding)

        # 训练


        self.n_epoch = 100
        self.batch_size = 128
        self.lr = 0.001
        self.dropout = 0.5
        self.top_dropout = 0.5
        self.d_top = 256
        self.optimizer = 'adam'
        self.l2_weight = 1e-7
        self.loss_mul = 0.27       # 正样本权重

        # 辅助损失


        self.contrast_weight = 0.1
        self.contrast_temperature = 0.5
        self.flow_weight = 0.05

        # 正则化


        self.label_smoothing = 0.05
        self.aug_noise = 0.1        # 特征噪声强度
        self.aug_spatial = 0.15     # 空间掩码比例

        # 回调


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
        self.K_Fold = 5
        self.callbacks_to_add = None
        self.swa_start = 3

        # 运行时变量


        self.distance_matrix = None
        self.node_feature = None
        self.node_neighbor = None
        self.spatial_matrix = None
        self.training = None
        self.exp_name = None
        self.model_name = None
        self.dataset = None
        self.concat_n_layers = None
