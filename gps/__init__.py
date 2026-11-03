# -*- coding: utf-8 -*-
"""GPS 图神经网络模块 (GatedGCN + 多头注意力 + FFN)。依赖 TensorFlow，无 TF 环境时惰性导入。"""

from .config_improved import ImprovedModelConfig as ModelConfig

try:
    from .models.gps_improved import ImprovedGPS, ImprovedGPSModel
    from .layers.gps_encoder import GPSBlock, GatedGCNLayer, MultiScaleFusion
    from .layers.multiHeadAttention import MultiHeadAttention
    from .layers.centralityEncoding import CentralityEncoding
    from .layers.spatialEncoding import SpatialEncoding
    from .layers.improved_fusion import ImprovedAttentionFusion
except ImportError:
    pass  # TensorFlow not available
