# -*- coding: utf-8 -*-
"""损失函数模块"""
try:
    from .weightedBinaryCrossEntropy import WeightedBinaryCrossEntropy
    from .improved_contrastive import AdaptiveContrastiveLoss
    from .improved_flow import AdaptiveFlowLoss
except ImportError:
    pass
