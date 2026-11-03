# -*- coding: utf-8 -*-
# Lazy imports to avoid circular dependency (node2vec → walker)



def get_node2vec():
    from .node2vec import Node2vec
    return Node2vec

Node2vec = None  # placeholder, use get_node2vec() or import directly

try:
    from .pipeline import create_gpu_dataset
    from .augment import edge_dropout_feature, spatial_dropout
except ImportError:
    pass  # TensorFlow not available
