"""GPS Graphormer inference — predict gene-disease scores for all CPDB genes.
(LEGACY: use gps_inference.py for the new GPS model)

Usage:  python inference/gps_inference_legacy.py
Output: gps_predictions_CPDB.json
"""
import os, sys, json, gc
import numpy as np

# Ensure cwd is project root


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# GPU / TF compat


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import tensorflow as tf
from tensorflow.keras import backend as K

tf.config.optimizer.set_jit(False)
tf.config.set_soft_device_placement(True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU: {len(gpus)} device(s)")

from config import ModelConfig, PROCESSED_DATA_DIR, ADJ_TEMPLATE, FEATURE_TEMPLATE, SUBGRAPHA_TEMPLATE, SPATIAL_TEMPLATE, H5_FILE
from utils import format_filename
from models import GPS


OUTPUT_PATH = r"D:\pyProject\GeneAgent\gps_predictions.json"


def run_inference():
    dataset = "CPDB"
    config = ModelConfig()

    # Match the checkpoint we found: GPS_CPDB_spatialType_rw_layerNums_3_graphsNum_6_neighborsNum_8_optimizer_adam_lr_0.001__epoch_30__fold_0


    config.dataset = dataset
    config.n_layers = 3
    config.n_graphs = 6
    config.n_neighbors = 8
    config.lr = 0.001
    config.optimizer = "adam"
    config.n_epoch = 30
    config.K_Fold = 0
    config.d_model = 64
    config.num_heads = 4
    config.d_sp_enc = 128
    config.dff = 128
    config.batch_size = 8
    config.dropout = 0.5
    config.loss_mul = 0.27
    config.max_degree = 1596  # CPDB actual max_degree + 1
    config.training = True     # required to build contrastive/flow auxiliary layers

    config.exp_name = (
        f"GPS_{dataset}_spatialType_rw_layerNums_{config.n_layers}"
        f"_graphsNum_{config.n_graphs}_neighborsNum_{config.n_neighbors}"
        f"_optimizer_{config.optimizer}_lr_{config.lr}"
        f"__epoch_{config.n_epoch}__fold_{config.K_Fold}"
    )

    # Load preprocessed data


    config.distance_matrix = np.load(
        format_filename(PROCESSED_DATA_DIR, ADJ_TEMPLATE, dataset=dataset), allow_pickle=True)
    config.node_feature = np.load(
        format_filename(PROCESSED_DATA_DIR, FEATURE_TEMPLATE, dataset=dataset), allow_pickle=True)
    config.node_neighbor = np.load(
        format_filename(PROCESSED_DATA_DIR, SUBGRAPHA_TEMPLATE, dataset=dataset,
                        strategy='rw', n_channel=config.n_graphs, n_neighbor=config.n_neighbors),
        allow_pickle=True)
    config.spatial_matrix = np.load(
        format_filename(PROCESSED_DATA_DIR, SPATIAL_TEMPLATE, dataset=dataset,
                        strategy='rw', n_channel=config.n_graphs, n_neighbor=config.n_neighbors),
        allow_pickle=True)

    n_genes = config.node_feature.shape[0]
    print(f"Loaded CPDB: {n_genes} genes, features={config.node_feature.shape}")

    # Build model & force-build ALL layers including auxiliary losses


    model = GPS(config)
    # Direct call with training=True forces auxiliary layer building


    dummy = tf.constant(np.arange(config.batch_size, dtype=np.int32))
    _ = model.model(dummy, training=True)
    # Clear any side effects and load pre-trained weights


    model.load_best_model()

    # Predict for ALL genes (use all indices 0..n_genes-1)


    all_indices = np.arange(n_genes, dtype=np.int32)
    print(f"Running inference on all {n_genes} genes...")
    scores = model.predict(all_indices)  # shape: (n_genes,)

    # Load gene names


    import h5py
    with h5py.File(H5_FILE[dataset], 'r') as f:
        gene_names_raw = f['gene_names'][:]
        gene_symbols = []
        for row in gene_names_raw:
            s = row[1]
            gene_symbols.append(s.decode() if isinstance(s, bytes) else str(s))

    # Build output: gene → GPS score


    predictions = {}
    for i in range(n_genes):
        predictions[gene_symbols[i]] = {
            "gene_id": i,
            "gene_name": gene_symbols[i],
            "gps_score": float(scores[i]) if hasattr(scores[i], '__iter__') else float(scores[i]),
        }

    # Sort by score descending for convenience


    sorted_preds = dict(sorted(predictions.items(),
                               key=lambda x: x[1]["gps_score"], reverse=True))

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            "model": config.exp_name,
            "n_genes": n_genes,
            "predictions": sorted_preds,
            "top_20": [{"gene_name": k, "gps_score": v["gps_score"]}
                       for k, v in list(sorted_preds.items())[:20]],
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved {n_genes} gene predictions to {OUTPUT_PATH}")

    # Print top 20


    print("\nTop 20 GPS predictions:")
    for i, (sym, pred) in enumerate(list(sorted_preds.items())[:20]):
        print(f"  {i+1:3d}. {sym:<12s}  score={pred['gps_score']:.4f}")

    # Cleanup


    del model
    gc.collect()
    K.clear_session()


if __name__ == '__main__':
    run_inference()
