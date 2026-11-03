#!/usr/bin/env python
"""GPS Block 推理 — 完全按 crossgene-new2/train.py 配置:
  - 预计算子图 .npy
  - 150 epochs, lr=0.002, dropout=0.3, bs=64, loss_mul=0.35
  - 5-Fold CV, XLA JIT, Mixed Precision
  - 输出: gps_predictions_{dataset}.json (全量基因 gps_score)
"""
import sys, os, gc, time, json, argparse
import numpy as np, h5py
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DATA_DIR = os.environ.get("CROSSGENE_H5_DIR", "/root/autodl-tmp/dataset")
PDATA_DIR = os.environ.get("GPS_PDATA_DIR", "/root/autodl-tmp/pdata")

def ts(): return datetime.now().strftime('%H:%M:%S')

# Dataset subgraph configs (from crossgene-new2/train.py)


DATASET_CONFIGS = {
    'CPDB':       {'h5': 'CPDB_multiomics.h5',      'src_G': 8, 'src_N': 9,  'tgt_G': 6, 'tgt_N': 8},
    'STRINGdb':   {'h5': 'STRINGdb_multiomics.h5',   'src_G': 8, 'src_N': 9,  'tgt_G': 6, 'tgt_N': 8},
    'IREF':       {'h5': 'IREF_multiomics.h5',       'src_G': 8, 'src_N': 10, 'tgt_G': 6, 'tgt_N': 8},
    'IREF_2015':  {'h5': 'IREF_2015_multiomics.h5',  'src_G': 8, 'src_N': 9,  'tgt_G': 6, 'tgt_N': 8},
    'LTG':        {'h5': 'LTG_multiomics.h5',        'src_G': 10,'src_N': 14, 'tgt_G': 6, 'tgt_N': 8},
    'MTG':        {'h5': 'MTG_multiomics.h5',        'src_G': 10,'src_N': 14, 'tgt_G': 6, 'tgt_N': 8},
    'MULTINET':   {'h5': 'MULTINET_multiomics.h5',   'src_G': 8, 'src_N': 9,  'tgt_G': 6, 'tgt_N': 8},
    'PCNET':      {'h5': 'PCNET_multiomics.h5',      'src_G': 10,'src_N': 14, 'tgt_G': 6, 'tgt_N': 8},
}

# Best hyperparameters (from crossgene-new2/train.py HP dict)


HP = {
    'd_model': 64, 'n_layers': 3, 'num_heads': 4, 'd_sp_enc': 128, 'dff': 128,
    'dropout': 0.3, 'loss_mul': 0.35, 'lr': 0.002, 'batch_size': 64, 'n_epoch': 150,
    'contrast_weight': 0.1, 'flow_weight': 0.05, 'label_smoothing': 0.05,
    'aug_noise': 0.1, 'aug_spatial': 0.15, 'n_folds': 5,
}


def train_and_infer(dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]
    tgt_G, tgt_N = cfg['tgt_G'], cfg['tgt_N']
    src_G, src_N = cfg['src_G'], cfg['src_N']

    print(f"\n{'#'*60}")
    print(f"[{ts()}] GPS Block: {dataset_name} (G={tgt_G}/N={tgt_N} sliced from {src_G}/{src_N})")
    print(f"{'#'*60}")

    # Load H5


    h5_path = f'{DATA_DIR}/{cfg["h5"]}'
    with h5py.File(h5_path, 'r') as f:
        adj = f['network'][:]
        features = f['features'][:]
        y_train = f['y_train'][:].flatten()
        y_test = f['y_test'][:].flatten()
        y_val = f['y_val'][:].flatten() if 'y_val' in f else None
        train_mask = f['mask_train'][:].flatten().astype(bool)
        test_mask = f['mask_test'][:].flatten().astype(bool)
        val_mask = f['mask_val'][:].flatten().astype(bool) if 'mask_val' in f else None
        gene_names_raw = f['gene_names'][:]

    n_nodes = adj.shape[0]
    test_id = np.where(test_mask)[0]
    y_test_final = y_test[test_id]
    train_val_mask = np.logical_or(train_mask, val_mask) if val_mask is not None else train_mask
    print(f"  Nodes={n_nodes}, Train+Val labeled={np.sum(train_val_mask)}, Test={len(test_id)}")

    # Decode gene names


    from tools.h5_reader import _decode_gene_names
    gene_symbols = _decode_gene_names(gene_names_raw)

    # Load pre-computed subgraphs


    sub_file = f'{PDATA_DIR}/{dataset_name}_multiomics_method_rw_channel_{src_G}_neighbor_{src_N}_subgraphs_p_4_q_4.npy'
    spa_file = f'{PDATA_DIR}/{dataset_name}_multiomics_method_rw_channel_{src_G}_neighbor_{src_N}_spatial_p_4_q_4.npy'
    print(f"  Loading subgraphs: {os.path.basename(sub_file)}...")
    t0 = time.time()
    neighbor_id = np.load(sub_file, allow_pickle=True)
    spatial_matrix = np.load(spa_file, allow_pickle=True)
    neighbor_id = neighbor_id[:, :tgt_G, :tgt_N]
    spatial_matrix = spatial_matrix[:, :tgt_G, :, :tgt_N, :tgt_N]
    print(f"  Loaded+sliced in {time.time()-t0:.0f}s: neighbor={neighbor_id.shape}, spatial={spatial_matrix.shape}")

    max_degree = int(np.max(np.sum(adj, axis=-1)) + 1)
    degree = np.expand_dims(np.sum(adj, axis=-1), axis=-1).astype(np.float32)

    # K-Fold CV


    from sklearn.model_selection import StratifiedKFold
    from gps.config_improved import ImprovedModelConfig
    from gps.models.gps_improved import ImprovedGPS
    import tensorflow as tf
    tf.config.optimizer.set_jit(True)
    print(f"  XLA JIT: ON")

    label_idx = np.where(train_val_mask)[0]
    y_train_val = y_train[test_id] if val_mask is None else np.logical_or(y_train, y_val).flatten()
    # Get labels for train_val_mask indices


    all_y = y_train.copy()  # y_train has labels for all masked positions
    if val_mask is not None:
        all_y[val_mask] = y_val[val_mask]
    kf = StratifiedKFold(n_splits=HP['n_folds'], shuffle=True, random_state=100)
    k_sets = list(kf.split(label_idx, all_y[label_idx]))

    cv_results = {"auc": [], "aupr": [], "acc": [], "prec": [], "rec": [], "f1": []}

    for i, (tr_fold_idx, vl_fold_idx) in enumerate(k_sets):
        train_id = label_idx[tr_fold_idx]
        val_id = label_idx[vl_fold_idx]
        y_train_fold = all_y[train_id]
        y_val_fold = all_y[val_id]
        print(f"\n  [{ts()}] Fold {i+1}/{HP['n_folds']}: train={len(train_id)}, val={len(val_id)}")

        config = ImprovedModelConfig()
        config.d_model = HP['d_model']
        config.n_layers = HP['n_layers']
        config.n_graphs = tgt_G
        config.n_neighbors = tgt_N
        config.max_degree = max_degree
        config.batch_size = HP['batch_size']
        config.n_epoch = HP['n_epoch']
        config.lr = HP['lr']
        config.num_heads = HP['num_heads']
        config.d_sp_enc = HP['d_sp_enc']
        config.dff = HP['dff']
        config.dropout = HP['dropout']
        config.loss_mul = HP['loss_mul']
        config.training = True
        config.dataset = dataset_name
        config.callbacks_to_add = ['modelcheckpoint', 'earlystopping']
        config.contrast_weight = HP['contrast_weight']
        config.flow_weight = HP['flow_weight']
        config.label_smoothing = HP['label_smoothing']
        config.aug_noise = HP['aug_noise']
        config.aug_spatial = HP['aug_spatial']
        config.optimizer = tf.keras.optimizers.Adam(learning_rate=HP['lr'])
        config.distance_matrix = degree
        config.node_feature = features.astype(np.float32)
        config.node_neighbor = neighbor_id
        config.spatial_matrix = spatial_matrix
        config.exp_name = f'GPSBlock_{dataset_name}_fold{i}'
        config.checkpoint_dir = os.path.join(PROJECT_ROOT, 'gps', 'ckpt')
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        model = ImprovedGPS(config)
        model.build_model()

        t0 = time.time()
        model.fit(
            x_train=train_id, y_train=y_train_fold.astype('float32').reshape(-1,1),
            x_val=val_id, y_val=y_val_fold.astype('float32').reshape(-1,1),
        )
        dt = time.time() - t0

        model.load_best_model()
        auc, acc, prec, rec, f1, aupr, _, _ = model.score(
            x=test_id, y=y_test_final.astype('float32').reshape(-1,1))
        print(f"  [{ts()}] Fold {i+1} Test: AUROC={auc:.4f} AUPRC={aupr:.4f} "
              f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} ({dt:.0f}s)")

        cv_results["auc"].append(auc); cv_results["aupr"].append(aupr)
        cv_results["acc"].append(acc); cv_results["prec"].append(prec)
        cv_results["rec"].append(rec); cv_results["f1"].append(f1)

        del model; gc.collect(); tf.keras.backend.clear_session()

    # CV Summary


    print(f"\n  [{ts()}] {dataset_name} {HP['n_folds']}-Fold CV Results:")
    for m in ["auc", "aupr", "acc", "prec", "rec", "f1"]:
        vals = cv_results[m]
        print(f"    {m.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Train FINAL model on all train+val, infer on all genes


    print(f"\n  [{ts()}] Training FINAL model on all {len(label_idx)} train+val genes...")
    config = ImprovedModelConfig()
    config.d_model = HP['d_model']; config.n_layers = HP['n_layers']
    config.n_graphs = tgt_G; config.n_neighbors = tgt_N; config.max_degree = max_degree
    config.batch_size = HP['batch_size']; config.n_epoch = HP['n_epoch']; config.lr = HP['lr']
    config.num_heads = HP['num_heads']; config.d_sp_enc = HP['d_sp_enc']; config.dff = HP['dff']
    config.dropout = HP['dropout']; config.loss_mul = HP['loss_mul']
    config.training = True; config.dataset = dataset_name
    config.callbacks_to_add = ['modelcheckpoint']
    config.contrast_weight = HP['contrast_weight']; config.flow_weight = HP['flow_weight']
    config.label_smoothing = HP['label_smoothing']; config.aug_noise = HP['aug_noise']
    config.aug_spatial = HP['aug_spatial']
    config.optimizer = tf.keras.optimizers.Adam(learning_rate=HP['lr'])
    config.distance_matrix = degree; config.node_feature = features.astype(np.float32)
    config.node_neighbor = neighbor_id; config.spatial_matrix = spatial_matrix
    config.exp_name = f'GPSBlock_{dataset_name}_FINAL'
    config.checkpoint_dir = os.path.join(PROJECT_ROOT, 'gps', 'ckpt')
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    model = ImprovedGPS(config)
    model.build_model()
    # Train on all labeled data, validate on test set (just for monitoring)


    model.fit(
        x_train=label_idx, y_train=all_y[label_idx].astype('float32').reshape(-1,1),
        x_val=test_id, y_val=y_test_final.astype('float32').reshape(-1,1),
    )
    model.load_best_model()

    # Full inference


    print(f"  [{ts()}] Inference on all {n_nodes} genes...")
    all_scores = []
    for i in range(0, n_nodes, 512):
        all_scores.append(model.predict(np.arange(i, min(i+512, n_nodes), dtype=np.int32)))
    all_scores = np.concatenate(all_scores)

    # Save


    predictions = {}
    for i in range(n_nodes):
        predictions[gene_symbols[i]] = {
            "gene_id": i, "gene_name": gene_symbols[i],
            "gps_score": float(all_scores[i]),
        }
    sorted_preds = dict(sorted(predictions.items(), key=lambda x: x[1]["gps_score"], reverse=True))

    output_path = os.path.join(PROJECT_ROOT, f"gps_predictions_{dataset_name}.json")
    cv_avg = {m: float(np.mean(cv_results[m])) for m in cv_results}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": f"GPSBlock_{dataset_name}",
            "architecture": "GatedGCNv2 + MultiHeadAttention(spatial bias) + FFN + LayerScale + StochasticDepth",
            "config": HP, "n_genes": n_nodes,
            "cv_results": cv_avg,
            "test_metrics": cv_avg,  # for backward compat
            "predictions": sorted_preds,
        }, f, indent=2, ensure_ascii=False)

    import shutil
    shutil.copy(output_path, os.path.join(PROJECT_ROOT, "gps_predictions.json"))
    print(f"  [{ts()}] Saved to {output_path}")
    return cv_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CPDB')
    args = parser.parse_args()
    train_and_infer(args.dataset)
