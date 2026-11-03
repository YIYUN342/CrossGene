#!/usr/bin/env python
"""CrossGene GPS Block Batch: 8 datasets, full pipeline."""
import sys, os, gc, json
import numpy as np, h5py
import tensorflow as tf
import asyncio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DATA_DIR = os.environ.get("CROSSGENE_H5_DIR", "/root/autodl-tmp/dataset")
PDATA_DIR = os.environ.get("GPS_PDATA_DIR", "/root/autodl-tmp/pdata")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "batch_gpsblock_results.json")

DATASET_CFG = {
    'CPDB':       ('CPDB_multiomics.h5',       8, 9,  6, 8),
    'STRINGdb':   ('STRINGdb_multiomics.h5',    8, 9,  6, 8),
    'LTG':        ('LTG_multiomics.h5',        10, 14, 6, 8),
    'PCNET':      ('PCNET_multiomics.h5',      10, 14, 6, 8),
    'MTG':        ('MTG_multiomics.h5',        10, 14, 6, 8),
    'MULTINET':   ('MULTINET_multiomics.h5',    8, 9,  6, 8),
    'IREF':       ('IREF_multiomics.h5',        8, 10, 6, 8),
    'IREF_2015':  ('IREF_2015_multiomics.h5',   8, 9,  6, 8),
}

HP = dict(d_model=64, n_layers=3, num_heads=4, d_sp_enc=128, dff=128,
          dropout=0.3, loss_mul=0.35, lr=0.002, batch_size=64, n_epoch=150,
          contrast_weight=0.1, flow_weight=0.05, label_smoothing=0.05,
          aug_noise=0.1, aug_spatial=0.15, n_folds=5)

from datetime import datetime
def ts(): return datetime.now().strftime('%H:%M:%S')


def train_gps_block(ds):
    """Train GPS Block for one dataset."""
    h5_file, src_G, src_N, tgt_G, tgt_N = DATASET_CFG[ds]
    h5_path = os.path.join(DATA_DIR, h5_file)
    if not os.path.exists(h5_path):
        print(f"  SKIP: {h5_path}")
        return None

    print(f"\n  [{ts()}] Loading {ds}...")
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
    train_val_mask = np.logical_or(train_mask, val_mask) if val_mask is not None else train_mask
    label_idx = np.where(train_val_mask)[0]

    from tools.h5_reader import _decode_gene_names
    gene_symbols = _decode_gene_names(gene_names_raw)

    # Load subgraphs


    sub_file = f'{PDATA_DIR}/{ds}_multiomics_method_rw_channel_{src_G}_neighbor_{src_N}_subgraphs_p_4_q_4.npy'
    spa_file = f'{PDATA_DIR}/{ds}_multiomics_method_rw_channel_{src_G}_neighbor_{src_N}_spatial_p_4_q_4.npy'

    neighbor_id = np.load(sub_file, allow_pickle=True)[:, :tgt_G, :tgt_N]
    spatial_matrix = np.load(spa_file, allow_pickle=True)[:, :tgt_G, :, :tgt_N, :tgt_N]

    max_degree = int(np.max(np.sum(adj, axis=-1)) + 1)
    degree = np.expand_dims(np.sum(adj, axis=-1), axis=-1).astype(np.float32)

    from gps.config_improved import ImprovedModelConfig
    from gps.models.gps_improved import ImprovedGPS

    tf.config.optimizer.set_jit(True)

    # 5-Fold CV


    from sklearn.model_selection import StratifiedKFold
    all_y = y_train.copy()
    if val_mask is not None:
        all_y[val_mask] = y_val[val_mask]

    kf = StratifiedKFold(n_splits=HP['n_folds'], shuffle=True, random_state=100)
    k_sets = list(kf.split(label_idx, all_y[label_idx]))
    cv_results = {"auc": [], "aupr": [], "acc": [], "prec": [], "rec": [], "f1": []}

    for i, (tr_idx, vl_idx) in enumerate(k_sets):
        tr_id, vl_id = label_idx[tr_idx], label_idx[vl_idx]
        y_tr, y_vl = all_y[tr_id], all_y[vl_id]
        print(f"  Fold {i+1}/{HP['n_folds']}: train={len(tr_id)} val={len(vl_id)}")

        config = ImprovedModelConfig()
        for k, v in HP.items():
            if hasattr(config, k):
                setattr(config, k, v)
        config.n_graphs = tgt_G; config.n_neighbors = tgt_N; config.max_degree = max_degree
        config.training = True; config.dataset = ds
        config.callbacks_to_add = ['modelcheckpoint']
        config.optimizer = tf.keras.optimizers.Adam(learning_rate=HP['lr'])
        config.distance_matrix = degree; config.node_feature = features.astype(np.float32)
        config.node_neighbor = neighbor_id; config.spatial_matrix = spatial_matrix
        config.exp_name = f'GPSBlock_{ds}_fold{i}'
        config.checkpoint_dir = os.path.join(PROJECT_ROOT, 'gps', 'ckpt')
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        model = ImprovedGPS(config)
        model.build_model()
        model.fit(x_train=tr_id, y_train=y_tr.astype('float32').reshape(-1, 1),
                  x_val=vl_id, y_val=y_vl.astype('float32').reshape(-1, 1))
        model.load_best_model()
        auc, acc, prec, rec, f1, aupr, _, _ = model.score(
            x=test_id, y=y_test[test_id].astype('float32').reshape(-1, 1))
        cv_results["auc"].append(auc); cv_results["aupr"].append(aupr)
        cv_results["acc"].append(acc); cv_results["prec"].append(prec)
        cv_results["rec"].append(rec); cv_results["f1"].append(f1)
        del model; gc.collect(); tf.keras.backend.clear_session()

    cv_avg = {m: round(float(np.mean(v)), 4) for m, v in cv_results.items()}
    cv_std = {m: round(float(np.std(v)), 4) for m, v in cv_results.items()}
    print(f"  CV: AUC={cv_avg['auc']}±{cv_std['auc']}, F1={cv_avg['f1']}±{cv_std['f1']}")

    # Final model


    print(f"  Training FINAL model on {len(label_idx)} genes...")
    config = ImprovedModelConfig()
    for k, v in HP.items():
        if hasattr(config, k): setattr(config, k, v)
    config.n_graphs = tgt_G; config.n_neighbors = tgt_N; config.max_degree = max_degree
    config.training = True; config.dataset = ds
    config.callbacks_to_add = ['modelcheckpoint']
    config.optimizer = tf.keras.optimizers.Adam(learning_rate=HP['lr'])
    config.distance_matrix = degree; config.node_feature = features.astype(np.float32)
    config.node_neighbor = neighbor_id; config.spatial_matrix = spatial_matrix
    config.exp_name = f'GPSBlock_{ds}_FINAL'
    config.checkpoint_dir = os.path.join(PROJECT_ROOT, 'gps', 'ckpt')
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    model = ImprovedGPS(config); model.build_model()
    model.fit(x_train=label_idx, y_train=all_y[label_idx].astype('float32').reshape(-1, 1),
              x_val=test_id, y_val=y_test[test_id].astype('float32').reshape(-1, 1))
    model.load_best_model()

    # Inference


    all_scores = []
    for i in range(0, n_nodes, 512):
        all_scores.append(model.predict(np.arange(i, min(i+512, n_nodes), dtype=np.int32)))
    all_scores = np.concatenate(all_scores)

    predictions = {}
    for i in range(n_nodes):
        predictions[gene_symbols[i]] = {"gene_id": i, "gene_name": gene_symbols[i],
                                         "gps_score": float(all_scores[i])}
    sorted_preds = dict(sorted(predictions.items(), key=lambda x: x[1]["gps_score"], reverse=True))

    output_path = os.path.join(PROJECT_ROOT, f"gps_predictions_{ds}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"model": f"GPSBlock_{ds}", "n_genes": n_nodes,
                   "cv_results": cv_avg, "predictions": sorted_preds}, f, indent=2, ensure_ascii=False)
    import shutil
    shutil.copy(output_path, os.path.join(PROJECT_ROOT, "gps_predictions.json"))

    del model; gc.collect(); tf.keras.backend.clear_session()
    return cv_avg


def run_agent(ds):
    """Run CrossGene Agent."""
    from workflow.graph import GeneWorkflow
    from core.config import Settings

    workflow = GeneWorkflow(settings=Settings())

    async def _run():
        return await workflow.run("Identify disease-associated genes",
                                  max_iterations=1, forced_dataset=ds)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    state = loop.run_until_complete(_run())
    loop.run_until_complete(workflow.shutdown())
    loop.close()

    fab = state.get("fab_results", {})
    ar = state.get("analysis_report", "")
    try:
        ar = json.loads(ar) if isinstance(ar, str) else ar
    except:
        ar = {}
    ev = ar.get("evaluation", {})
    base = ev.get("baseline_llm_only", {})
    prop = ev.get("proposed_ensemble", {})
    return {
        "baseline": {k: base.get(k) for k in ["auc","aupr","accuracy","precision","recall","f1"]},
        "proposed": {k: prop.get(k) for k in ["auc","aupr","accuracy","precision","recall","f1"]},
    }


def main():
    datasets = list(DATASET_CFG.keys())
    all_results = {}
    if os.path.exists(RESULTS_FILE):
        all_results = json.load(open(RESULTS_FILE))

    print(f"[{ts()}] ====== GPS Block Batch: {len(datasets)} datasets ======")

    for ds in datasets:
        if ds in all_results and all_results[ds].get("agent"):
            print(f"\n[{ds}] Done, skip")
            continue

        print(f"\n{'='*60}")
        print(f"[{ts()}] {ds}")
        print(f"{'='*60}")

        result = {"dataset": ds}

        # GPS Block


        if ds not in all_results or not all_results[ds].get("gps"):
            gps = train_gps_block(ds)
            if gps:
                result["gps"] = gps
        else:
            result["gps"] = all_results[ds]["gps"]

        # Agent


        print(f"  [{ts()}] Running Agent on {ds}...")
        agent = run_agent(ds)
        if agent:
            result["agent"] = agent

        all_results[ds] = result
        json.dump(all_results, open(RESULTS_FILE, 'w'), indent=2, ensure_ascii=False)
        print(f"  [{ts()}] {ds} DONE!")

    # Summary


    print(f"\n{'='*80}")
    header = f"{'Dataset':<12} {'GPS-AUC':>8} {'GPS-F1':>8} | {'LLM-AUC':>8} {'LLM-F1':>8} | {'ENS-AUC':>8} {'ENS-F1':>8}"
    print(header)
    print("-" * len(header))
    for ds in datasets:
        r = all_results.get(ds, {})
        g = r.get("gps", {}) or {}
        a = r.get("agent", {}) or {}
        b = a.get("baseline", {}) or {}
        p = a.get("proposed", {}) or {}
        print(f"{ds:<12} {g.get('auc',0):>8.4f} {g.get('f1',0):>8.4f} | "
              f"{b.get('auc',0):>8.4f} {b.get('f1',0):>8.4f} | "
              f"{p.get('auc',0):>8.4f} {p.get('f1',0):>8.4f}")

    print(f"\nResults: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
