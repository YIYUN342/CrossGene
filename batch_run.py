#!/usr/bin/env python
"""CrossGene batch run: all 8 datasets (train GPS + evaluate Agent)."""
import sys, os, gc, time, json
import numpy as np
import h5py
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

H5_DATA_DIR = os.environ.get("CROSSGENE_H5_DIR", "/root/autodl-tmp/dataset")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "batch_results.json")
RESULT_DIR_LOCAL = r"E:\work2026\kddWork\result"

DATASETS = ["CPDB", "LTG", "STRINGdb", "PCNET", "MTG", "MULTINET", "IREF", "IREF_2015"]
H5_FILE = {ds: os.path.join(H5_DATA_DIR, f"{ds}_multiomics.h5") for ds in DATASETS}

from datetime import datetime
def ts(): return datetime.now().strftime('%H:%M:%S')


def train_gps_for_dataset(dataset_name):
    """Train GPS on one dataset and save predictions JSON."""
    from gps.config import ModelConfig
    from gps.models.gps_model import GPS
    from gps.utils.node2vec import Node2vec

    h5_path = H5_FILE[dataset_name]
    if not os.path.exists(h5_path):
        print(f"  SKIP: H5 file not found: {h5_path}")
        return None

    print(f"  [{ts()}] Loading {dataset_name}...")
    with h5py.File(h5_path, 'r') as f:
        adj = f['network'][:]
        features = f['features'][:]
        train_mask = f['mask_train'][:].flatten().astype(bool)
        val_mask = f['mask_val'][:].flatten().astype(bool)
        test_mask = f['mask_test'][:].flatten().astype(bool)
        y_train = f['y_train'][:].flatten()
        y_test = f['y_test'][:].flatten()
        gene_names_raw = f['gene_names'][:]

    n_nodes = adj.shape[0]
    train_ids = np.where(train_mask)[0]
    val_ids = np.where(val_mask)[0]
    test_ids = np.where(test_mask)[0]

    # Decode gene names (use shared decoder that handles all formats)


    from tools.h5_reader import _decode_gene_names
    gene_symbols = _decode_gene_names(gene_names_raw)

    print(f"    Genes={n_nodes}, Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Build graph


    import networkx as nx
    edge_rows, edge_cols = np.where(adj > 0)
    n_edges = len(edge_rows)
    max_edges = 3_000_000
    if n_edges > max_edges:
        sample_idx = np.random.choice(n_edges, max_edges, replace=False)
        edge_rows, edge_cols = edge_rows[sample_idx], edge_cols[sample_idx]

    G = nx.Graph()
    G.add_edges_from(zip(edge_rows.tolist(), edge_cols.tolist()))

    # Subgraphs


    n_graphs, n_neighbors = 6, 8
    walks = Node2vec(graph=G, path_length=n_neighbors, num_paths=n_graphs,
                     workers=8, dw=True).get_walks()

    neighbor_id = np.zeros((n_nodes, n_graphs, n_neighbors), dtype=int)
    for w in walks:
        start = w[0][0]
        for g in range(n_graphs):
            neighbor_id[start, g, :] = w[g] if g < len(w) else w[0]
    missing = (neighbor_id.sum(axis=(1, 2)) == 0)
    for idx in np.where(missing)[0]:
        neighbor_id[idx, :, :] = idx

    spatial_matrix = np.zeros((n_nodes, n_graphs, 1, n_neighbors, n_neighbors), dtype=np.float32)
    for g in range(n_graphs):
        spatial_matrix[:, g, 0, :, :] = np.abs(
            neighbor_id[:, g, :, None] - neighbor_id[:, g, None, :]
        ).astype(np.float32) / max(n_nodes, 1)

    max_degree = int(np.max(np.sum(adj, axis=-1)) + 1)
    degree = np.expand_dims(np.sum(adj, axis=-1), axis=-1).astype(np.float32)

    # Config


    config = ModelConfig()
    config.dataset = dataset_name
    config.d_model = 64
    config.n_layers = 3
    config.n_graphs = n_graphs
    config.n_neighbors = n_neighbors
    config.max_degree = max_degree
    config.num_heads = 4
    config.d_sp_enc = 128
    config.dff = 128
    config.batch_size = 128
    config.n_epoch = 30
    config.lr = 0.001
    config.dropout = 0.5
    config.loss_mul = 0.27
    config.training = True
    config.callbacks_to_add = ['modelcheckpoint', 'earlystopping']
    config.contrast_weight = 0.1
    config.flow_weight = 0.05
    config.label_smoothing = 0.05
    config.aug_noise = 0.1
    config.aug_spatial = 0.15
    config.sp_enc_activation = 'relu'
    config.top_activation = 'relu'
    config.exp_name = f'GPS_{dataset_name}'
    config.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    config.checkpoint_dir = os.path.join(PROJECT_ROOT, 'gps', 'ckpt')
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    config.distance_matrix = degree
    config.node_feature = features.astype(np.float32)
    config.node_neighbor = neighbor_id
    config.spatial_matrix = spatial_matrix

    # Train


    print(f"    [{ts()}] Training GPS on {dataset_name}...")
    model = GPS(config)
    model.build_model()

    split = int(len(train_ids) * 0.8)
    tr_ids, vl_ids = train_ids[:split], train_ids[split:]

    t0 = time.time()
    model.fit(
        x_train=tr_ids,
        y_train=y_train[tr_ids].reshape(-1, 1).astype('float32'),
        x_val=vl_ids,
        y_val=y_train[vl_ids].reshape(-1, 1).astype('float32'),
    )
    train_time = time.time() - t0

    # Test eval


    auc, acc, prec, rec, f1, aupr, _, _ = model.score(
        x=test_ids, y=y_test[test_ids].astype('float32').reshape(-1, 1))
    print(f"    [{ts()}] GPS Test: AUC={auc:.4f}, AUPR={aupr:.4f}, F1={f1:.4f} ({train_time:.0f}s)")

    # Inference on all genes


    print(f"    [{ts()}] Inference on {n_nodes} genes...")
    all_scores = []
    for i in range(0, n_nodes, 512):
        all_scores.append(model.predict(np.arange(i, min(i+512, n_nodes), dtype=np.int32)))
    all_scores = np.concatenate(all_scores)

    # Save predictions


    predictions = {}
    for i in range(n_nodes):
        predictions[gene_symbols[i]] = {
            "gene_id": i, "gene_name": gene_symbols[i],
            "gps_score": float(all_scores[i]),
        }
    sorted_preds = dict(sorted(predictions.items(), key=lambda x: x[1]["gps_score"], reverse=True))

    output_path = os.path.join(PROJECT_ROOT, f"gps_predictions_{dataset_name}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": f"GPS_{dataset_name}",
            "architecture": "GatedGCN + DualBranchAttention + FFN",
            "n_genes": n_nodes,
            "test_metrics": {"auc": auc, "aupr": aupr, "f1": f1, "acc": acc},
            "predictions": sorted_preds,
        }, f, indent=2, ensure_ascii=False)

    # Also copy as generic fallback


    import shutil
    shutil.copy(output_path, os.path.join(PROJECT_ROOT, "gps_predictions.json"))

    del model, config
    gc.collect()
    tf.keras.backend.clear_session()

    return {
        "gps_auc": auc, "gps_aupr": aupr, "gps_f1": f1, "gps_acc": acc,
        "train_time_s": round(train_time), "n_nodes": n_nodes,
        "n_train": len(train_ids), "n_test": len(test_ids),
    }


def run_agent_for_dataset(dataset_name):
    """Run CrossGene Agent on one dataset and extract metrics."""
    import asyncio
    from workflow.graph import GeneWorkflow
    from core.config import Settings

    print(f"  [{ts()}] Running CrossGene Agent on {dataset_name}...")

    settings = Settings()
    workflow = GeneWorkflow(settings=settings)

    async def _run():
        goal = "Identify disease-associated genes"
        return await workflow.run(goal, max_iterations=1, forced_dataset=dataset_name)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        state = loop.run_until_complete(_run())
        loop.run_until_complete(workflow.shutdown())
        loop.close()
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

    # Extract metrics


    fab = state.get("fab_results", {})
    predictions = fab.get("predictions", {})
    ar = state.get("analysis_report", "")
    try:
        ar = json.loads(ar) if isinstance(ar, str) else ar
    except Exception:
        ar = {}

    ev = ar.get("evaluation", {})
    baseline = ev.get("baseline_llm_only", {})
    proposed = ev.get("proposed_ensemble", {})
    align = ar.get("alignment_summary", {})

    n_predicted = len(predictions)
    n_ensemble = len(fab.get("proposed_ensemble", {}))
    n_disagreements = sum(1 for g in ar.get("per_gene_analysis", [])
                          if abs((g.get("llm_prob") or 0.5) - (g.get("gps_score") or 0.5)) > 0.3)

    return {
        "n_predicted": n_predicted,
        "n_ensemble": n_ensemble,
        "baseline": {
            "auc": baseline.get("auc"), "aupr": baseline.get("aupr"),
            "acc": baseline.get("accuracy"), "precision": baseline.get("precision"),
            "recall": baseline.get("recall"), "f1": baseline.get("f1"),
        },
        "proposed": {
            "auc": proposed.get("auc"), "aupr": proposed.get("aupr"),
            "acc": proposed.get("accuracy"), "precision": proposed.get("precision"),
            "recall": proposed.get("recall"), "f1": proposed.get("f1"),
        },
        "pubmed_concordance": align.get("concordance_rate"),
        "n_concordant": align.get("concordant"),
        "n_discordant": align.get("discordant", 0),
        "n_disagreements": n_disagreements,
    }


def main():
    print(f"[{ts()}] ========== CrossGene Batch Run: {len(DATASETS)} Datasets ==========")

    all_results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} datasets already completed")

    for ds in DATASETS:
        if ds in all_results and all_results[ds].get("agent"):
            print(f"\n{'='*60}")
            print(f"[{ds}] Already done, skipping")
            print(f"{'='*60}")
            continue

        print(f"\n{'='*60}")
        print(f"[{ts()}] DATASET: {ds}")
        print(f"{'='*60}")

        result = {"dataset": ds}

        # Step 1: Train GPS


        if ds not in all_results or not all_results[ds].get("gps"):
            gps_metrics = train_gps_for_dataset(ds)
            if gps_metrics:
                result["gps"] = gps_metrics
        else:
            result["gps"] = all_results[ds]["gps"]
            print(f"  GPS already trained, reusing")

        # Step 2: Run Agent


        agent_metrics = run_agent_for_dataset(ds)
        if agent_metrics:
            result["agent"] = agent_metrics

        all_results[ds] = result

        # Save after each dataset


        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"  [{ts()}] {ds} complete!")
        gc.collect()

    # Summary table


    print(f"\n\n{'='*100}")
    print(f"[{ts()}] CROSSGENE BATCH RESULTS SUMMARY")
    print(f"{'='*100}")

    header = f"{'Dataset':<12s} {'GPS-AUC':>8s} {'GPS-AUPR':>8s} {'GPS-F1':>8s} | {'LLM-AUC':>8s} {'LLM-F1':>8s} | {'ENS-AUC':>8s} {'ENS-AUPR':>8s} {'ENS-F1':>8s} {'ENS-Acc':>8s} | {'ΔAUC':>8s}"
    print(header)
    print("-" * len(header))

    for ds in DATASETS:
        r = all_results.get(ds, {})
        gps = r.get("gps", {}) or {}
        agent = r.get("agent", {}) or {}
        baseline = agent.get("baseline", {}) or {}
        proposed = agent.get("proposed", {}) or {}

        g_auc = gps.get("gps_auc") or 0
        g_aupr = gps.get("gps_aupr") or 0
        g_f1 = gps.get("gps_f1") or 0
        l_auc = baseline.get("auc") or 0
        l_f1 = baseline.get("f1") or 0
        e_auc = proposed.get("auc") or 0
        e_aupr = proposed.get("aupr") or 0
        e_f1 = proposed.get("f1") or 0
        e_acc = proposed.get("acc") or 0
        delta = (e_auc or 0) - (l_auc or 0)

        print(f"{ds:<12s} {g_auc:>8.4f} {g_aupr:>8.4f} {g_f1:>8.4f} | {l_auc:>8.4f} {l_f1:>8.4f} | {e_auc:>8.4f} {e_aupr:>8.4f} {e_f1:>8.4f} {e_acc:>8.4f} | {delta:>+8.4f}")

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
