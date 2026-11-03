#!/usr/bin/env python
"""GPS: Train on full CPDB training set + inference on all 13,627 genes.
Output: gps_predictions_CPDB.json

Usage: python inference/gps_inference.py
"""
import sys, os, gc, time, json
import numpy as np
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Project root


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

H5_DATA_DIR = os.environ.get("CROSSGENE_H5_DIR", r"D:\pyProject\GPS\dataset\networks")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'gps_predictions_CPDB.json')
DATASET = 'CPDB'
H5_PATH = os.path.join(H5_DATA_DIR, f'{DATASET}_multiomics.h5')

from datetime import datetime
def ts(): return datetime.now().strftime('%H:%M:%S')

print(f"[{ts()}] ====== CrossGene GPS Full CPDB Training + Inference ======")
print(f"  Project: {PROJECT_ROOT}")
print(f"  H5 Data: {H5_DATA_DIR}")

# Step 1: Load CPDB data


print(f"[{ts()}] Loading CPDB data...")
with h5py.File(H5_PATH, 'r') as f:
    adj = f['network'][:]
    features = f['features'][:]
    y_train = f['y_train'][:].flatten()
    y_val = f['y_val'][:].flatten()
    y_test = f['y_test'][:].flatten()
    train_mask = f['mask_train'][:].flatten().astype(bool)
    val_mask = f['mask_val'][:].flatten().astype(bool)
    test_mask = f['mask_test'][:].flatten().astype(bool)
    gene_names_raw = f['gene_names'][:]

n_nodes = adj.shape[0]
print(f"  Genes: {n_nodes}, Features: {features.shape}")

# Decode gene names


gene_symbols = []
for row in gene_names_raw:
    s = row[1]
    gene_symbols.append(s.decode() if isinstance(s, bytes) else str(s))

train_ids = np.where(train_mask)[0]
val_ids = np.where(val_mask)[0]
test_ids = np.where(test_mask)[0]
print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

# Step 2: Build subgraphs (node2vec random walks)


print(f"[{ts()}] Building subgraphs...")
import networkx as nx

# Edge sampling: take top edges by weight to keep memory manageable


edge_rows, edge_cols = np.where(adj > 0)
n_edges = len(edge_rows)
print(f"  Total edges: {n_edges:,}")

# Build graph (sample if too large)


max_edges = 5_000_000
if n_edges > max_edges:
    sample_idx = np.random.choice(n_edges, max_edges, replace=False)
    edge_rows = edge_rows[sample_idx]
    edge_cols = edge_cols[sample_idx]

G = nx.Graph()
edges = list(zip(edge_rows.tolist(), edge_cols.tolist()))
G.add_edges_from(edges)
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Node2vec walks


from gps.config import ModelConfig
from gps.models.gps_model import GPS
from gps.utils.node2vec import Node2vec
n_graphs = 6
n_neighbors = 8
print(f"  Generating {n_graphs} walks of length {n_neighbors}...")

walks = Node2vec(
    graph=G, path_length=n_neighbors, num_paths=n_graphs,
    workers=8, dw=True
).get_walks()

neighbor_id = np.zeros((n_nodes, n_graphs, n_neighbors), dtype=int)
for w in walks:
    start_node = w[0][0]
    for g in range(n_graphs):
        neighbor_id[start_node, g, :] = w[g] if g < len(w) else w[0]

# Fill missing nodes with self-loops


missing_mask = (neighbor_id.sum(axis=(1, 2)) == 0)
for idx in np.where(missing_mask)[0]:
    neighbor_id[idx, :, :] = idx
print(f"  Self-loop fill: {missing_mask.sum()} nodes had no walks")

# Spatial matrix


print(f"  Building spatial matrix...")
spatial_matrix = np.zeros((n_nodes, n_graphs, 1, n_neighbors, n_neighbors), dtype=np.float32)
for g in range(n_graphs):
    spatial_matrix[:, g, 0, :, :] = np.abs(
        neighbor_id[:, g, :, None] - neighbor_id[:, g, None, :]
    ).astype(np.float32) / max(n_nodes, 1)

max_degree = int(np.max(np.sum(adj, axis=-1)) + 1)
degree = np.expand_dims(np.sum(adj, axis=-1), axis=-1).astype(np.float32)

# Step 3: Build and train GPS model


print(f"[{ts()}] Building GPS model...")
import tensorflow as tf

config = ModelConfig()
config.dataset = DATASET
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
config.exp_name = 'GPS_CPDB_full'
config.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
config.checkpoint_dir = '/root/GPS/ckpt'

config.distance_matrix = degree
config.node_feature = features.astype(np.float32)
config.node_neighbor = neighbor_id
config.spatial_matrix = spatial_matrix

model = GPS(config)
model.build_model()

# Train on 80% of training set, validate on 20%


split = int(len(train_ids) * 0.8)
tr_ids, vl_ids = train_ids[:split], train_ids[split:]
y_tr = y_train[tr_ids].reshape(-1, 1)  # shape (N, 1) for weighted_cross_entropy
y_vl = y_train[vl_ids].reshape(-1, 1)

print(f"[{ts()}] Training GPS ({config.n_epoch} epochs, batch={config.batch_size})...")
print(f"  Train samples: {len(tr_ids)}, Val samples: {len(vl_ids)}")

t0 = time.time()
model.fit(
    x_train=tr_ids,
    y_train=y_tr.astype('float32'),
    x_val=vl_ids,
    y_val=y_vl.astype('float32'),
)
train_time = time.time() - t0
print(f"[{ts()}] Training completed in {train_time:.0f}s ({train_time/60:.1f}min)")

# Test evaluation


print(f"[{ts()}] Test set evaluation...")
auc, acc, prec, rec, f1, aupr, _, _ = model.score(
    x=test_ids,
    y=y_test[test_ids].astype('float32').reshape(-1, 1),
)
print(f"  Test: AUC={auc:.4f}  AUPR={aupr:.4f}  F1={f1:.4f}  Acc={acc:.4f}")

# Step 4: Inference on ALL genes


print(f"[{ts()}] Running inference on all {n_nodes} genes...")
all_indices = np.arange(n_nodes, dtype=np.int32)

# Predict in batches


batch_size = 512
all_scores = []
for i in range(0, n_nodes, batch_size):
    batch = all_indices[i:i+batch_size]
    scores = model.predict(batch)
    all_scores.append(scores)
    if (i // batch_size) % 10 == 0:
        print(f"  Inference: {i}/{n_nodes}")

all_scores = np.concatenate(all_scores)

# Step 5: Export predictions


print(f"[{ts()}] Exporting predictions...")

predictions = {}
for i in range(n_nodes):
    predictions[gene_symbols[i]] = {
        "gene_id": i,
        "gene_name": gene_symbols[i],
        "gps_score": float(all_scores[i]),
    }

sorted_preds = dict(
    sorted(predictions.items(), key=lambda x: x[1]["gps_score"], reverse=True)
)

output = {
    "model": "GPS_CPDB_full",
    "architecture": "GatedGCN + DualBranchAttention + FFN (GPSBlock × 3)",
    "n_genes": n_nodes,
    "test_metrics": {"auc": auc, "aupr": aupr, "f1": f1, "acc": acc},
    "predictions": sorted_preds,
    "top_20": [
        {"gene_name": k, "gps_score": v["gps_score"]}
        for k, v in list(sorted_preds.items())[:20]
    ],
}

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"[{ts()}] Saved {n_nodes} predictions to {OUTPUT_PATH}")

# Show top 20


print(f"\n  Top 20 GPS predictions:")
for i, (sym, pred) in enumerate(list(sorted_preds.items())[:20]):
    print(f"  {i+1:3d}. {sym:<12s}  gps_score={pred['gps_score']:.4f}")

# Also save a copy to gps_predictions.json (fallback)


backup_path = '/root/GeneAgent/gps_predictions.json'
with open(backup_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"[{ts()}] Backup saved to {backup_path}")

# Cleanup


del model, config
gc.collect()
tf.keras.backend.clear_session()
print(f"[{ts()}] DONE!")
