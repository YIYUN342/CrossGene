"""H5 multi-omics data reader for GeneAgent. Reads local gene network datasets."""
import os
from typing import Dict, Any
import numpy as np

H5_DATA_DIR = os.environ.get("CROSSGENE_H5_DIR", r"D:\pyProject\GPS\dataset\networks")
DATASET_MAP = {
    'CPDB': 'CPDB_multiomics.h5', 'PCNET': 'PCNET_multiomics.h5',
    'MULTINET': 'MULTINET_multiomics.h5', 'IREF': 'IREF_multiomics.h5',
    'IREF_2015': 'IREF_2015_multiomics.h5', 'STRINGdb': 'STRINGdb_multiomics.h5',
    'LTG': 'LTG_multiomics.h5', 'MTG': 'MTG_multiomics.h5',
}


def _decode_gene_names(gene_names_raw) -> list[str]:
    """Decode gene names from H5 dataset, handling both formats:

    CPDB-style: shape (N, 2) — each row is [ENSG_ID, SYMBOL] (bytes)
    LTG-style:  shape (N,)  — each row is an array of ASCII int codes

    Returns list of gene symbol strings.
    """
    symbols = []
    first = gene_names_raw[0]

    # Detect format


    if hasattr(first, '__len__') and not isinstance(first, (bytes, str)):
        arr_len = len(first)
        # CPDB-style: first row is [ENSG_ID_bytes, SYMBOL_bytes]


        if arr_len == 2 and isinstance(first[0], (bytes, str)):
            for row in gene_names_raw:
                s = row[1]
                symbols.append(s.decode() if isinstance(s, bytes) else str(s))
            return symbols
        # LTG-style: first row is a list/array of ASCII codes (ints)


        if arr_len >= 2 and isinstance(first[0], (int, np.integer)):
            for row in gene_names_raw:
                chars = [chr(int(c)) for c in row]
                symbols.append("".join(chars))
            return symbols

    # Other formats: try direct decode


    for row in gene_names_raw:
        if hasattr(row, 'decode'):
            symbols.append(row.decode())
        elif isinstance(row, np.ndarray):
            chars = [chr(int(c)) for c in row]
            symbols.append("".join(chars))
        else:
            symbols.append(str(row))
    return symbols

def list_datasets() -> Dict[str, Any]:
    results = {}
    for ds, fname in DATASET_MAP.items():
        path = os.path.join(H5_DATA_DIR, fname)
        results[ds] = {'file_exists': os.path.exists(path),
                       'file_size_mb': round(os.path.getsize(path)/(1024*1024),1) if os.path.exists(path) else 0}
    return {'total': len(results), 'ready': sum(1 for v in results.values() if v['file_exists']), 'datasets': results}

def get_dataset_summary(dataset_name: str) -> Dict[str, Any]:
    import h5py, numpy as np
    if dataset_name not in DATASET_MAP: return {'status': 'error', 'message': f'Unknown: {dataset_name}'}
    path = os.path.join(H5_DATA_DIR, DATASET_MAP[dataset_name])
    if not os.path.exists(path): return {'status': 'error', 'message': f'File not found'}
    with h5py.File(path, 'r') as f:
        net_ds = f['network']
        feat = f['features'][:]
        yt, tm = f['y_train'][:], f['mask_train'][:]
        n = net_ds.shape[0]
        # compute edges in chunks


        chunk_size = 4096
        e = 0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = net_ds[start:end, :]
            e += int(np.count_nonzero(chunk))
        nt, npt = int(np.sum(tm)), int(np.sum(yt[tm == 1] > 0))
    return {'status': 'success', 'dataset_name': dataset_name, 'n_genes': n, 'n_edges': e,
            'network_density': round(e/(n*n),6) if n>0 else 0, 'feature_dim': feat.shape[1],
            'label_distribution': {'train': {'total': nt, 'positive': npt,
            'positive_ratio': round(npt/nt,4) if nt>0 else 0}}}

def query_gene_network(dataset_name: str, gene_id: int = 0, top_k: int = 5) -> Dict[str, Any]:
    import h5py, numpy as np
    path = os.path.join(H5_DATA_DIR, DATASET_MAP[dataset_name])
    with h5py.File(path, 'r') as f:
        net_ds = f['network']
        feat = f['features'][:]
        n = net_ds.shape[0]
        if gene_id < 0 or gene_id >= n: return {'status': 'error', 'message': f'gene_id {gene_id} out of range'}
        row = net_ds[gene_id, :]  # read single row
        deg = int(np.count_nonzero(row))
        gf = feat[gene_id]
        ni = np.where(row > 0)[0]; nw = row[ni]
        si = np.argsort(nw)[::-1][:top_k]
        name = None
        if 'gene_names' in f:
            gene_names_raw = f['gene_names'][:]
            all_symbols = _decode_gene_names(gene_names_raw)
            if gene_id < len(all_symbols):
                name = all_symbols[gene_id]
    return {'status': 'success', 'dataset_name': dataset_name, 'gene_id': gene_id,
            'gene_name': name, 'gene_degree': deg, 'feature_mean': float(np.mean(gf)),
            'top_neighbors': [{'gene_id': int(ni[i]), 'edge_weight': float(nw[i])} for i in si]}


def select_candidate_genes(
    dataset_name: str = "CPDB",
    n_test_pos: int | None = None,
    n_test_neg: int | None = None,
    n_hub: int = 20,
    include_neighbors: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Select candidate genes from network for LLM prediction.

    Returns:
      1. ALL test set genes (WITH ground truth labels → AUC evaluation)
         If n_test_pos/n_test_neg is None, uses ALL available test genes.
      2. Top hub genes not in train/val/test (discovery candidates)
      3. Per-gene network context: degree, tier, top-N neighbor names+weights
    """
    import h5py, numpy as np

    if dataset_name not in DATASET_MAP:
        return {'status': 'error', 'message': f'Unknown dataset: {dataset_name}'}
    path = os.path.join(H5_DATA_DIR, DATASET_MAP[dataset_name])
    if not os.path.exists(path):
        return {'status': 'error', 'message': f'File not found: {path}'}

    np.random.seed(random_seed)
    with h5py.File(path, 'r') as f:
        net_ds = f['network']      # don't load entire matrix — too big
        n_genes = net_ds.shape[0]
        chunk_size = 4096          # read in chunks to compute degrees

        # Compute degrees in chunks (avoids loading entire network)


        degrees = np.zeros(n_genes, dtype=np.int64)
        for start in range(0, n_genes, chunk_size):
            end = min(start + chunk_size, n_genes)
            chunk = net_ds[start:end, :]  # read rows
            degrees[start:end] = np.count_nonzero(chunk, axis=1)

        gene_names_raw = f['gene_names'][:]
        gene_symbols = _decode_gene_names(gene_names_raw)

        mask_train = f['mask_train'][:].flatten().astype(bool)
        mask_val = f['mask_val'][:].flatten().astype(bool)
        mask_test = f['mask_test'][:].flatten().astype(bool)
        y_test = f['y_test'][:].flatten()

        # 1. Test set genes (ALL by default, or sampled if limits set)


        test_indices = np.where(mask_test)[0]
        test_pos_all = test_indices[y_test[test_indices] == 1]
        test_neg_all = test_indices[y_test[test_indices] == 0]

        n_pos = len(test_pos_all) if n_test_pos is None else min(n_test_pos, len(test_pos_all))
        n_neg = len(test_neg_all) if n_test_neg is None else min(n_test_neg, len(test_neg_all))
        chosen_pos = np.random.choice(test_pos_all, n_pos, replace=False)
        chosen_neg = np.random.choice(test_neg_all, n_neg, replace=False)

        # 2. Top hub genes NOT in train/val/test (discovery)


        used = mask_train | mask_val | mask_test
        unused_indices = np.where(~used)[0]
        unused_degrees = degrees[unused_indices]
        hub_order = np.argsort(unused_degrees)[::-1]
        top_hub_indices = unused_indices[hub_order[:n_hub]]

        def _entry(idx):
            deg = int(degrees[idx])
            tier = ('super-hub' if deg >= 100 else
                    'hub' if deg >= 37 else
                    'mid' if deg >= 17 else
                    'peripheral')
            # Get top-N neighbors — read single row from H5 (lazy)


            neighbors = []
            if deg > 0:
                row = net_ds[idx, :]  # read just this row
                nz = np.where(row > 0)[0]
                nz_weights = row[nz]
                top_k = np.argsort(nz_weights)[::-1][:include_neighbors]
                for ni in top_k:
                    nid = int(nz[ni])
                    nw = float(nz_weights[ni])
                    nname = gene_symbols[nid] if nid < len(gene_symbols) else '?'
                    neighbors.append({'gene_name': nname, 'weight': round(nw, 4)})
            return {
                'gene_id': int(idx),
                'gene_name': gene_symbols[idx],
                'gene_degree': deg,
                'degree_tier': tier,
                'top_neighbors': neighbors,
            }

        candidates = []
        for idx in chosen_pos:
            e = _entry(idx); e['ground_truth'] = 1; e['source'] = 'test_positive'
            candidates.append(e)
        for idx in chosen_neg:
            e = _entry(idx); e['ground_truth'] = 0; e['source'] = 'test_negative'
            candidates.append(e)
        for idx in top_hub_indices:
            e = _entry(idx); e['ground_truth'] = None; e['source'] = 'hub_discovery'
            candidates.append(e)

        np.random.shuffle(candidates)

    return {
        'status': 'success', 'dataset_name': dataset_name,
        'total_genes_in_network': n_genes,
        'total_edges': int(degrees.sum()),
        'network_mean_degree': float(degrees.mean()),
        'test_set_total': len(test_indices),
        'test_set_positive_available': len(test_pos_all),
        'test_set_negative_available': len(test_neg_all),
        'candidates_selected': len(candidates),
        'n_test_genes': n_pos + n_neg,
        'n_hub_genes': n_hub,
        'candidates': candidates,
        'selection_method': (
            f'All test set genes ({n_pos} pos + {n_neg} neg) '
            f'+ top {n_hub} hubs from {len(unused_indices):,} unlabeled genes. '
            f'Each gene includes degree, tier, and top-{include_neighbors} neighbors.'
        ),
    }


GENE_DATA_TOOLS = [
    {"type":"function","function":{"name":"list_available_datasets","description":"List available gene network datasets with file status","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"get_dataset_summary","description":"Get summary: gene count, edges, density, feature dim, label dist","parameters":{"type":"object","properties":{"dataset_name":{"type":"string","description":"e.g., CPDB, STRINGdb"}},"required":["dataset_name"]}}},
    {"type":"function","function":{"name":"query_gene_network","description":"Query a gene: degree, feature stats, top neighbors","parameters":{"type":"object","properties":{"dataset_name":{"type":"string"},"gene_id":{"type":"integer"},"top_k":{"type":"integer"}},"required":["dataset_name","gene_id"]}}},
    {"type":"function","function":{"name":"select_candidate_genes","description":"Select ALL test set genes + top hub discovery genes from the network. Each gene includes degree, degree tier, and top-5 neighbor genes for network context. Use default parameters to get all available test genes.","parameters":{"type":"object","properties":{"dataset_name":{"type":"string","description":"e.g., CPDB"},"n_hub":{"type":"integer","description":"Number of hub discovery genes (default 20)"}},"required":["dataset_name"]}}},
]
