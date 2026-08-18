import json
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PCA_DIR = ROOT / "phase1_pca"
N_PCS_FOR_CLUSTERING = 20
VARIANT = "covariance_all_dims"
CONDITION = "joint"

with open(ROOT / "data" / "rank_lookup_row_indices.json") as f:
    rank_rows = json.load(f)

# All 19 real binders with their row index (None for b44, freshly embedded, not in the 80k set)
LABELED = {
    "p130": (2007, 7.26), "p144": (16250, 9.72), "p154_prev": (30263, 6.44), "b44": (None, 4.99),
    "new154": (rank_rows["rank0016_design_spec_154"], 0.88),
    "new149": (rank_rows["rank0034_design_spec_149"], 3.10),
    "new241": (rank_rows["rank0019_design_spec_241"], 4.92),
    "new226": (rank_rows["rank0004_design_spec_226"], 1.05),
    "new146": (rank_rows["rank0037_design_spec_146"], 2.72),
    "new208": (rank_rows["rank0003_design_spec_208"], 3.38),
    "new211": (rank_rows["rank0009_design_spec_211"], 0.92),
    "new249": (rank_rows["rank0056_design_spec_249"], 0.64),
    "new089": (rank_rows["rank0077_design_spec_089"], 5.12),
    "new174": (rank_rows["rank0002_design_spec_174"], 1.62),
    "new248": (rank_rows["rank0057_design_spec_248"], 2.80),
    "new163": (rank_rows["rank0054_design_spec_163"], 1.41),
    "new108": (rank_rows["rank0005_design_spec_108"], 4.03),
    "new228": (rank_rows["rank0001_design_spec_228"], 2.72),
    "new157": (rank_rows["rank0014_design_spec_157"], 9.19),
}

for layer in [7, 8, 9]:
    print(f"\n{'=' * 70}\nlayer {layer}\n{'=' * 70}")
    layer_dir = PCA_DIR / CONDITION / f"layer_{layer:02d}"
    scores = np.load(layer_dir / f"{VARIANT}_scores.npy")[:, :N_PCS_FOR_CLUSTERING]
    scores_std = StandardScaler().fit_transform(scores)

    clusterer = HDBSCAN(min_cluster_size=100, min_samples=25)
    labels = clusterer.fit_predict(scores_std)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"found {n_clusters} clusters, {n_noise} noise points ({n_noise / len(labels):.1%} of 80k)")

    cluster_sizes = {c: (labels == c).sum() for c in set(labels) if c != -1}
    print(f"cluster sizes: {sorted(cluster_sizes.values(), reverse=True)[:10]}")

    print(f"\nlabeled real binders -> cluster assignment:")
    for name, (row_idx, mcherry) in LABELED.items():
        if row_idx is None:
            print(f"  {name:12s} mCherry={mcherry:5.2f}%  cluster=N/A (not in 80k set, no HDBSCAN assignment)")
        else:
            c = labels[row_idx]
            print(f"  {name:12s} mCherry={mcherry:5.2f}%  cluster={c}")
