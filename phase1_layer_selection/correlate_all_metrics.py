import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "boltzgen_conf_1_batch_generation-scores.csv"
IDS_PATH = ROOT / "data" / "phase1_mean_pooled" / "ids.csv"
PCA_DIR = ROOT / "data" / "phase1_pca"
N_PCS_TO_CHECK = 20

INTERFACE_METRICS = {
    "iptm": "design_to_target_iptm - boltzgen",
    "min_pae": "min_design_to_target_pae - boltzgen",
    "hbonds": "plip_hbonds_refolded - boltzgen",
    "delta_sasa": "delta_sasa_refolded - boltzgen",
    "bindsite_3rmsd": "bindsite_under_3rmsd - boltzgen",
    "bindsite_4rmsd": "bindsite_under_4rmsd - boltzgen",
    "bindsite_5rmsd": "bindsite_under_5rmsd - boltzgen",
    "bindsite_6rmsd": "bindsite_under_6rmsd - boltzgen",
    "bindsite_7rmsd": "bindsite_under_7rmsd - boltzgen",
    "bindsite_8rmsd": "bindsite_under_8rmsd - boltzgen",
    "bindsite_9rmsd": "bindsite_under_9rmsd - boltzgen",
    "iiptm": "design_iiptm - boltzgen",
    "residue_iptm": "design_residue_iptm - boltzgen",
    "ipsae_min": "design_ipsae_min - boltzgen",
    "to_target_ipsae": "design_to_target_ipsae - boltzgen",
    "complex_iplddt": "complex_iplddt - boltzgen",
    "complex_ipde": "complex_ipde - boltzgen",
    "saltbridge": "plip_saltbridge_refolded - boltzgen",
    "neg_filter_rmsd": "neg_filter_rmsd - boltzgen",
    "neg_filter_rmsd_design": "neg_filter_rmsd_design - boltzgen",
}

INTRINSIC_METRICS = {
    "design_ptm": "design_ptm - boltzgen",
    "hydrophobic_patch": "design_largest_hydrophobic_patch_refolded - boltzgen",
    "liability_score": "liability_score - boltzgen",
    "complex_plddt": "complex_plddt - boltzgen",
    "complex_pde": "complex_pde - boltzgen",
}

COMPOSITE_METRICS = {
    "pass_filters": "pass_filters - boltzgen",
}

ALL_METRICS = {**INTERFACE_METRICS, **INTRINSIC_METRICS, **COMPOSITE_METRICS}
CONDITIONS = ["binder_alone", "joint"]
VARIANTS = ["correlation_all_dims", "covariance_all_dims", "covariance_outliers_excluded"]

# --- Load labels ---
labels = {k: [] for k in ALL_METRICS}
csv_ids = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for k, col in ALL_METRICS.items():
            v = row[col]
            if k == "pass_filters":
                labels[k].append(1.0 if v.strip().lower() == "true" else 0.0)
            else:
                try:
                    labels[k].append(float(v) if v not in ("", "nan") else np.nan)
                except ValueError:
                    labels[k].append(np.nan)
        csv_ids.append(row["id - boltzgen"])
for k in labels:
    labels[k] = np.array(labels[k], dtype=np.float64)
n = len(csv_ids)
print(f"loaded {n} rows, {len(ALL_METRICS)} metrics")

with open(IDS_PATH, newline="") as f:
    saved_ids = [row["id"] for row in csv.DictReader(f)]
assert saved_ids == csv_ids, "row order mismatch!"
print("alignment check passed\n")


def pearson_matrix(pcs: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(y)
    pcs, y = pcs[mask], y[mask]
    y_c = y - y.mean()
    pcs_c = pcs - pcs.mean(axis=0)
    num = pcs_c.T @ y_c
    denom = np.sqrt((pcs_c ** 2).sum(axis=0) * (y_c ** 2).sum())
    return num / denom


# best |r| per (condition, metric) across all layers/variants/PCs
best_overall = {cond: {k: (0.0, None, None, None) for k in ALL_METRICS} for cond in CONDITIONS}

for condition in CONDITIONS:
    for layer in range(36):
        layer_dir = PCA_DIR / condition / f"layer_{layer:02d}"
        for variant in VARIANTS:
            scores = np.load(layer_dir / f"{variant}_scores.npy")[:, :N_PCS_TO_CHECK]
            for k, y in labels.items():
                r = pearson_matrix(scores, y)
                idx = np.nanargmax(np.abs(r))
                if abs(r[idx]) > abs(best_overall[condition][k][0]):
                    best_overall[condition][k] = (r[idx], layer, variant, idx)

print(f"{'metric':24s} {'category':10s} {'binder_alone r (layer)':26s} {'joint r (layer)':26s} {'diff':8s}")
category_of = {}
for k in INTERFACE_METRICS: category_of[k] = "interface"
for k in INTRINSIC_METRICS: category_of[k] = "intrinsic"
for k in COMPOSITE_METRICS: category_of[k] = "composite"

interface_diffs, intrinsic_diffs = [], []
for k in ALL_METRICS:
    ba_r, ba_layer, ba_var, ba_pc = best_overall["binder_alone"][k]
    j_r, j_layer, j_var, j_pc = best_overall["joint"][k]
    diff = abs(j_r) - abs(ba_r)
    print(f"{k:24s} {category_of[k]:10s} {ba_r:+.3f} (L{ba_layer:2d})           {j_r:+.3f} (L{j_layer:2d})           {diff:+.3f}")
    if category_of[k] == "interface":
        interface_diffs.append(diff)
    elif category_of[k] == "intrinsic":
        intrinsic_diffs.append(diff)

print(f"\nmean |r|(joint) - |r|(binder_alone), interface metrics: {np.mean(interface_diffs):+.4f} (n={len(interface_diffs)})")
print(f"mean |r|(joint) - |r|(binder_alone), intrinsic metrics: {np.mean(intrinsic_diffs):+.4f} (n={len(intrinsic_diffs)})")
print("(positive = joint does better on average; near zero = essentially tied)")
