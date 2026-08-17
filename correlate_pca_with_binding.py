import csv
from pathlib import Path

import numpy as np

CSV_PATH = "boltzgen_conf_1_batch_generation-scores.csv"
IDS_PATH = "stageA_mean_pooled/ids.csv"
PCA_DIR = Path("stageA_pca")
N_PCS_TO_CHECK = 20

LABEL_COLS = {
    "iptm": "design_to_target_iptm - boltzgen",
    "ipsae": "design_to_target_ipsae - boltzgen",
    "pae": "min_design_to_target_pae - boltzgen",  # lower = better
    "pass_filters": "pass_filters - boltzgen",
}

CONDITIONS = ["binder_alone", "joint"]
VARIANTS = ["correlation_all_dims", "covariance_all_dims", "covariance_outliers_excluded"]

# --- Load labels, aligned by row order (same CSV iteration order used to build embeddings) ---
labels = {k: [] for k in LABEL_COLS}
lengths = []
csv_ids = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for k, col in LABEL_COLS.items():
            v = row[col]
            if k == "pass_filters":
                labels[k].append(1.0 if v.strip().lower() == "true" else 0.0)
            else:
                labels[k].append(float(v) if v not in ("", "nan") else np.nan)
        lengths.append(len(row["designed_sequence - boltzgen"]))
        csv_ids.append(row["id - boltzgen"])

for k in labels:
    labels[k] = np.array(labels[k], dtype=np.float64)
lengths = np.array(lengths, dtype=np.float64)
n = len(lengths)
print(f"loaded {n} labeled rows")

# --- Sanity check alignment against saved ids.csv ---
with open(IDS_PATH, newline="") as f:
    reader = csv.DictReader(f)
    saved_ids = [row["id"] for row in reader]
assert saved_ids == csv_ids, "row order mismatch between ids.csv and scores CSV — embeddings and labels are NOT aligned!"
print("alignment check passed: ids.csv matches scores CSV row order")


def pearson_matrix(pcs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r between each column of pcs (N, K) and y (N,), NaN-safe."""
    mask = ~np.isnan(y)
    pcs = pcs[mask]
    y = y[mask]
    y_c = y - y.mean()
    pcs_c = pcs - pcs.mean(axis=0)
    num = pcs_c.T @ y_c
    denom = np.sqrt((pcs_c ** 2).sum(axis=0) * (y_c ** 2).sum())
    return num / denom


results = []  # rows: condition, layer, best per label -> (variant, pc_idx, r)

for condition in CONDITIONS:
    for layer in range(36):
        layer_dir = PCA_DIR / condition / f"layer_{layer:02d}"
        best = {k: (None, None, 0.0) for k in LABEL_COLS}  # label -> (variant, pc_idx, r)
        best_length = (None, None, 0.0)

        for variant in VARIANTS:
            scores = np.load(layer_dir / f"{variant}_scores.npy")[:, :N_PCS_TO_CHECK]

            for k, y in labels.items():
                r = pearson_matrix(scores, y)
                idx = np.nanargmax(np.abs(r))
                if abs(r[idx]) > abs(best[k][2]):
                    best[k] = (variant, idx, r[idx])

            r_len = pearson_matrix(scores, lengths)
            idx = np.nanargmax(np.abs(r_len))
            if abs(r_len[idx]) > abs(best_length[2]):
                best_length = (variant, idx, r_len[idx])

        results.append((condition, layer, best, best_length))

print(f"\n{'cond':12s} {'layer':5s} {'iptm(r,pc,var)':28s} {'ipsae(r,pc,var)':28s} {'pae(r,pc,var)':28s} {'pass(r,pc,var)':28s} {'length(r,pc,var)':28s}")
for condition, layer, best, best_length in results:
    def fmt(entry):
        variant, idx, r = entry
        vshort = {"correlation_all_dims": "corr", "covariance_all_dims": "cov_all", "covariance_outliers_excluded": "cov_excl"}[variant]
        return f"{r:+.3f} PC{idx} {vshort}"

    print(
        f"{condition:12s} {layer:<5d} "
        f"{fmt(best['iptm']):28s} {fmt(best['ipsae']):28s} {fmt(best['pae']):28s} "
        f"{fmt(best['pass_filters']):28s} {fmt(best_length):28s}"
    )

# --- Overall best layer per condition per label ---
print("\n=== overall best layer per (condition, label) ===")
for condition in CONDITIONS:
    cond_results = [r for r in results if r[0] == condition]
    for k in LABEL_COLS:
        best_overall = max(cond_results, key=lambda r: abs(r[2][k][2]))
        variant, idx, r_val = best_overall[2][k]
        print(f"{condition:12s} label={k:14s} best layer={best_overall[1]:2d}  r={r_val:+.3f}  PC{idx}  variant={variant}")
