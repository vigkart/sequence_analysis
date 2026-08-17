import json
from pathlib import Path

import joblib
import numpy as np

N, L, D = 80000, 36, 1152
N_PCS = 20
VARIANTS = ["correlation_all_dims", "covariance_all_dims", "covariance_outliers_excluded"]
CONDITIONS = ["binder_alone", "joint"]

IN_DIR = Path("stageA_mean_pooled")
PCA_DIR = Path("stageA_pca")

with open("rank_lookup_row_indices.json") as f:
    rank_rows = json.load(f)

# name -> (row_index or None [needs fresh embed], mCherry %, clean/flagged)
BINDERS = {
    "p130": (2007, 7.26, "clean"),
    "p144": (16250, 9.72, "clean"),
    "p154_prev": (30263, 6.44, "clean"),
    "b44": (None, 4.99, "clean"),
    "new154": (rank_rows["rank0016_design_spec_154"], 0.88, "flagged: too few cells, inaccurate"),
    "new149": (rank_rows["rank0034_design_spec_149"], 3.1, "flagged: too few cells (Zuli)"),
    "new241": (rank_rows["rank0019_design_spec_241"], 4.92, "clean"),
    "new226": (rank_rows["rank0004_design_spec_226"], 1.05, "flagged: too few cells (Zuli)"),
    "new146": (rank_rows["rank0037_design_spec_146"], 2.72, "flagged: too few cells (Zuli)"),
    "new208": (rank_rows["rank0003_design_spec_208"], 3.38, "clean"),
    "new211": (rank_rows["rank0009_design_spec_211"], 0.92, "clean"),
    "new249": (rank_rows["rank0056_design_spec_249"], 0.64, "clean"),
    "new089": (rank_rows["rank0077_design_spec_089"], 5.12, "clean"),
    "new174": (rank_rows["rank0002_design_spec_174"], 1.62, "clean"),
    "new248": (rank_rows["rank0057_design_spec_248"], 2.8, "clean"),
    "new163": (rank_rows["rank0054_design_spec_163"], 1.41, "clean"),
    "new108": (rank_rows["rank0005_design_spec_108"], 4.03, "clean"),
    "new228": (rank_rows["rank0001_design_spec_228"], 2.72, "clean"),
    "new157": (rank_rows["rank0014_design_spec_157"], 9.19, "flagged: contamination-suspected duplicate (alt=4.99)"),
}

NAMES = list(BINDERS.keys())
MCHERRY = np.array([BINDERS[n][1] for n in NAMES])
CLEAN_MASK = np.array([BINDERS[n][2] == "clean" for n in NAMES])
print(f"{len(NAMES)} binders total, {CLEAN_MASK.sum()} clean\n")


def get_raw_embeddings(condition: str) -> np.ndarray:
    fname = "binder_alone_mean_hidden.f32" if condition == "binder_alone" else "binder_target_joint_mean_hidden.f32"
    arr = np.memmap(IN_DIR / fname, dtype="float32", mode="r", shape=(N, L, D))
    rows = []
    for name in NAMES:
        row_idx = BINDERS[name][0]
        if row_idx is not None:
            rows.append(np.array(arr[row_idx]))
        else:
            rows.append(np.load(IN_DIR / f"binder3_4_{condition}.npy"))
    return np.stack(rows, axis=0)  # (19, L, D)


def pearson(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return (xc @ yc) / denom if denom > 1e-9 else np.nan


results = []  # (r_all, r_clean, condition, layer, variant, pc)

for condition in CONDITIONS:
    raw = get_raw_embeddings(condition)  # (19, L, D)
    for layer in range(L):
        layer_dir = PCA_DIR / condition / f"layer_{layer:02d}"
        X_layer = raw[:, layer, :]
        for variant in VARIANTS:
            pca = joblib.load(layer_dir / f"{variant}_pca.joblib")
            if variant == "correlation_all_dims":
                scaler = joblib.load(layer_dir / f"{variant}_scaler.joblib")
                X_in = scaler.transform(X_layer)
            elif variant == "covariance_outliers_excluded":
                keep_dims = np.load(layer_dir / "keep_dims.npy")
                X_in = X_layer[:, keep_dims]
            else:
                X_in = X_layer
            pc_scores = pca.transform(X_in)[:, :N_PCS]
            for pc in range(pc_scores.shape[1]):
                s = pc_scores[:, pc]
                r_all = pearson(s, MCHERRY)
                r_clean = pearson(s[CLEAN_MASK], MCHERRY[CLEAN_MASK])
                results.append((r_all, r_clean, condition, layer, variant, pc))

results.sort(key=lambda r: -abs(r[1]) if not np.isnan(r[1]) else 0)

print("=== top 20 by |r| on CLEAN subset (n=%d) ===" % CLEAN_MASK.sum())
for r_all, r_clean, condition, layer, variant, pc in results[:20]:
    print(f"r_clean={r_clean:+.3f}  r_all(n=19)={r_all:+.3f}  {condition:12s} layer={layer:2d}  {variant:28s} PC{pc}")

print("\n=== layer 14 specifically, all variants, top 3 PCs ===")
for r_all, r_clean, condition, layer, variant, pc in results:
    if layer == 14 and pc < 3:
        print(f"r_clean={r_clean:+.3f}  r_all(n=19)={r_all:+.3f}  {condition:12s} layer={layer:2d}  {variant:28s} PC{pc}")

print("\nbinder list (name, mCherry%, status):")
for n in NAMES:
    print(f"  {n:12s} {BINDERS[n][1]:6.2f}%  {BINDERS[n][2]}")
