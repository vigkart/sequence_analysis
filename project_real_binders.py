from itertools import combinations
from pathlib import Path

import joblib
import numpy as np

N, L, D = 80000, 36, 1152
N_PCS = 20
VARIANTS = ["correlation_all_dims", "covariance_all_dims", "covariance_outliers_excluded"]
CONDITIONS = ["binder_alone", "joint"]

IN_DIR = Path("stageA_mean_pooled")
PCA_DIR = Path("stageA_pca")

BINDER_NAMES = ["Binder1", "Binder2", "Binder3", "Binder3-4"]
MCHERRY = np.array([7.26, 9.73, 6.44, 4.99])
ROW_IDX = {"Binder1": 30263, "Binder2": 2007, "Binder3": 16250}  # Binder3-4 embedded fresh

TRUE_ORDER_PAIRS = list(combinations(range(4), 2))  # (i, j) pairs, i<j


def get_raw_embeddings(condition: str) -> np.ndarray:
    """Return (4, L, D) raw mean-pooled embeddings for the 4 binders, in BINDER_NAMES order."""
    fname = "binder_alone_mean_hidden.f32" if condition == "binder_alone" else "binder_target_joint_mean_hidden.f32"
    arr = np.memmap(IN_DIR / fname, dtype="float32", mode="r", shape=(N, L, D))
    rows = []
    for name in BINDER_NAMES[:3]:
        rows.append(np.array(arr[ROW_IDX[name]]))  # (L, D)
    fresh = np.load(IN_DIR / f"binder3_4_{condition}.npy")  # (L, D)
    rows.append(fresh)
    return np.stack(rows, axis=0)  # (4, L, D)


def concordant_pairs(scores: np.ndarray, labels: np.ndarray) -> int:
    count = 0
    for i, j in TRUE_ORDER_PAIRS:
        if (scores[i] - scores[j]) * (labels[i] - labels[j]) > 0:
            count += 1
    return count


all_hits = []  # (r, concordant, condition, layer, variant, pc)
n_candidates = 0

for condition in CONDITIONS:
    raw = get_raw_embeddings(condition)  # (4, L, D)

    for layer in range(L):
        layer_dir = PCA_DIR / condition / f"layer_{layer:02d}"
        X_layer = raw[:, layer, :]  # (4, D)

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

            pc_scores = pca.transform(X_in)[:, :N_PCS]  # (4, N_PCS)

            for pc in range(pc_scores.shape[1]):
                s = pc_scores[:, pc]
                if np.std(s) == 0:
                    continue
                r = np.corrcoef(s, MCHERRY)[0, 1]
                conc = concordant_pairs(s, MCHERRY)
                all_hits.append((r, conc, condition, layer, variant, pc))
                n_candidates += 1

n_perfect = sum(1 for h in all_hits if h[1] == 6)
n_reverse_perfect = sum(1 for h in all_hits if h[1] == 0)
expected_perfect = n_candidates / 24

print(f"total candidate axes searched: {n_candidates}")
print(f"perfect ranking match (6/6 concordant): {n_perfect}  (chance expectation: ~{expected_perfect:.0f})")
print(f"perfect REVERSE ranking (0/6 concordant): {n_reverse_perfect}  (chance expectation: ~{expected_perfect:.0f})")

print("\n=== top 20 by |r| across the full search (exploratory — see chance caveat above) ===")
all_hits.sort(key=lambda h: -abs(h[0]))
for r, conc, condition, layer, variant, pc in all_hits[:20]:
    print(f"r={r:+.3f}  concordant={conc}/6  {condition:12s} layer={layer:2d}  {variant:28s} PC{pc}")

print("\n=== pre-registered check: layer 14 only (chosen before seeing this real data) ===")
for r, conc, condition, layer, variant, pc in all_hits:
    if layer == 14 and pc < 3:
        print(f"r={r:+.3f}  concordant={conc}/6  {condition:12s} layer={layer:2d}  {variant:28s} PC{pc}")

print("\nreal values used: " + ", ".join(f"{n}={v}" for n, v in zip(BINDER_NAMES, MCHERRY)))
print(f"true rank order (high to low activity): {[BINDER_NAMES[i] for i in np.argsort(-MCHERRY)]}")
