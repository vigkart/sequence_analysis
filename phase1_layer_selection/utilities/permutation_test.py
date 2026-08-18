from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np

N, L, D = 80000, 36, 1152
N_PCS = 20
VARIANTS = ["correlation_all_dims", "covariance_all_dims", "covariance_outliers_excluded"]
CONDITIONS = ["binder_alone", "joint"]

ROOT = Path(__file__).resolve().parent.parent.parent
IN_DIR = ROOT / "data" / "phase1_mean_pooled"
PCA_DIR = ROOT / "data" / "phase1_pca"

# Plasmid-number naming (corrected mapping: p130=Binder1, p144=Binder2, p154=Binder3, b44=Binder3-4)
BINDER_NAMES = ["p130", "p144", "p154", "b44"]
MCHERRY = np.array([7.26, 9.72, 6.44, 4.99])
ROW_IDX = {"p130": 2007, "p144": 16250, "p154": 30263}
TRUE_ORDER_PAIRS = list(combinations(range(4), 2))


def get_raw_embeddings(condition: str) -> np.ndarray:
    fname = "binder_alone_mean_hidden.f32" if condition == "binder_alone" else "binder_target_joint_mean_hidden.f32"
    arr = np.memmap(IN_DIR / fname, dtype="float32", mode="r", shape=(N, L, D))
    rows = [np.array(arr[ROW_IDX[name]]) for name in BINDER_NAMES[:3]]
    rows.append(np.load(IN_DIR / f"binder3_4_{condition}.npy"))
    return np.stack(rows, axis=0)


def concordant_pairs(scores: np.ndarray, labels: np.ndarray) -> int:
    return sum(1 for i, j in TRUE_ORDER_PAIRS if (scores[i] - scores[j]) * (labels[i] - labels[j]) > 0)


# --- Pass 1: compute and cache pc_scores for every (condition, layer, variant), fixed geometry ---
cache = {}  # (condition, layer, variant) -> (4, N_PCS) pc scores
for condition in CONDITIONS:
    raw = get_raw_embeddings(condition)
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
            cache[(condition, layer, variant)] = pca.transform(X_in)[:, :N_PCS]

print(f"cached {len(cache)} (condition, layer, variant) score matrices")


def max_combined_count_for_labels(labels: np.ndarray) -> tuple[int, int]:
    """Returns (max combined perfect-or-reverse count across layers, which layer)."""
    per_layer_combined = np.zeros(L, dtype=int)
    for (condition, layer, variant), pc_scores in cache.items():
        for pc in range(pc_scores.shape[1]):
            s = pc_scores[:, pc]
            if np.std(s) == 0:
                continue
            conc = concordant_pairs(s, labels)
            if conc == 6 or conc == 0:
                per_layer_combined[layer] += 1
    best_layer = int(np.argmax(per_layer_combined))
    return int(per_layer_combined[best_layer]), best_layer


# --- Pass 2: exact permutation test — all 4! = 24 label assignments ---
results = []
for perm in permutations(range(4)):
    labels = MCHERRY[list(perm)]
    max_count, best_layer = max_combined_count_for_labels(labels)
    results.append((perm, max_count, best_layer))

results.sort(key=lambda r: -r[1])
true_perm = tuple(range(4))
true_rank = next(i for i, r in enumerate(results) if r[0] == true_perm) + 1
true_count = next(r[1] for r in results if r[0] == true_perm)

print(f"\nall 24 label permutations, sorted by max-combined-count-across-layers (descending):")
for perm, max_count, best_layer in results:
    marker = " <-- TRUE ASSIGNMENT" if perm == true_perm else ""
    names_in_order = [BINDER_NAMES[i] for i in perm]
    print(f"  labels_assigned_as={names_in_order}  max_count={max_count}  best_layer={best_layer}{marker}")

print(f"\ntrue assignment's max_count={true_count}, rank {true_rank} of 24 (1=highest)")
print(f"exact permutation p-value = {true_rank}/24 = {true_rank / 24:.3f}")
print("(this is the best-possible resolution with n=4 — cannot get finer than 1/24 no matter what)")
