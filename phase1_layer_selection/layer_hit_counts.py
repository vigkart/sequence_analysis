from itertools import combinations
from pathlib import Path

import joblib
import numpy as np

N, L, D = 80000, 36, 1152
N_PCS = 20
VARIANTS = ["correlation_all_dims", "covariance_all_dims", "covariance_outliers_excluded"]
CONDITIONS = ["binder_alone", "joint"]

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "phase1_mean_pooled"
PCA_DIR = ROOT / "phase1_pca"

BINDER_NAMES = ["Binder1", "Binder2", "Binder3", "Binder3-4"]
MCHERRY = np.array([7.26, 9.73, 6.44, 4.99])
ROW_IDX = {"Binder1": 30263, "Binder2": 2007, "Binder3": 16250}
TRUE_ORDER_PAIRS = list(combinations(range(4), 2))


def get_raw_embeddings(condition: str) -> np.ndarray:
    fname = "binder_alone_mean_hidden.f32" if condition == "binder_alone" else "binder_target_joint_mean_hidden.f32"
    arr = np.memmap(IN_DIR / fname, dtype="float32", mode="r", shape=(N, L, D))
    rows = [np.array(arr[ROW_IDX[name]]) for name in BINDER_NAMES[:3]]
    rows.append(np.load(IN_DIR / f"binder3_4_{condition}.npy"))
    return np.stack(rows, axis=0)


def concordant_pairs(scores: np.ndarray, labels: np.ndarray) -> int:
    return sum(1 for i, j in TRUE_ORDER_PAIRS if (scores[i] - scores[j]) * (labels[i] - labels[j]) > 0)


# per-layer counts: candidates checked, perfect-forward (6/6), perfect-reverse (0/6)
per_layer = {layer: {"n": 0, "perfect_fwd": 0, "perfect_rev": 0} for layer in range(L)}

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
            pc_scores = pca.transform(X_in)[:, :N_PCS]
            for pc in range(pc_scores.shape[1]):
                s = pc_scores[:, pc]
                if np.std(s) == 0:
                    continue
                conc = concordant_pairs(s, MCHERRY)
                per_layer[layer]["n"] += 1
                if conc == 6:
                    per_layer[layer]["perfect_fwd"] += 1
                elif conc == 0:
                    per_layer[layer]["perfect_rev"] += 1

print(f"{'layer':6s} {'n_candidates':13s} {'perfect_fwd':12s} {'perfect_rev':12s} {'combined':9s}")
combined_counts = []
for layer in range(L):
    d = per_layer[layer]
    combined = d["perfect_fwd"] + d["perfect_rev"]
    combined_counts.append(combined)
    print(f"{layer:<6d} {d['n']:<13d} {d['perfect_fwd']:<12d} {d['perfect_rev']:<12d} {combined:<9d}")

combined_counts = np.array(combined_counts)
n_per_layer = per_layer[0]["n"]
p_null = 2 / 24  # perfect-forward or perfect-reverse, each 1/24 under random ranking
mean_null = n_per_layer * p_null
var_null = n_per_layer * p_null * (1 - p_null)
std_null = np.sqrt(var_null)

best_layer = int(np.argmax(combined_counts))
best_count = combined_counts[best_layer]
z = (best_count - mean_null) / std_null

print(f"\ncandidates per layer: {n_per_layer}")
print(f"null model (pure chance): mean={mean_null:.2f}, std={std_null:.2f} combined perfect-or-reverse hits per layer")
print(f"best layer: {best_layer}, combined count={best_count}, z={z:.2f}")
print(f"mean combined count across all 36 layers: {combined_counts.mean():.2f} (should be close to null mean if no real effect)")

# Expected max-of-36 under pure null, via simulation, for honest significance framing.
rng = np.random.default_rng(0)
sim_max = []
for _ in range(20000):
    draws = rng.binomial(n_per_layer, p_null, size=L)
    sim_max.append(draws.max())
sim_max = np.array(sim_max)
p_value = (sim_max >= best_count).mean()
print(f"\nsimulated null distribution of max-count-across-36-layers (20000 sims):")
print(f"  mean max = {sim_max.mean():.2f}, 95th pct = {np.percentile(sim_max, 95):.0f}, 99th pct = {np.percentile(sim_max, 99):.0f}")
print(f"  P(pure-chance max across 36 layers >= observed best layer's count {best_count}) = {p_value:.3f}")
