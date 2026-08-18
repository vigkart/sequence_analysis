"""Exact permutation test on the sign-locked reversal pattern found in
screen_then_confirm_signlocked.py -- is the observed skew toward
conc4_locked reversal itself distinguishable from chance, given the top-50
candidates are highly correlated (not independent trials)?

Follows the same exact-permutation precedent as
phase1_layer_selection/utilities/permutation_test.py (all 4! = 24 possible
label assignments for "the 4"), but applied holistically to the sign-locked
methodology: for each of the 24 possible relabelings of PREV4_LABELS onto
the 4 embedded real-binder points, recompute conc4_locked for ALL top-50
candidates (screening on the 14, and thus each candidate's locked direction,
is unaffected by how the 4 is relabeled) and compute an aggregate reversal
statistic. Comparing the TRUE labeling's aggregate stat against the full
distribution over all 24 relabelings properly accounts for correlation
between candidates, unlike a naive per-candidate chance comparison.

Usage:
    python signlock_permutation_test.py
"""

from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from screen_then_confirm_signlocked import (
        NEW14, PREV4, PREV4_LABELS, get_raw_embeddings, PCA_DIR,
        screen_results, TOP_N,
    )

top_candidates = screen_results[:TOP_N]
PREV4_PAIRS = list(combinations(range(4), 2))
PREV4_NAMES = list(PREV4.keys())

raw4_cache = {}
s4_by_candidate = []
for wc14, auc14, direction, condition, layer, variant, pc in top_candidates:
    if condition not in raw4_cache:
        raw4_cache[condition] = get_raw_embeddings(PREV4, condition)
    raw4 = raw4_cache[condition]
    layer_dir = PCA_DIR / condition / f"layer_{layer:02d}"
    X4 = raw4[:, layer, :]
    pca = joblib.load(layer_dir / f"{variant}_pca.joblib")
    if variant == "correlation_all_dims":
        scaler = joblib.load(layer_dir / f"{variant}_scaler.joblib")
        X4_in = scaler.transform(X4)
    elif variant == "covariance_outliers_excluded":
        keep_dims = np.load(layer_dir / "keep_dims.npy")
        X4_in = X4[:, keep_dims]
    else:
        X4_in = X4
    s4 = pca.transform(X4_in)[:, pc]
    s4_by_candidate.append((s4, direction))

print(f"Testing {len(top_candidates)} top candidates against all 4! = 24 label permutations of 'the 4'\n")


def aggregate_stats(labels: np.ndarray) -> dict:
    """Given one assignment of labels to the 4 embedded points, compute
    conc4_locked for every top-50 candidate and summarize."""
    locked_scores = []
    for s4, direction in s4_by_candidate:
        conc = sum(1 for i, j in PREV4_PAIRS if (s4[i] - s4[j]) * (labels[i] - labels[j]) > 0) / len(PREV4_PAIRS)
        locked = conc if direction == 1 else (1 - conc)
        locked_scores.append(locked)
    locked_scores = np.array(locked_scores)
    return {
        "mean_locked": locked_scores.mean(),
        "n_perfect_reversal": int((locked_scores == 0.0).sum()),
        "n_genuinely_confirmed": int((locked_scores == 1.0).sum()),
    }


true_labels = PREV4_LABELS
true_stats = aggregate_stats(true_labels)
print(f"TRUE labeling: mean_locked={true_stats['mean_locked']:.4f}  "
      f"n_perfect_reversal={true_stats['n_perfect_reversal']}/{TOP_N}  "
      f"n_genuinely_confirmed={true_stats['n_genuinely_confirmed']}/{TOP_N}")

all_perm_stats = []
for perm in permutations(range(4)):
    permuted_labels = true_labels[list(perm)]
    all_perm_stats.append(aggregate_stats(permuted_labels))

mean_locked_dist = np.array([s["mean_locked"] for s in all_perm_stats])
n_reversal_dist = np.array([s["n_perfect_reversal"] for s in all_perm_stats])

# Rank of the TRUE labeling among all 24 (including itself), for each direction of "extreme"
rank_low_mean = int((mean_locked_dist <= true_stats["mean_locked"]).sum())   # how many perms are AS OR MORE reversed
rank_high_reversal = int((n_reversal_dist >= true_stats["n_perfect_reversal"]).sum())  # how many perms have AS MANY OR MORE perfect reversals

print(f"\nAcross all 24 label permutations:")
print(f"  mean_locked distribution: min={mean_locked_dist.min():.4f} max={mean_locked_dist.max():.4f} "
      f"mean={mean_locked_dist.mean():.4f} (true={true_stats['mean_locked']:.4f})")
print(f"  n_perfect_reversal distribution: min={n_reversal_dist.min()} max={n_reversal_dist.max()} "
      f"mean={n_reversal_dist.mean():.2f} (true={true_stats['n_perfect_reversal']})")

print(f"\nExact permutation p-value (mean_locked as low or lower than true): {rank_low_mean}/24 = {rank_low_mean/24:.4f}")
print(f"Exact permutation p-value (n_perfect_reversal as high or higher than true): {rank_high_reversal}/24 = {rank_high_reversal/24:.4f}")

print("\nAll 24 permutations, sorted by mean_locked (ascending = most reversed first):")
for perm, stats in sorted(zip(permutations(range(4)), all_perm_stats), key=lambda x: x[1]["mean_locked"]):
    labels_this_perm = true_labels[list(perm)]
    is_true = "  <-- TRUE LABELING" if list(perm) == [0, 1, 2, 3] else ""
    print(f"  perm={perm}  labels={[f'{l:.2f}' for l in labels_this_perm]}  "
          f"mean_locked={stats['mean_locked']:.4f}  n_perfect_reversal={stats['n_perfect_reversal']}{is_true}")
