"""Corrected re-run of screen_then_confirm.py: fixes a real methodological
bug found while reviewing the SAE feature-selection plan (see docs/TODO.md).

THE BUG: screen_then_confirm.py's confirmation step scores the held-out 4
with `concordance_plain`, which -- like the screening step's
`weighted_concordance` -- is `max(frac, 1-frac)`: SIGN-AGNOSTIC. This means
the confirm step independently re-optimizes which of the two possible
directions (higher-PC-score-is-better vs. lower-is-better) scores best on
the 4, rather than checking whether the SPECIFIC direction that won during
screening on the 14 also holds on the 4. No numerical model is fit on the
4 (that part of the methodology is fine) -- but allowing the confirm
metric to silently pick whichever of 2 directions flatters it is still a
one-bit form of fitting to the "held out" set, hidden inside the metric.

Confirmed empirically before writing this fix: for the headline result
(joint, layer 9, covariance_all_dims, PC7 -- 0-indexed column 7, not 6; see
also the PC7_INDEX fix in phase2_feature_extraction/sae/extract_binder_residues.py),
the negative direction wins 90.4% of weighted pairs on the 14, but the
POSITIVE direction wins 100% of pairs on the 4 -- genuinely opposite
directions, not a computation error (re-verified by calling
screen_then_confirm.py's own functions directly, not a reimplementation).

THE FIX: lock the sign found during screening (on the 14) and score the 4
using THAT SAME fixed direction -- no re-optimization on held-out data.
Reports both the OLD sign-agnostic confirm score and the NEW locked-direction
score side by side so the gap between them is visible for every candidate,
not just the previous top pick.

Usage:
    python screen_then_confirm_signlocked.py
"""

import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np

N, L, D = 80000, 36, 1152
N_PCS = 20
VARIANTS = ["correlation_all_dims", "covariance_all_dims", "covariance_outliers_excluded"]
CONDITIONS = ["binder_alone", "joint"]
HIGH_THRESHOLD = 4.0
HIGH_WEIGHT = 3.0

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "phase1_mean_pooled"
PCA_DIR = ROOT / "data" / "phase1_pca"

with open(ROOT / "data" / "rank_lookup_row_indices.json") as f:
    rank_rows = json.load(f)

NEW14 = {
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
PREV4 = {
    "p130": (2007, 7.26),
    "p144": (16250, 9.72),
    "p154_prev": (30263, 6.44),
    "b44": (None, 4.99),
}

NEW14_NAMES = list(NEW14.keys())
NEW14_LABELS = np.array([NEW14[n][1] for n in NEW14_NAMES])
NEW14_PAIRS = list(combinations(range(len(NEW14_NAMES)), 2))

PREV4_NAMES = list(PREV4.keys())
PREV4_LABELS = np.array([PREV4[n][1] for n in PREV4_NAMES])
PREV4_PAIRS = list(combinations(range(len(PREV4_NAMES)), 2))


def get_raw_embeddings(names_dict, condition):
    fname = "binder_alone_mean_hidden.f32" if condition == "binder_alone" else "binder_target_joint_mean_hidden.f32"
    arr = np.memmap(IN_DIR / fname, dtype="float32", mode="r", shape=(N, L, D))
    rows = []
    for name, (row_idx, _) in names_dict.items():
        if row_idx is not None:
            rows.append(np.array(arr[row_idx]))
        else:
            rows.append(np.load(IN_DIR / f"binder3_4_{condition}.npy"))
    return np.stack(rows, axis=0)


def signed_weighted_fraction(scores, labels, pairs):
    """Raw (NOT sign-flipped) weighted concordant fraction -- the positive-
    direction score. weighted_concordance's `max(frac, 1-frac)` is derived
    from this; we keep the raw value so a direction can be locked in."""
    w = np.where(labels >= HIGH_THRESHOLD, HIGH_WEIGHT, 1.0)
    num, den = 0.0, 0.0
    for i, j in pairs:
        pw = w[i] * w[j]
        concordant = (scores[i] - scores[j]) * (labels[i] - labels[j]) > 0
        num += pw * concordant
        den += pw
    return num / den


def signed_plain_fraction(scores, labels, pairs):
    """Unweighted version, matching concordance_plain but without the
    max(frac, 1-frac) sign flip."""
    conc = sum(1 for i, j in pairs if (scores[i] - scores[j]) * (labels[i] - labels[j]) > 0)
    return conc / len(pairs)


def auc_score(scores, labels):
    binary = (labels >= HIGH_THRESHOLD).astype(int)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return np.nan
    pos, neg = scores[binary == 1], scores[binary == 0]
    count = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    auc = count / (len(pos) * len(neg))
    return max(auc, 1 - auc)


# ---- Screen on "the 14" (unchanged from the original -- selecting
# candidates using only screen-set information is legitimate; the bug was
# entirely in how the confirm step scored them) ----
screen_results = []
for condition in CONDITIONS:
    raw14 = get_raw_embeddings(NEW14, condition)
    for layer in range(L):
        layer_dir = PCA_DIR / condition / f"layer_{layer:02d}"
        X14 = raw14[:, layer, :]
        for variant in VARIANTS:
            pca = joblib.load(layer_dir / f"{variant}_pca.joblib")
            if variant == "correlation_all_dims":
                scaler = joblib.load(layer_dir / f"{variant}_scaler.joblib")
                X14_in = scaler.transform(X14)
            elif variant == "covariance_outliers_excluded":
                keep_dims = np.load(layer_dir / "keep_dims.npy")
                X14_in = X14[:, keep_dims]
            else:
                X14_in = X14
            pcs14 = pca.transform(X14_in)[:, :N_PCS]
            for pc in range(pcs14.shape[1]):
                s14 = pcs14[:, pc]
                frac14 = signed_weighted_fraction(s14, NEW14_LABELS, NEW14_PAIRS)
                direction = 1 if frac14 >= 0.5 else -1
                wc = max(frac14, 1 - frac14)
                auc = auc_score(s14, NEW14_LABELS)
                screen_results.append((wc, auc, direction, condition, layer, variant, pc))

screen_results.sort(key=lambda r: -r[0])
print(f"total candidates screened on 'the 14': {len(screen_results)}\n")

# ---- Confirm: SIGN LOCKED to whatever won on the 14 ----
TOP_N = 50
top_candidates = screen_results[:TOP_N]

print(f"=== top {TOP_N} by weighted concordance on 'the 14', confirmed with the LOCKED direction on 'the 4' ===")
print(f"{'wconc14':>8} {'auc14':>6} {'dir14':>5} | {'conc4_locked':>12} {'conc4_old(sign-agnostic)':>24} | axis")
raw4_cache = {}
n_genuinely_confirmed = 0
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

    frac4 = signed_plain_fraction(s4, PREV4_LABELS, PREV4_PAIRS)
    conc4_locked = frac4 if direction == 1 else (1 - frac4)
    conc4_old = max(frac4, 1 - frac4)  # what the original script reported

    genuinely_confirmed = conc4_locked == 1.0
    n_genuinely_confirmed += genuinely_confirmed
    marker = "  <-- GENUINELY CONFIRMED (direction from the 14 also holds on the 4)" if genuinely_confirmed else ""
    flip_flag = "  [DIRECTION FLIPPED vs. old sign-agnostic report]" if (conc4_locked != conc4_old and conc4_old == 1.0) else ""
    print(
        f"{wc14:8.3f} {auc14:6.3f} {'+' if direction==1 else '-':>5} | "
        f"{conc4_locked:12.3f} {conc4_old:24.3f} | "
        f"{condition:12s} layer={layer:2d}  {variant:28s} PC{pc}{marker}{flip_flag}"
    )

print(f"\n{n_genuinely_confirmed}/{TOP_N} candidates GENUINELY confirmed (locked direction from the 14 also gets 6/6 on the 4).")
print("Compare to the original script's count of sign-agnostic 'CONFIRMED' hits to see how many were direction-flip artifacts.")

print("\nlabels used:")
print("the 14 (new exp):", {k: v[1] for k, v in NEW14.items()})
print("the 4 (prev exp):", {k: v[1] for k, v in PREV4.items()})
