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
IN_DIR = ROOT / "phase1_mean_pooled"
PCA_DIR = ROOT / "phase1_pca"

with open(ROOT / "data" / "rank_lookup_row_indices.json") as f:
    rank_rows = json.load(f)

# "the 14": new-experiment binders, excluding only new154 (the one explicitly
# flagged in the table as inaccurate). Zuli's informal "too few cells" notes
# (149/226/146) and the contamination-suspected 157 are still included here.
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

# "the 4": previous experiment, confirmed mapping.
PREV4 = {
    "p130": (2007, 7.26),
    "p144": (16250, 9.72),
    "p154_prev": (30263, 6.44),
    "b44": (None, 4.99),  # fresh-embedded, no row index
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


def weighted_concordance(scores, labels, pairs):
    """Weighted fraction of concordant pairs; pair weight = product of point
    weights (HIGH_WEIGHT if label >= threshold else 1). Sign-agnostic: takes
    max(concordance, 1-concordance) since PC sign is arbitrary."""
    w = np.where(labels >= HIGH_THRESHOLD, HIGH_WEIGHT, 1.0)
    num, den = 0.0, 0.0
    for i, j in pairs:
        pw = w[i] * w[j]
        concordant = (scores[i] - scores[j]) * (labels[i] - labels[j]) > 0
        num += pw * concordant
        den += pw
    frac = num / den
    return max(frac, 1 - frac)


def auc_score(scores, labels):
    """Sign-agnostic AUC for separating >=threshold vs <threshold."""
    binary = (labels >= HIGH_THRESHOLD).astype(int)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return np.nan
    pos, neg = scores[binary == 1], scores[binary == 0]
    count = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    auc = count / (len(pos) * len(neg))
    return max(auc, 1 - auc)


def concordance_plain(scores, labels, pairs):
    conc = sum(1 for i, j in pairs if (scores[i] - scores[j]) * (labels[i] - labels[j]) > 0)
    frac = conc / len(pairs)
    return max(frac, 1 - frac)


# ---- Screen on "the 14" ----
screen_results = []
for condition in CONDITIONS:
    raw14 = get_raw_embeddings(NEW14, condition)  # (14, L, D)
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
                wc = weighted_concordance(s14, NEW14_LABELS, NEW14_PAIRS)
                auc = auc_score(s14, NEW14_LABELS)
                screen_results.append((wc, auc, condition, layer, variant, pc))

screen_results.sort(key=lambda r: -r[0])

print(f"total candidates screened on 'the 14': {len(screen_results)}\n")
print("=== top 25 by weighted concordance on 'the 14' (new experiment) ===")
for wc, auc, condition, layer, variant, pc in screen_results[:25]:
    print(f"weighted_conc={wc:.3f}  AUC(>=4%)={auc:.3f}  {condition:12s} layer={layer:2d}  {variant:28s} PC{pc}")

# ---- Confirm top-N on "the 4" (previous experiment, kept fully separate) ----
TOP_N = 25
top_candidates = screen_results[:TOP_N]

print(f"\n=== confirmation on 'the 4' (previous experiment) for the top {TOP_N} ===")
raw4_cache = {}
for wc14, auc14, condition, layer, variant, pc in top_candidates:
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
    conc4 = concordance_plain(s4, PREV4_LABELS, PREV4_PAIRS)
    marker = "  <-- CONFIRMED (6/6 on the 4)" if conc4 == 1.0 else ""
    print(
        f"screen: wconc14={wc14:.3f} AUC14={auc14:.3f}  |  confirm: conc4={conc4:.3f} (of 6 pairs)  "
        f"{condition:12s} layer={layer:2d}  {variant:28s} PC{pc}{marker}"
    )

print("\nlabels used:")
print("the 14 (new exp):", {k: v[1] for k, v in NEW14.items()})
print("the 4 (prev exp):", {k: v[1] for k, v in PREV4.items()})
