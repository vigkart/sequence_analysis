import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np

N, L, D = 80000, 36, 1152
HIGH_THRESHOLD = 4.0
HIGH_WEIGHT = 3.0

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "phase1_mean_pooled"
PCA_DIR = ROOT / "phase1_pca"

with open(ROOT / "data" / "rank_lookup_row_indices.json") as f:
    rank_rows = json.load(f)
with open(ROOT / "data" / "rank_lookup_sequences.json") as f:
    rank_seqs = json.load(f)

NEW14 = {
    "new149": (rank_rows["rank0034_design_spec_149"], 3.10, "flagged: too few cells (Zuli)"),
    "new241": (rank_rows["rank0019_design_spec_241"], 4.92, "clean"),
    "new226": (rank_rows["rank0004_design_spec_226"], 1.05, "flagged: too few cells (Zuli)"),
    "new146": (rank_rows["rank0037_design_spec_146"], 2.72, "flagged: too few cells (Zuli)"),
    "new208": (rank_rows["rank0003_design_spec_208"], 3.38, "clean"),
    "new211": (rank_rows["rank0009_design_spec_211"], 0.92, "clean"),
    "new249": (rank_rows["rank0056_design_spec_249"], 0.64, "clean"),
    "new089": (rank_rows["rank0077_design_spec_089"], 5.12, "clean"),
    "new174": (rank_rows["rank0002_design_spec_174"], 1.62, "clean"),
    "new248": (rank_rows["rank0057_design_spec_248"], 2.80, "clean"),
    "new163": (rank_rows["rank0054_design_spec_163"], 1.41, "clean"),
    "new108": (rank_rows["rank0005_design_spec_108"], 4.03, "clean"),
    "new228": (rank_rows["rank0001_design_spec_228"], 2.72, "clean"),
    "new157": (rank_rows["rank0014_design_spec_157"], 9.19, "flagged: contamination-suspected (alt=4.99)"),
}
PREV4 = {
    "p130": (2007, 7.26, "SLEQQL..."),
    "p144": (16250, 9.72, "SPLERAR..."),
    "p154_prev": (30263, 6.44, "SREEEI..."),
    "b44": (None, 4.99, "NENKKK... (fresh-embedded, not in 80k set)"),
}

print("=" * 70)
print("EXACT DATA USED — 'the 14' (new experiment, screen set)")
print("=" * 70)
for name, (row, label, note) in NEW14.items():
    src = [k for k, v in rank_rows.items() if v == row][0]
    seq = rank_seqs[src]
    print(f"{name:10s} label={label:5.2f}%  row={row:6d}  source={src}  len={len(seq)}  [{note}]")

print("\n" + "=" * 70)
print("EXACT DATA USED — 'the 4' (previous experiment, confirmation set)")
print("=" * 70)
for name, (row, label, note) in PREV4.items():
    print(f"{name:10s} label={label:5.2f}%  row={row}  [{note}]")

# ---- Metric explanation via worked example ----
print("\n" + "=" * 70)
print("METRIC EXPLANATION")
print("=" * 70)
print("""
weighted_concordance: for every PAIR of binders (i,j), check whether the
axis score and the real mCherry label agree on which one is bigger
("concordant"). Each pair gets a weight = weight(i) * weight(j), where
weight(point) = 3.0 if its mCherry >= 4%, else 1.0. So a pair where BOTH
binders are high-activity counts 9x as much as a pair where both are low.
weighted_concordance = (sum of weights of concordant pairs) / (sum of all
pair weights). Since PCA component sign is arbitrary (a component and its
negation carry identical information), we report max(frac, 1-frac) — a
perfectly anti-correlated axis scores the same as a perfectly correlated
one, because we haven't fixed a sign convention. 0.5 = chance, 1.0 = every
pair correctly ordered (accounting for sign).

AUC: standard Mann-Whitney-U AUC for separating the binary groups
(mCherry >= 4% vs < 4%) — the probability a random "high" binder's score
ranks above a random "low" binder's score. Also sign-agnostic (max(auc,
1-auc)). 0.5 = no separation, 1.0 = perfect separation.
""")

# ---- Reprint raw scores for the top hit, for direct visual inspection ----
def get_raw_embeddings(names_dict, condition):
    fname = "binder_alone_mean_hidden.f32" if condition == "binder_alone" else "binder_target_joint_mean_hidden.f32"
    arr = np.memmap(IN_DIR / fname, dtype="float32", mode="r", shape=(N, L, D))
    rows = []
    for name, v in names_dict.items():
        row_idx = v[0]
        if row_idx is not None:
            rows.append(np.array(arr[row_idx]))
        else:
            rows.append(np.load(IN_DIR / f"binder3_4_{condition}.npy"))
    return np.stack(rows, axis=0)


def project(names_dict, condition, layer, variant, pc):
    raw = get_raw_embeddings(names_dict, condition)
    X = raw[:, layer, :]
    layer_dir = PCA_DIR / condition / f"layer_{layer:02d}"
    pca = joblib.load(layer_dir / f"{variant}_pca.joblib")
    if variant == "correlation_all_dims":
        scaler = joblib.load(layer_dir / f"{variant}_scaler.joblib")
        X_in = scaler.transform(X)
    elif variant == "covariance_outliers_excluded":
        keep_dims = np.load(layer_dir / "keep_dims.npy")
        X_in = X[:, keep_dims]
    else:
        X_in = X
    return pca.transform(X_in)[:, pc]


print("=" * 70)
print("RAW SCORES for top hit: joint, layer 9, covariance_all_dims, PC7")
print("=" * 70)
s14 = project(NEW14, "joint", 9, "covariance_all_dims", 7)
s4 = project(PREV4, "joint", 9, "covariance_all_dims", 7)

print("\n'the 14', sorted by real mCherry label:")
order = np.argsort([v[1] for v in NEW14.values()])
names14 = list(NEW14.keys())
for i in order:
    print(f"  {names14[i]:10s} label={NEW14[names14[i]][1]:5.2f}%  PC7_score={s14[i]:+.3f}  [{NEW14[names14[i]][2]}]")

print("\n'the 4', sorted by real mCherry label:")
order4 = np.argsort([v[1] for v in PREV4.values()])
names4 = list(PREV4.keys())
for i in order4:
    print(f"  {names4[i]:10s} label={PREV4[names4[i]][1]:5.2f}%  PC7_score={s4[i]:+.3f}")

# ---- Robustness check: strict-clean subset of "the 14" (drop ALL flagged points) ----
print("\n" + "=" * 70)
print("ROBUSTNESS CHECK: does the finding survive using ONLY the 10 fully-clean")
print("new-experiment points (dropping new149/new226/new146/new157 too, not")
print("just new154)?")
print("=" * 70)

STRICT_NAMES = [n for n, v in NEW14.items() if v[2] == "clean"]
print(f"strict-clean subset (n={len(STRICT_NAMES)}): {STRICT_NAMES}\n")


def weighted_concordance(scores, labels, pairs):
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
    binary = (labels >= HIGH_THRESHOLD).astype(int)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return np.nan
    pos, neg = scores[binary == 1], scores[binary == 0]
    count = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    auc = count / (len(pos) * len(neg))
    return max(auc, 1 - auc)


strict_idx = [names14.index(n) for n in STRICT_NAMES]
strict_labels = np.array([NEW14[n][1] for n in STRICT_NAMES])
strict_pairs = list(combinations(range(len(STRICT_NAMES)), 2))

for condition, layer, variant, pc in [
    ("joint", 9, "covariance_all_dims", 7),
    ("joint", 8, "covariance_all_dims", 7),
    ("joint", 7, "covariance_all_dims", 7),
    ("joint", 11, "covariance_all_dims", 8),
]:
    s_full = project(NEW14, condition, layer, variant, pc)
    s_strict = s_full[strict_idx]
    wc_strict = weighted_concordance(s_strict, strict_labels, strict_pairs)
    auc_strict = auc_score(s_strict, strict_labels)
    print(f"{condition} layer={layer} {variant} PC{pc}: strict-clean(n=10) weighted_conc={wc_strict:.3f} AUC={auc_strict:.3f}")
