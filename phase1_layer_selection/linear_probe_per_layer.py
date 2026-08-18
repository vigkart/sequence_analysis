import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import LeaveOneOut

N, L, D = 80000, 36, 1152
N_PCS_FOR_PROBE = 10
VARIANT = "covariance_all_dims"
HIGH_THRESHOLD = 4.0
RIDGE_ALPHAS = np.logspace(-2, 4, 40)
LOGREG_CS = np.logspace(-3, 3, 40)

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "phase1_mean_pooled"
PCA_DIR = ROOT / "phase1_pca"

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

names14 = list(NEW14.keys())
y14_cont = np.array([v[1] for v in NEW14.values()])
y14_bin = (y14_cont >= HIGH_THRESHOLD).astype(int)

names4 = list(PREV4.keys())
y4_cont = np.array([v[1] for v in PREV4.values()])
pairs4 = list(combinations(range(4), 2))


def get_raw(names_dict, condition):
    fname = "binder_alone_mean_hidden.f32" if condition == "binder_alone" else "binder_target_joint_mean_hidden.f32"
    arr = np.memmap(IN_DIR / fname, dtype="float32", mode="r", shape=(N, L, D))
    rows = []
    for name, v in names_dict.items():
        row_idx = v[0]
        rows.append(np.array(arr[row_idx]) if row_idx is not None else np.load(IN_DIR / f"binder3_4_{condition}.npy"))
    return np.stack(rows, axis=0)


def project(raw, condition, layer, n_pcs):
    layer_dir = PCA_DIR / condition / f"layer_{layer:02d}"
    pca = joblib.load(layer_dir / f"{VARIANT}_pca.joblib")
    X = raw[:, layer, :]
    return pca.transform(X)[:, :n_pcs]


def concordance(pred, true, pairs):
    conc = sum(1 for i, j in pairs if (pred[i] - pred[j]) * (true[i] - true[j]) > 0)
    frac = conc / len(pairs)
    return max(frac, 1 - frac)


results = []
for condition in ["binder_alone", "joint"]:
    raw14 = get_raw(NEW14, condition)
    raw4 = get_raw(PREV4, condition)

    for layer in range(L):
        X14 = project(raw14, condition, layer, N_PCS_FOR_PROBE)
        X4 = project(raw4, condition, layer, N_PCS_FOR_PROBE)

        # Regression probe: RidgeCV, alpha tuned via leave-one-out within the 14.
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=LeaveOneOut(), scoring="neg_mean_squared_error")
        ridge.fit(X14, y14_cont)
        pred4_reg = ridge.predict(X4)
        reg_concordance = concordance(pred4_reg, y4_cont, pairs4)
        reg_pearson = np.corrcoef(pred4_reg, y4_cont)[0, 1]

        # Classification probe: L2 logistic regression, C tuned via leave-one-out within the 14.
        logreg = LogisticRegressionCV(
            Cs=LOGREG_CS, cv=LeaveOneOut(), penalty="l2", max_iter=5000, class_weight="balanced"
        )
        logreg.fit(X14, y14_bin)
        pred4_proba = logreg.predict_proba(X4)[:, 1]
        pred4_class = (pred4_proba >= 0.5).astype(int)
        clf_hit_rate = pred4_class.mean()  # all 4 true labels are 1 (high), so this is sensitivity only
        clf_proba_concordance = concordance(pred4_proba, y4_cont, pairs4)

        results.append({
            "condition": condition, "layer": layer,
            "reg_concordance": reg_concordance, "reg_pearson": reg_pearson,
            "reg_alpha": ridge.alpha_,
            "clf_hit_rate": clf_hit_rate, "clf_proba_concordance": clf_proba_concordance,
            "clf_C": logreg.C_[0],
        })

print(f"{'cond':12s} {'layer':5s} {'reg_conc':9s} {'reg_r':7s} {'alpha':9s} | {'clf_hit4/4':10s} {'clf_proba_conc':14s} {'C':8s}")
for r in results:
    print(f"{r['condition']:12s} {r['layer']:<5d} {r['reg_concordance']:.3f}     {r['reg_pearson']:+.3f}  {r['reg_alpha']:8.2f} | "
          f"{r['clf_hit_rate']*4:.0f}/4        {r['clf_proba_concordance']:.3f}          {r['clf_C']:.4f}")

print("\n=== ranked by regression concordance on the 4 (held out) ===")
for r in sorted(results, key=lambda r: -r["reg_concordance"])[:15]:
    print(f"{r['condition']:12s} layer={r['layer']:2d}  reg_concordance={r['reg_concordance']:.3f}  reg_pearson={r['reg_pearson']:+.3f}  clf_proba_concordance={r['clf_proba_concordance']:.3f}")

print("\n=== ranked by classification probability concordance on the 4 ===")
for r in sorted(results, key=lambda r: -r["clf_proba_concordance"])[:15]:
    print(f"{r['condition']:12s} layer={r['layer']:2d}  clf_proba_concordance={r['clf_proba_concordance']:.3f}  clf_hit_rate={r['clf_hit_rate']*4:.0f}/4  reg_concordance={r['reg_concordance']:.3f}")

with open(ROOT / "data" / "linear_probe_results.json", "w") as f:
    json.dump(results, f, indent=2)
