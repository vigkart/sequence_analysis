"""Sparse multivariate probe: can a COMBINATION of SAE features (possibly
different features mattering for different binders, unlike the single-
feature concordance test in correlate_features_with_binding.py) predict
real binding activity? Same approach iGEM's own prior version of this
pipeline used (igemmodel/sae/training/feature_analysis.py's
run_linear_probe), reintroduced here specifically to test this hypothesis
after the single-feature and aggregate-distinctiveness tests both came
back null/weak.

Fits LassoCV (L1-regularized regression -- with d_hidden=16,384 vastly
exceeding n=14 training points, L1 drives most coefficients to exactly
zero, leaving a short list of features that COMBINE to predict the
target; this is the right tool for "several different features, each
explaining a subset of binders" in a way a single-feature test cannot
detect) on the 14-binder screen set ONLY, with nested leave-one-out CV
for an honest (but, given n=14, still noisy) R^2 on the training data
itself.

CRITICAL, directly informed by tonight's PC7 investigation: the fitted
model is evaluated on "the 4" confirm set with ZERO refitting or
re-optimization -- and scored with a genuinely SIGNED metric (Pearson
correlation between the model's actual predictions and the true labels),
not a sign-agnostic concordance. That distinction is exactly what caught
the PC7 direction bug in the first place (a sign-agnostic metric let the
original probe's "perfect concordance" mask predictions that were
actually strongly backwards, reg_pearson=-0.96 to -0.99 -- see
docs/pc7_sign_reversal_findings.md). Activity is control-ratio-normalized
throughout so the 14 and 4 are on a comparable scale.

Given n=14 (or n=4 for the actual generalization check) against 16,384
candidates, treat ANY result here as a lead, not a validated finding --
the training-set nested-CV R^2 in particular is close to meaningless in a
p>>n regime this extreme; the only check that matters is the never-fit
performance on the 4.

Usage:
    python lasso_probe_binding.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score

from data import center_scale
from sae_model import SparseAutoencoder

SAE_DIR = Path(__file__).resolve().parent
EMBED_DIR = Path("/Volumes/Extreme Pro/esmc_residue_embeddings/sae_binder_only/layer_09")
CHECKPOINT = SAE_DIR / "checkpoints" / "layer_09_run1" / "best.pt"
B44_ACTS_PATH = SAE_DIR / "b44_layer9_binder_residues.npy"

NEW14_LABELS_RAW = {
    "new149": 3.10, "new241": 4.92, "new226": 1.05, "new146": 2.72, "new208": 3.38,
    "new211": 0.92, "new249": 0.64, "new089": 5.12, "new174": 1.62, "new248": 2.80,
    "new163": 1.41, "new108": 4.03, "new228": 2.72, "new157": 9.19,
}
PREV4_LABELS_RAW = {"p130": 7.26, "p144": 9.72, "p154_prev": 6.44, "b44": 4.99}
CONTROL_NEW, CONTROL_PREV = 36.56, 72.8


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = pick_device()
    with open(SAE_DIR / "real_binder_row_indices.json") as f:
        real_binders = json.load(f)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean, scale = ckpt["mean"].to(device), ckpt["scale"]

    index_df = pd.read_csv(EMBED_DIR / "index.csv")
    index_by_row = {r.row_index: r for r in index_df.itertuples()}
    acts_file = np.load(EMBED_DIR / "activations.npy", mmap_mode="r")

    def encode_design(name):
        if name == "b44":
            acts = np.load(B44_ACTS_PATH).astype(np.float32)
        else:
            r = index_by_row[real_binders[name]]
            acts = np.array(acts_file[r.start : r.start + r.length]).astype(np.float32)
        x = torch.from_numpy(acts).to(device)
        x_proc = center_scale(x, mean, scale)
        with torch.no_grad():
            codes = model.encode(x_proc)
        return codes.max(dim=0).values.cpu().numpy(), codes.mean(dim=0).cpu().numpy()

    print("Encoding all 18 real binders...")
    max_pooled, mean_pooled = {}, {}
    for name in list(NEW14_LABELS_RAW.keys()) + list(PREV4_LABELS_RAW.keys()):
        mx, mn = encode_design(name)
        max_pooled[name], mean_pooled[name] = mx, mn

    new14_names = list(NEW14_LABELS_RAW.keys())
    prev4_names = list(PREV4_LABELS_RAW.keys())
    y14 = np.array([NEW14_LABELS_RAW[n] for n in new14_names]) / CONTROL_NEW
    y4 = np.array([PREV4_LABELS_RAW[n] for n in prev4_names]) / CONTROL_PREV

    stats = pd.read_csv(SAE_DIR / "feature_analysis_results" / "layer_09" / "feature_stats.csv")
    labels_df = pd.read_csv(SAE_DIR / "feature_analysis_results" / "layer_09" / "feature_labels.csv").set_index("feature")["label"]
    density = stats.set_index("feature")["density"]

    for pool_name, pooled in [("max", max_pooled), ("mean", mean_pooled)]:
        print(f"\n{'='*70}\n{pool_name}-pool\n{'='*70}")
        X14 = np.stack([pooled[n] for n in new14_names])  # (14, 16384)
        X4 = np.stack([pooled[n] for n in prev4_names])  # (4, 16384)

        loo = LeaveOneOut()
        probe = LassoCV(cv=loo, max_iter=20_000, n_jobs=-1, random_state=0)

        # Nested honest R^2 on the 14 itself -- caveat heavily in the writeup,
        # this is close to meaningless at n=14, p=16384. Outer loop is also
        # leave-one-out given how little data there is.
        print("Running nested leave-one-out CV on the 14 (this is slow: 14 outer folds, "
              "each fitting a full LassoCV with its own 13-fold inner LOO)...")
        oof_pred = cross_val_predict(probe, X14, y14, cv=loo, n_jobs=1)
        honest_r2 = r2_score(y14, oof_pred)
        honest_rho, honest_p = spearmanr(oof_pred, y14)
        print(f"  nested LOO-CV on the 14: R^2={honest_r2:+.3f}  spearman_rho={honest_rho:+.3f} (p={honest_p:.3f})")

        # Final model: fit on ALL 14, alpha chosen by its own internal LOO-CV.
        final_model = LassoCV(cv=loo, max_iter=20_000, n_jobs=-1, random_state=0).fit(X14, y14)
        nonzero = np.nonzero(final_model.coef_)[0]
        print(f"  final model (fit on all 14): alpha={final_model.alpha_:.6g}, {len(nonzero)} nonzero coefficients")

        # THE test that matters: apply this already-fitted model, completely
        # unchanged, to the 4 -- no refitting, no re-optimization. Score with
        # a genuinely SIGNED metric, not concordance.
        pred4 = final_model.predict(X4)
        pearson_r, pearson_p = pearsonr(pred4, y4)
        spearman_r, spearman_p = spearmanr(pred4, y4)
        r2_confirm = r2_score(y4, pred4)
        print(f"\n  === Generalization to the 4 (never touched during fitting) ===")
        print(f"  predictions: {dict(zip(prev4_names, pred4.round(4)))}")
        print(f"  true (normalized): {dict(zip(prev4_names, y4.round(4)))}")
        print(f"  pearson_r={pearson_r:+.3f} (p={pearson_p:.3f})   spearman_rho={spearman_r:+.3f} (p={spearman_p:.3f})   R^2={r2_confirm:+.3f}")

        if len(nonzero) > 0:
            print(f"\n  nonzero features (sorted by |coef|):")
            coef_df = pd.DataFrame({"feature": nonzero, "coef": final_model.coef_[nonzero]})
            coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
            for row in coef_df.itertuples():
                lbl = labels_df.get(row.feature, "?")
                dens = density.get(row.feature, float("nan"))
                print(f"    feature={row.feature:6d}  coef={row.coef:+.5f}  density={dens:.5f}")
                print(f"      {lbl[:180] if isinstance(lbl, str) else lbl}")

            out_path = SAE_DIR / "feature_analysis_results" / "layer_09" / f"lasso_probe_{pool_name}pool.csv"
            coef_df.to_csv(out_path, index=False)
            print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
