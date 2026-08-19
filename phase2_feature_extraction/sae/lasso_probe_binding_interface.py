"""Interface-restricted version of lasso_probe_binding.py: same LassoCV
multivariate probe, but max/mean pooling is computed only over each
binder's own closest-quartile-to-target residues (interface_utils.py)
instead of the whole sequence.

COVERAGE CAVEAT -- read before interpreting results (same as
correlate_features_with_binding_interface.py): only 14/14 screen binders
plus 1/4 confirm binders (`p144`) have verified structures; `p130`,
`p154_prev`, and `b44` do not (see interface_utils.py). The nested
leave-one-out CV on the 14 (training-set "honest" R^2) is run exactly as
before. The generalization check that mattered most in the original
script -- applying the fitted-and-frozen model to an INDEPENDENT batch
and scoring with a signed correlation -- cannot be computed the same way
with n=1: Pearson/Spearman need at least 2 points. p144's prediction vs.
its true value is reported as a single-point sanity check only (does the
model's prediction land on the correct side of the training set's
range, given p144 is one of the two highest-activity binders across all
18) -- explicitly NOT a statistical generalization test.

Usage:
    python lasso_probe_binding_interface.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score

from data import center_scale
from interface_utils import BINDER_SEQS, compute_interface_masks
from sae_model import SparseAutoencoder

SAE_DIR = Path(__file__).resolve().parent
EMBED_DIR = Path("/Volumes/Extreme Pro/esmc_residue_embeddings/sae_binder_only/layer_09")
CHECKPOINT = SAE_DIR / "checkpoints" / "layer_09_run1" / "best.pt"

NEW14_LABELS_RAW = {
    "new149": 3.10, "new241": 4.92, "new226": 1.05, "new146": 2.72, "new208": 3.38,
    "new211": 0.92, "new249": 0.64, "new089": 5.12, "new174": 1.62, "new248": 2.80,
    "new163": 1.41, "new108": 4.03, "new228": 2.72, "new157": 9.19,
}
P144_LABEL_RAW = 9.72
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

    print("Computing interface masks from predicted structures...")
    names = list(BINDER_SEQS.keys())  # 14 screen + p144
    interface_masks = compute_interface_masks(names)

    print("\nEncoding all 15 structurally-verified real binders through the SAE...")
    index_df = pd.read_csv(EMBED_DIR / "index.csv")
    index_by_row = {r.row_index: r for r in index_df.itertuples()}
    acts_file = np.load(EMBED_DIR / "activations.npy", mmap_mode="r")

    interface_max_pooled, interface_mean_pooled = {}, {}
    for name in names:
        r = index_by_row[real_binders[name]]
        acts = np.array(acts_file[r.start : r.start + r.length]).astype(np.float32)
        x = torch.from_numpy(acts).to(device)
        x_proc = center_scale(x, mean, scale)
        with torch.no_grad():
            codes = model.encode(x_proc).cpu().numpy()
        mask = interface_masks[name]
        interface_max_pooled[name] = codes[mask].max(axis=0)
        interface_mean_pooled[name] = codes[mask].mean(axis=0)

    new14_names = list(NEW14_LABELS_RAW.keys())
    y14 = np.array([NEW14_LABELS_RAW[n] for n in new14_names]) / CONTROL_NEW
    y_p144 = P144_LABEL_RAW / CONTROL_PREV

    stats = pd.read_csv(SAE_DIR / "feature_analysis_results" / "layer_09" / "feature_stats.csv")
    labels_df = pd.read_csv(SAE_DIR / "feature_analysis_results" / "layer_09" / "feature_labels.csv").set_index("feature")["label"]
    density = stats.set_index("feature")["density"]

    for pool_name, pooled in [("max", interface_max_pooled), ("mean", interface_mean_pooled)]:
        print(f"\n{'='*70}\n{pool_name}-pool (interface-restricted)\n{'='*70}")
        X14 = np.stack([pooled[n] for n in new14_names])  # (14, 16384)
        x_p144 = pooled["p144"][None, :]  # (1, 16384)

        loo = LeaveOneOut()
        probe = LassoCV(cv=loo, max_iter=20_000, n_jobs=-1, random_state=0)

        print("Running nested leave-one-out CV on the 14 (interface-pooled features)...")
        oof_pred = cross_val_predict(probe, X14, y14, cv=loo, n_jobs=1)
        honest_r2 = r2_score(y14, oof_pred)
        honest_rho, honest_p = spearmanr(oof_pred, y14)
        print(f"  nested LOO-CV on the 14: R^2={honest_r2:+.3f}  spearman_rho={honest_rho:+.3f} (p={honest_p:.3f})")

        final_model = LassoCV(cv=loo, max_iter=20_000, n_jobs=-1, random_state=0).fit(X14, y14)
        nonzero = np.nonzero(final_model.coef_)[0]
        print(f"  final model (fit on all 14): alpha={final_model.alpha_:.6g}, {len(nonzero)} nonzero coefficients")

        # Single-point check on p144 -- NOT a statistical generalization
        # test (n=1 gives no correlation to compute). Reported descriptively.
        pred_p144 = float(final_model.predict(x_p144)[0])
        print(f"\n  === Single-point check on p144 (never touched during fitting) ===")
        print(f"  predicted (normalized): {pred_p144:.4f}   true (normalized): {y_p144:.4f}")
        print(f"  training-set y14 range: [{y14.min():.4f}, {y14.max():.4f}], "
              f"p144's true value is the {'2nd-highest' if y_p144 > y14.max() else 'within-range'} of all 15 points")
        print(f"  NOTE: with a single confirm point there is no Pearson/Spearman to compute -- "
              f"this is a sanity check, not a validation.")

        if len(nonzero) > 0:
            print(f"\n  nonzero features (sorted by |coef|):")
            coef_df = pd.DataFrame({"feature": nonzero, "coef": final_model.coef_[nonzero]})
            coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
            for row in coef_df.itertuples():
                lbl = labels_df.get(row.feature, "?")
                dens = density.get(row.feature, float("nan"))
                print(f"    feature={row.feature:6d}  coef={row.coef:+.5f}  density={dens:.5f}")
                print(f"      {lbl[:180] if isinstance(lbl, str) else lbl}")

            out_path = SAE_DIR / "feature_analysis_results" / "layer_09" / f"lasso_probe_interface_{pool_name}pool.csv"
            coef_df.to_csv(out_path, index=False)
            print(f"\n  wrote {out_path}")
        else:
            print("\n  0 nonzero coefficients -- internal CV declined to fit anything.")


if __name__ == "__main__":
    main()
