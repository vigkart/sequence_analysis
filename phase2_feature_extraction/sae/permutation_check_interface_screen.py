"""Is correlate_features_with_binding_interface.py's raw count of features
hitting 'perfect' screen concordance (wc14==1.0) on the interface-
restricted 14-binder screen actually enriched over chance, or -- like the
original PC7 sign-agnostic bug and the whole-binder SAE screen before it
-- just what noise produces at this scale? A raw count is not evidence by
itself; every other statistical claim in this project has been checked
against a chance baseline (docs/pc7_sign_reversal_findings.md,
docs/sae_feature_labeling_findings.md), and this one needs the same
treatment, ESPECIALLY because interface-restricted pooling makes far more
designs exactly 0.0 at a given feature (only ~35 of ~140 residues
contribute), which mechanically produces smaller effective-pair counts
and makes hitting "perfect" concordance easier by pure luck than in the
whole-binder version.

Method: fix the interface-pooled activation matrix (14 designs x 16,384
features, real, unpermuted), then repeatedly shuffle the WHICH-DESIGN-
GOT-WHICH-LABEL assignment (200 shuffles) and recompute, for every
feature, whether the shuffled screen also hits wc14==1.0 -- vectorized
across all features per shuffle so this runs in seconds, not hours.
Compares the observed count against this null distribution's mean/std,
z-score, and empirical p-value.

Usage:
    python permutation_check_interface_screen.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
N_PERM = 200


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

    interface_masks = compute_interface_masks(list(BINDER_SEQS.keys()))

    index_df = pd.read_csv(EMBED_DIR / "index.csv")
    index_by_row = {r.row_index: r for r in index_df.itertuples()}
    acts_file = np.load(EMBED_DIR / "activations.npy", mmap_mode="r")

    new14_names = list(NEW14_LABELS_RAW.keys())
    y_true = np.array([NEW14_LABELS_RAW[n] for n in new14_names])

    pooled_by_type = {"max": {}, "mean": {}}
    for name in new14_names:
        r = index_by_row[real_binders[name]]
        acts = np.array(acts_file[r.start : r.start + r.length]).astype(np.float32)
        x = torch.from_numpy(acts).to(device)
        x_proc = center_scale(x, mean, scale)
        with torch.no_grad():
            codes = model.encode(x_proc).cpu().numpy()
        mask = interface_masks[name]
        pooled_by_type["max"][name] = codes[mask].max(axis=0)
        pooled_by_type["mean"][name] = codes[mask].mean(axis=0)

    pairs = list(combinations(range(14), 2))
    i_idx = np.array([p[0] for p in pairs])
    j_idx = np.array([p[1] for p in pairs])
    rng = np.random.default_rng(0)

    for pool_name in ["max", "mean"]:
        X14 = np.stack([pooled_by_type[pool_name][n] for n in new14_names])
        S = np.sign(X14[i_idx] - X14[j_idx])
        eff_count = (S != 0).sum(axis=0)
        evaluable = eff_count > 0
        S_eval = S[:, evaluable]
        eff_eval = eff_count[evaluable]
        n_eval = int(evaluable.sum())

        label_sign_true = np.sign(y_true[i_idx] - y_true[j_idx])
        agree_true = (S_eval == label_sign_true[:, None]) & (S_eval != 0)
        frac_true = agree_true.sum(axis=0) / eff_eval
        wc_true = np.maximum(frac_true, 1 - frac_true)
        n_perfect_obs = int(np.isclose(wc_true, 1.0).sum())

        perfect_counts = np.zeros(N_PERM, dtype=int)
        for p in range(N_PERM):
            y_perm = y_true[rng.permutation(14)]
            label_sign = np.sign(y_perm[i_idx] - y_perm[j_idx])
            agree = (S_eval == label_sign[:, None]) & (S_eval != 0)
            frac = agree.sum(axis=0) / eff_eval
            wc = np.maximum(frac, 1 - frac)
            perfect_counts[p] = int(np.isclose(wc, 1.0).sum())

        z = (n_perfect_obs - perfect_counts.mean()) / perfect_counts.std() if perfect_counts.std() > 0 else float("nan")
        p_empirical = (perfect_counts >= n_perfect_obs).mean()
        print(f"\n=== {pool_name}-pool interface-restricted screen (n={n_eval} evaluable features) ===")
        print(f"  observed 'perfect' (wc14==1.0): {n_perfect_obs}")
        print(f"  chance ({N_PERM}-shuffle permutation null): mean={perfect_counts.mean():.1f} "
              f"std={perfect_counts.std():.1f} range=[{perfect_counts.min()},{perfect_counts.max()}]")
        print(f"  ratio observed/chance = {n_perfect_obs / perfect_counts.mean():.2f}x   z={z:.2f}   "
              f"empirical p(chance >= observed) = {p_empirical:.3f}")


if __name__ == "__main__":
    main()
