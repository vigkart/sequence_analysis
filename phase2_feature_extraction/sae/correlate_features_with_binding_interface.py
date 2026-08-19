"""Interface-restricted version of correlate_features_with_binding.py:
same locked screen-then-confirm methodology, but max/mean pooling is
computed only over each binder's own closest-quartile-to-target residues
(interface_utils.py) instead of the whole ~140-residue sequence -- the
same narrowing rationale as interface_restricted_feature_search.py,
applied to the full-dictionary statistical test rather than the
qualitative "surprise" search.

COVERAGE CAVEAT (read before interpreting results): only 14/14 screen
binders plus 1/4 confirm binders (`p144`) have verified predicted
structures in this repo -- see interface_utils.py's docstring for the
structure search that came up empty for `p130`, `p154_prev`, and `b44`.
The screen half of the methodology (rank 16,384 features by locked
concordance on the 14) is run exactly as before, at full strength. The
"confirm on an independent batch, sign locked" step CANNOT be run in its
original pairwise form with n=1 -- there are no pairs to compute
concordance over. Instead, p144's interface-pooled activation for each
top-screening feature is reported as a single-point sanity check
(direction-consistent or not with its own high activity), explicitly
NOT a statistical confirm test. Any feature this script calls
"promising" is therefore only screen-tested, one full rigor level below
the whole-binder version's screen-then-confirm result.

Usage:
    python correlate_features_with_binding_interface.py
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
P144_LABEL_RAW = 9.72
CONTROL_NEW, CONTROL_PREV = 36.56, 72.8
HIGH_THRESHOLD, HIGH_WEIGHT = 4.0, 3.0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def signed_weighted_fraction(scores: np.ndarray, labels: np.ndarray, pairs, weighted: bool) -> tuple[float, float]:
    """Same tie-aware concordance as correlate_features_with_binding.py --
    see that script's docstring for why ties must be excluded, not
    scored as fake 'perfect' agreement."""
    if weighted:
        w = np.where(labels >= HIGH_THRESHOLD, HIGH_WEIGHT, 1.0)
    else:
        w = np.ones(len(labels))
    num = den = 0.0
    for i, j in pairs:
        if scores[i] == scores[j]:
            continue
        pw = w[i] * w[j]
        num += pw * ((scores[i] - scores[j]) * (labels[i] - labels[j]) > 0)
        den += pw
    if den == 0:
        return 0.5, 0
    return num / den, den


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def main() -> None:
    device = pick_device()
    print(f"Device: {device}")

    with open(SAE_DIR / "real_binder_row_indices.json") as f:
        real_binders = json.load(f)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean, scale = ckpt["mean"].to(device), ckpt["scale"]
    print(f"Loaded SAE: d_hidden={model.d_hidden} k={model.k}, checkpoint epoch={ckpt['epoch']}")

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
    new14_labels_raw = np.array([NEW14_LABELS_RAW[n] for n in new14_names])
    new14_pairs = list(combinations(range(14), 2))

    results = {}
    for pool_name, pooled in [("max", interface_max_pooled), ("mean", interface_mean_pooled)]:
        X14 = np.stack([pooled[n] for n in new14_names])  # (14, d_hidden)
        x_p144 = pooled["p144"]
        d_hidden = X14.shape[1]

        wc14 = np.zeros(d_hidden)
        direction = np.zeros(d_hidden, dtype=int)
        screen_eff_pairs = np.zeros(d_hidden)
        for feat in range(d_hidden):
            f14, eff14 = signed_weighted_fraction(X14[:, feat], new14_labels_raw, new14_pairs, weighted=True)
            direction[feat] = 1 if f14 >= 0.5 else -1
            wc14[feat] = max(f14, 1 - f14)
            screen_eff_pairs[feat] = eff14

        evaluable = screen_eff_pairs > 0
        n_degenerate = int((~evaluable).sum())
        print(f"\n  {n_degenerate}/{d_hidden} features are degenerate on the interface-restricted 14 "
              f"(never fire at any interface residue / tied on all pairs) -- excluded from ranking")

        # Pooled, control-normalized Spearman across the 14 + p144 (15 points)
        # -- same weak/diagnostic-only role as the whole-binder version.
        from scipy.stats import spearmanr
        pooled_labels_norm = np.concatenate([new14_labels_raw / CONTROL_NEW, [P144_LABEL_RAW / CONTROL_PREV]])
        pooled_X = np.concatenate([X14, x_p144[None, :]], axis=0)
        rho = np.zeros(d_hidden)
        pval = np.zeros(d_hidden)
        for feat in range(d_hidden):
            if np.all(pooled_X[:, feat] == pooled_X[0, feat]):
                rho[feat], pval[feat] = 0.0, 1.0
                continue
            rho[feat], pval[feat] = spearmanr(pooled_X[:, feat], pooled_labels_norm)
        qval = benjamini_hochberg(pval)

        # p144 single-point check: is its interface activation, for each
        # feature, consistent with the screen-locked direction and with
        # its own high activity (9.72%, second-highest of all 18 real
        # binders)? NOT a statistical test (n=1) -- purely descriptive.
        p144_consistent = np.zeros(d_hidden, dtype=bool)
        for feat in range(d_hidden):
            if not evaluable[feat]:
                continue
            # "consistent" = p144's interface activation for this feature
            # is at or above the screen-set's own median in the locked
            # direction (a coarse, illustrative check only).
            screen_vals = X14[:, feat] * direction[feat]
            p144_consistent[feat] = (x_p144[feat] * direction[feat]) >= np.median(screen_vals)

        df = pd.DataFrame({
            "feature": np.arange(d_hidden),
            "screen_wc14_interface": wc14,
            "screen_eff_pairs": screen_eff_pairs,
            "direction": direction,
            "evaluable": evaluable,
            "p144_interface_act": x_p144,
            "p144_consistent_with_direction": p144_consistent,
            "pooled_rho_15pt": rho,
            "pooled_pval_15pt": pval,
            "pooled_qval_15pt": qval,
        })
        results[pool_name] = df
        out_path = SAE_DIR / "feature_analysis_results" / "layer_09" / f"binding_correlation_interface_{pool_name}pool.csv"
        df.sort_values("screen_wc14_interface", ascending=False).to_csv(out_path, index=False)
        print(f"  wrote {out_path}")

        eval_df = df[df["evaluable"]]
        n_perfect = int((eval_df["screen_wc14_interface"] == 1.0).sum())
        n_significant_fdr = int((df["pooled_qval_15pt"] < 0.05).sum())
        print(f"  === {pool_name}-pool (interface-restricted) ===")
        print(f"  of {len(eval_df)} evaluable features: {n_perfect} hit perfect screen_wc14 (1.0) on the 14")
        print(f"  features with pooled (15-point) BH-corrected q<0.05: {n_significant_fdr}/{d_hidden}")

        top = eval_df.sort_values("screen_wc14_interface", ascending=False).head(15)
        print(f"  top 15 by screen_wc14_interface (evaluable only):")
        for row in top.itertuples():
            print(f"    feature={row.feature:6d}  screen_wc14={row.screen_wc14_interface:.3f} "
                  f"({int(row.screen_eff_pairs)} eff.pairs)  dir={'+' if row.direction==1 else '-'}  "
                  f"p144_act={row.p144_interface_act:.2f} ({'consistent' if row.p144_consistent_with_direction else 'inconsistent'})  "
                  f"pooled_rho={row.pooled_rho_15pt:+.3f}  pooled_q={row.pooled_qval_15pt:.3f}")


if __name__ == "__main__":
    main()
