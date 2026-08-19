"""Does any SAE feature represent binding? Correlates every one of the
16,384 layer-9 features' pooled activation against the real, wet-lab
mCherry activity data for all 18 real binders -- using the SAME rigor
established for the PC7 investigation (see docs/pc7_sign_reversal_findings.md
and docs/PCA_analysis_full_details.md), not the sign-agnostic mistake that
investigation started from:

  - Direction is determined ONLY on the 14-binder screen set, then LOCKED
    and applied unchanged to the 4-binder confirm set -- no independent
    re-optimization of direction on "held out" data.
  - Activity values are control-ratio-normalized (X = mCherry / own
    experiment's positive control) before pooling across the two batches,
    per the batch-scale-shift finding.
  - 16,384 candidates against 18 real points is an even more severe
    multiple-comparisons regime than PC7's 4,320-candidate sweep --
    Benjamini-Hochberg FDR correction is applied, and results are reported
    with the same explicit skepticism the PC7 sweep required (a strong-
    looking single feature out of 16,384 tries is not automatically real).

b44 (the one real binder not in the 80k pool) is included via a freshly
encoded activation file (encode_b44_layer9.py) -- all 18 real binders are
used, not 17.

Usage:
    python correlate_features_with_binding.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import center_scale
from sae_model import SparseAutoencoder

MAIN = Path(__file__).resolve().parent.parent.parent
SAE_DIR = Path(__file__).resolve().parent
EMBED_DIR = Path("/Volumes/Extreme Pro/esmc_residue_embeddings/sae_binder_only/layer_09")
CHECKPOINT = SAE_DIR / "checkpoints" / "layer_09_run1" / "best.pt"
B44_ACTS_PATH = SAE_DIR / "b44_layer9_binder_residues.npy"

# Raw % mCherry, no-TET condition -- same values used throughout tonight's
# PC7 investigation, cross-checked against the full wet-lab tables.
NEW14_LABELS_RAW = {
    "new149": 3.10, "new241": 4.92, "new226": 1.05, "new146": 2.72, "new208": 3.38,
    "new211": 0.92, "new249": 0.64, "new089": 5.12, "new174": 1.62, "new248": 2.80,
    "new163": 1.41, "new108": 4.03, "new228": 2.72, "new157": 9.19,
}
PREV4_LABELS_RAW = {"p130": 7.26, "p144": 9.72, "p154_prev": 6.44, "b44": 4.99}
CONTROL_NEW, CONTROL_PREV = 36.56, 72.8
HIGH_THRESHOLD, HIGH_WEIGHT = 4.0, 3.0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_design_activations(row_indices: dict) -> dict[str, np.ndarray]:
    """name -> (n_residues, 1152) float32 raw layer-9 joint activations,
    for the binder residues only."""
    index_df = pd.read_csv(EMBED_DIR / "index.csv")
    index_by_row = {r.row_index: r for r in index_df.itertuples()}
    acts = np.load(EMBED_DIR / "activations.npy", mmap_mode="r")

    out = {}
    for name, row_index in row_indices.items():
        r = index_by_row[row_index]
        out[name] = np.array(acts[r.start : r.start + r.length]).astype(np.float32)
    out["b44"] = np.load(B44_ACTS_PATH).astype(np.float32)
    return out


def pool_codes(model, mean, scale, device, acts: np.ndarray, batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Encodes every residue of one design, returns (max_pooled, mean_pooled)
    each (d_hidden,)."""
    codes_parts = []
    for start in range(0, acts.shape[0], batch_size):
        x = torch.from_numpy(acts[start : start + batch_size]).to(device)
        x_proc = center_scale(x, mean, scale)
        with torch.no_grad():
            codes = model.encode(x_proc)
        codes_parts.append(codes.cpu())
    codes = torch.cat(codes_parts, dim=0)
    return codes.max(dim=0).values.numpy(), codes.mean(dim=0).numpy()


def signed_weighted_fraction(scores: np.ndarray, labels: np.ndarray, pairs, weighted: bool) -> float:
    """Returns (concordant_frac, effective_pairs) where effective_pairs
    excludes ties (scores[i]==scores[j]) from the denominator -- NOT the
    same as the PC7 version of this function, which never needed this: SAE
    feature activations are sparse (most features are exactly 0 for most
    residues, so max/mean-pooled per-design activation is very often
    exactly 0.0 for many designs at once), so ties are common here in a way
    they essentially never are for continuous PCA scores. Without this fix,
    an always-inactive feature (all ties) gets frac=0/0-treated-as-0, and
    max(frac, 1-frac) trivially reports a fake "perfect" 1.0 -- degenerate,
    not evidence of anything. A feature with too few non-tied pairs to
    evaluate is flagged via effective_pairs, not silently scored as if it
    were informative."""
    if weighted:
        w = np.where(labels >= HIGH_THRESHOLD, HIGH_WEIGHT, 1.0)
    else:
        w = np.ones(len(labels))
    num = den = 0.0
    for i, j in pairs:
        if scores[i] == scores[j]:
            continue  # tie -- excluded entirely, not counted as concordant or discordant
        pw = w[i] * w[j]
        num += pw * ((scores[i] - scores[j]) * (labels[i] - labels[j]) > 0)
        den += pw
    if den == 0:
        return 0.5, 0  # fully degenerate (constant feature) -- report as uninformative, not "perfect"
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
        real_binders = json.load(f)  # name -> row_index, 17 entries (no b44)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = ckpt["mean"].to(device)
    scale = ckpt["scale"]
    print(f"Loaded SAE: d_hidden={model.d_hidden} k={model.k}, checkpoint epoch={ckpt['epoch']}")

    print("Loading + encoding all 18 real binders...")
    raw_acts = load_design_activations(real_binders)
    max_pooled, mean_pooled = {}, {}
    for name, acts in raw_acts.items():
        mx, mn = pool_codes(model, mean, scale, device, acts)
        max_pooled[name] = mx
        mean_pooled[name] = mn
    print(f"Encoded {len(max_pooled)} designs.")

    new14_names = list(NEW14_LABELS_RAW.keys())
    prev4_names = list(PREV4_LABELS_RAW.keys())
    new14_labels_raw = np.array([NEW14_LABELS_RAW[n] for n in new14_names])
    prev4_labels_raw = np.array([PREV4_LABELS_RAW[n] for n in prev4_names])
    new14_pairs = list(combinations(range(14), 2))
    prev4_pairs = list(combinations(range(4), 2))

    results = {}
    for pool_name, pooled in [("max", max_pooled), ("mean", mean_pooled)]:
        X14 = np.stack([pooled[n] for n in new14_names])  # (14, d_hidden)
        X4 = np.stack([pooled[n] for n in prev4_names])  # (4, d_hidden)
        d_hidden = X14.shape[1]

        wc14 = np.zeros(d_hidden)
        direction = np.zeros(d_hidden, dtype=int)
        locked4 = np.zeros(d_hidden)
        screen_eff_pairs = np.zeros(d_hidden)
        confirm_eff_pairs = np.zeros(d_hidden)
        for feat in range(d_hidden):
            f14, eff14 = signed_weighted_fraction(X14[:, feat], new14_labels_raw, new14_pairs, weighted=True)
            direction[feat] = 1 if f14 >= 0.5 else -1
            wc14[feat] = max(f14, 1 - f14)
            screen_eff_pairs[feat] = eff14
            f4, eff4 = signed_weighted_fraction(X4[:, feat], prev4_labels_raw, prev4_pairs, weighted=False)
            locked4[feat] = f4 if direction[feat] == 1 else (1 - f4)
            confirm_eff_pairs[feat] = eff4

        # Features with zero effective (non-tied) pairs on either set are
        # fully degenerate (e.g. never fires on any of these 18 designs at
        # all) -- their wc14/locked4 values are meaningless placeholders
        # (0.5), not genuine scores. Flagged via *_eff_pairs, not silently
        # ranked as if informative.
        evaluable = (screen_eff_pairs > 0) & (confirm_eff_pairs > 0)
        n_degenerate = int((~evaluable).sum())
        print(f"  {n_degenerate}/{d_hidden} features are degenerate (never fire / tied on all pairs "
              f"in the screen or confirm set) -- excluded from ranking, not scored as 'perfect'")

        # Pooled, control-normalized Spearman correlation across all 18 (same
        # diagnostic used for PC7 -- weak evidence on its own, not a
        # substitute for the locked-confirm test above).
        from scipy.stats import spearmanr
        pooled_labels_norm = np.concatenate([new14_labels_raw / CONTROL_NEW, prev4_labels_raw / CONTROL_PREV])
        pooled_X = np.concatenate([X14, X4], axis=0)
        rho = np.zeros(d_hidden)
        pval = np.zeros(d_hidden)
        for feat in range(d_hidden):
            if np.all(pooled_X[:, feat] == pooled_X[0, feat]):
                rho[feat], pval[feat] = 0.0, 1.0
                continue
            rho[feat], pval[feat] = spearmanr(pooled_X[:, feat], pooled_labels_norm)
        qval = benjamini_hochberg(pval)

        df = pd.DataFrame({
            "feature": np.arange(d_hidden),
            "screen_wc14": wc14,
            "screen_eff_pairs": screen_eff_pairs,
            "direction": direction,
            "confirm_locked": locked4,
            "confirm_eff_pairs": confirm_eff_pairs,
            "evaluable": evaluable,
            "pooled_rho": rho,
            "pooled_pval": pval,
            "pooled_qval": qval,
        })
        results[pool_name] = df
        out_path = SAE_DIR / "feature_analysis_results" / "layer_09" / f"binding_correlation_{pool_name}pool.csv"
        df.sort_values("screen_wc14", ascending=False).to_csv(out_path, index=False)
        print(f"\n=== {pool_name}-pool: wrote {out_path} ===")

        eval_df = df[df["evaluable"]]
        n_genuinely_confirmed = int((eval_df["confirm_locked"] == 1.0).sum())
        n_significant_fdr = int((df["pooled_qval"] < 0.05).sum())
        print(f"  of {len(eval_df)} evaluable (non-degenerate) features: "
              f"{n_genuinely_confirmed} have confirm_locked==1.0 ('genuinely confirmed')")
        print(f"  features with pooled BH-corrected q<0.05: {n_significant_fdr}/{d_hidden}")

        top = eval_df.sort_values("screen_wc14", ascending=False).head(10)
        print(f"  top 10 by screen_wc14 (evaluable only):")
        for row in top.itertuples():
            print(f"    feature={row.feature:6d}  screen_wc14={row.screen_wc14:.3f} ({int(row.screen_eff_pairs)} eff.pairs)  "
                  f"dir={'+' if row.direction==1 else '-'}  confirm_locked={row.confirm_locked:.3f} ({int(row.confirm_eff_pairs)} eff.pairs)  "
                  f"pooled_rho={row.pooled_rho:+.3f}  pooled_q={row.pooled_qval:.3f}")


if __name__ == "__main__":
    main()
