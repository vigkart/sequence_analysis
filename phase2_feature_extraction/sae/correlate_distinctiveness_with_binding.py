"""Tests a different hypothesis than correlate_features_with_binding.py:
not "does one SPECIFIC feature explain binding globally" (tested there,
came back null), but "does having SOME unusually rare/intense feature
active -- regardless of which one -- correlate with binding?" This is the
right test if binding is achieved via multiple structurally distinct
solutions (different binders using different motifs), where no single
feature would show a global correlation even if each subgroup's own
feature is a real, strong, causal driver within that subgroup.

Per-design "distinctiveness" score: for each of a design's ACTIVE
(nonzero max-pooled) features, compute how surprising that activation is
-- rarity (1/density, from feature_stats.csv) times intensity relative to
the feature's own typical activation-when-active (from the same file) --
then sum across the design's active features. A design with several
rare features firing unusually strongly gets a high score; a design
whose active features are all common, unremarkable ones gets a low
score.

Uses the SAME rigor as everything else tonight: direction locked on the
14-binder screen, applied unchanged to the 4-binder confirm set,
activity control-ratio-normalized.

Usage:
    python correlate_distinctiveness_with_binding.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
HIGH_THRESHOLD, HIGH_WEIGHT = 4.0, 3.0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def signed_weighted_fraction(scores, labels, pairs, weighted):
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


def main() -> None:
    device = pick_device()
    with open(SAE_DIR / "real_binder_row_indices.json") as f:
        real_binders = json.load(f)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean, scale = ckpt["mean"].to(device), ckpt["scale"]

    stats = pd.read_csv(SAE_DIR / "feature_analysis_results" / "layer_09" / "feature_stats.csv")
    density = stats.set_index("feature")["density"].to_numpy()
    mean_when_active = stats.set_index("feature")["mean_activation_when_active"].to_numpy()
    mean_when_active_safe = np.where(mean_when_active > 0, mean_when_active, 1.0)

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
        return codes.max(dim=0).values.cpu().numpy()

    scores = {}
    for name in list(NEW14_LABELS_RAW.keys()) + list(PREV4_LABELS_RAW.keys()):
        max_act = encode_design(name)
        active = max_act > 0
        rarity = 1.0 / np.maximum(density[active], 1e-6)
        intensity = max_act[active] / mean_when_active_safe[active]
        surprise = rarity * intensity
        scores[name] = {
            "distinctiveness_sum": float(surprise.sum()),
            "distinctiveness_max": float(surprise.max()) if surprise.size else 0.0,
            "n_active": int(active.sum()),
            "n_rare_active": int((density[active] < 0.001).sum()),
        }
        print(f"{name:12s} n_active={scores[name]['n_active']:5d}  n_rare(<0.1%%)={scores[name]['n_rare_active']:3d}  "
              f"distinctiveness_sum={scores[name]['distinctiveness_sum']:10.1f}  distinctiveness_max={scores[name]['distinctiveness_max']:8.1f}")

    new14_names = list(NEW14_LABELS_RAW.keys())
    prev4_names = list(PREV4_LABELS_RAW.keys())
    new14_labels = np.array([NEW14_LABELS_RAW[n] for n in new14_names])
    prev4_labels = np.array([PREV4_LABELS_RAW[n] for n in prev4_names])
    new14_pairs = list(combinations(range(14), 2))
    prev4_pairs = list(combinations(range(4), 2))

    print("\n=== Testing each distinctiveness metric against real activity (locked direction) ===")
    for metric in ["distinctiveness_sum", "distinctiveness_max", "n_active", "n_rare_active"]:
        s14 = np.array([scores[n][metric] for n in new14_names])
        s4 = np.array([scores[n][metric] for n in prev4_names])
        f14, eff14 = signed_weighted_fraction(s14, new14_labels, new14_pairs, weighted=True)
        direction = 1 if f14 >= 0.5 else -1
        wc14 = max(f14, 1 - f14)
        f4, eff4 = signed_weighted_fraction(s4, prev4_labels, prev4_pairs, weighted=False)
        locked4 = f4 if direction == 1 else (1 - f4)
        print(f"  {metric:22s}  screen_wc14={wc14:.3f} ({eff14:.0f} eff.pairs, dir={'+' if direction==1 else '-'})  "
              f"confirm_locked={locked4:.3f} ({eff4:.0f} eff.pairs)")


if __name__ == "__main__":
    main()
