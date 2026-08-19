"""Qualitative deep-dive: what's distinctive, feature-wise, about our
specific strongest binders -- not a correlation/significance test (that's
correlate_features_with_binding.py and lasso_probe_binding.py, both null
at n=14/4), just "what does this particular successful design's SAE
encoding actually look like."

For each target binder, ranks its active (nonzero max-pooled) features by
a "surprise" score (rarity x intensity-relative-to-typical, same
definition as correlate_distinctiveness_with_binding.py) and reports the
top N with their LLM labels, InterPro evidence (where available), and
density. Also checks for feature overlap across the target binders --
informal, not a substitute for statistical testing, but directly
observable without needing significance.

Usage:
    python inspect_top_binder_features.py
"""

import json
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

# Strongest 3 from the 14-binder screen experiment (by raw % mCherry --
# same ranking as control-normalized, since a constant divisor doesn't
# change within-experiment order), plus the strongest from the 4-binder
# confirm experiment.
TARGETS = {
    "new157": 9.19, "new089": 5.12, "new241": 4.92,  # top 3 of the 14
    "p144": 9.72,  # top 1 of the 4
}
TOP_N = 12


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

    stats = pd.read_csv(SAE_DIR / "feature_analysis_results" / "layer_09" / "feature_stats.csv")
    density = stats.set_index("feature")["density"]
    mean_when_active = stats.set_index("feature")["mean_activation_when_active"]
    labels_df = pd.read_csv(SAE_DIR / "feature_analysis_results" / "layer_09" / "feature_labels.csv").set_index("feature")["label"]
    interpro = pd.read_csv(SAE_DIR / "feature_analysis_results" / "layer_09" / "interpro_annotations.csv")
    interpro_by_feat = interpro.groupby("feature")["interpro_annotation"].apply(lambda s: s.dropna().tolist())

    top_features_by_binder = {}
    for name in TARGETS:
        if name == "b44":
            acts = np.load(B44_ACTS_PATH).astype(np.float32)
        else:
            r = index_by_row[real_binders[name]]
            acts = np.array(acts_file[r.start : r.start + r.length]).astype(np.float32)
        x = torch.from_numpy(acts).to(device)
        x_proc = center_scale(x, mean, scale)
        with torch.no_grad():
            codes = model.encode(x_proc)
        max_act = codes.max(dim=0).values.cpu().numpy()

        active = np.nonzero(max_act)[0]
        dens = density.reindex(active).to_numpy()
        mwa = mean_when_active.reindex(active).to_numpy()
        mwa_safe = np.where(mwa > 0, mwa, 1.0)
        surprise = (1.0 / np.maximum(dens, 1e-6)) * (max_act[active] / mwa_safe)
        order = np.argsort(-surprise)
        top = active[order][:TOP_N]
        top_features_by_binder[name] = set(top.tolist())

        print(f"\n{'='*80}\n{name}  (activity={TARGETS[name]}%, {len(active)} active features total)\n{'='*80}")
        for feat in top:
            lbl = labels_df.get(feat, "?")
            d = density.get(feat, float("nan"))
            ip = interpro_by_feat.get(feat, [])
            ip_str = f"  [InterPro: {ip}]" if ip else ""
            print(f"  feature={feat:6d}  density={d:.5f}  max_act={max_act[feat]:.2f}{ip_str}")
            print(f"    {lbl[:200] if isinstance(lbl, str) else lbl}")

    print(f"\n{'='*80}\nOverlap among these {len(TARGETS)} binders' top-{TOP_N} distinctive features\n{'='*80}")
    names = list(TARGETS.keys())
    from itertools import combinations
    any_overlap = False
    for n1, n2 in combinations(names, 2):
        shared = top_features_by_binder[n1] & top_features_by_binder[n2]
        if shared:
            any_overlap = True
            print(f"  {n1} & {n2} share: {sorted(shared)}")
            for feat in sorted(shared):
                print(f"    feature={feat}: {labels_df.get(feat, '?')[:150]}")
    if not any_overlap:
        print("  No overlap at all -- every one of these 4 binders' top distinctive features is unique to it.")


if __name__ == "__main__":
    main()
