"""Different question than correlate_features_with_binding.py (which
requires a feature to positively correlate with activity across the WHOLE
set -- null at n=14/4) and inspect_top_binder_features.py (which just
looks at what's distinctive on strong binders alone, no comparison): does
each strong binder's distinctive feature also fire on the WEAK binders?
If so, it's not actually discriminating strong from weak -- just generic
background activity that happened to look "distinctive" (rare + intense)
on one specific design. If a feature stays silent (never fires) across
every weak binder while firing strongly on its source strong binder, it
survives this elimination test -- still not statistically proven at this
n, but a meaningfully better-supported lead than one that fires on both.

Usage:
    python discriminate_strong_vs_weak_features.py
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

STRONG = {"new157": 9.19, "new089": 5.12, "new241": 4.92, "p144": 9.72}
WEAK = {"new249": 0.64, "new211": 0.92, "new226": 1.05, "new163": 1.41, "b44": 4.99}
TOP_N = 12


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(name, real_binders, index_by_row, acts_file, model, mean, scale, device):
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

    print("Encoding all strong + weak binders...")
    max_act_by_name = {}
    for name in list(STRONG.keys()) + list(WEAK.keys()):
        max_act_by_name[name] = encode(name, real_binders, index_by_row, acts_file, model, mean, scale, device)

    # Re-derive each strong binder's top-N distinctive features (same
    # "surprise" definition as inspect_top_binder_features.py).
    print(f"\n{'='*100}")
    survivors = []
    for name in STRONG:
        max_act = max_act_by_name[name]
        active = np.nonzero(max_act)[0]
        dens = density.reindex(active).to_numpy()
        mwa = mean_when_active.reindex(active).to_numpy()
        mwa_safe = np.where(mwa > 0, mwa, 1.0)
        surprise = (1.0 / np.maximum(dens, 1e-6)) * (max_act[active] / mwa_safe)
        order = np.argsort(-surprise)
        top = active[order][:TOP_N]

        print(f"\n{name} (activity={STRONG[name]}%) -- checking its top-{TOP_N} distinctive features against {len(WEAK)} weak binders:")
        for feat in top:
            weak_acts = {wn: float(max_act_by_name[wn][feat]) for wn in WEAK}
            n_fires_on_weak = sum(1 for v in weak_acts.values() if v > 0)
            status = "SILENT on all weak binders (survives)" if n_fires_on_weak == 0 else f"fires on {n_fires_on_weak}/{len(WEAK)} weak binders (not discriminating)"
            if n_fires_on_weak == 0:
                survivors.append((name, feat, max_act[feat]))
            weak_str = ", ".join(f"{wn}={v:.2f}" for wn, v in weak_acts.items())
            print(f"  feature={feat:6d}  strong_act={max_act[feat]:.2f}  |  weak: {weak_str}  -> {status}")

    print(f"\n{'='*100}\nSURVIVORS: features distinctive on a strong binder AND silent on ALL {len(WEAK)} weak binders\n{'='*100}")
    if not survivors:
        print("  None.")
    else:
        for name, feat, act in survivors:
            lbl = labels_df.get(feat, "?")
            print(f"  {name}: feature={feat}  activation={act:.2f}  density={density.get(feat, float('nan')):.5f}")
            print(f"    {lbl[:200] if isinstance(lbl, str) else lbl}")


if __name__ == "__main__":
    main()
