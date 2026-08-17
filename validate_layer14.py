import csv
from pathlib import Path

import numpy as np

CSV_PATH = "boltzgen_conf_1_batch_generation-scores.csv"
IDS_PATH = "stageA_mean_pooled/ids.csv"
PCA_DIR = Path("stageA_pca")
N_PCS = 20
VARIANTS = ["correlation_all_dims", "covariance_all_dims", "covariance_outliers_excluded"]
CONDITIONS = ["binder_alone", "joint"]

# --- Load iptm + length, aligned by row order ---
iptm, lengths, csv_ids = [], [], []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        v = row["design_to_target_iptm - boltzgen"]
        iptm.append(float(v) if v not in ("", "nan") else np.nan)
        lengths.append(len(row["designed_sequence - boltzgen"]))
        csv_ids.append(row["id - boltzgen"])
iptm = np.array(iptm, dtype=np.float64)
lengths = np.array(lengths, dtype=np.float64)
n = len(iptm)

with open(IDS_PATH, newline="") as f:
    assert [r["id"] for r in csv.DictReader(f)] == csv_ids
print(f"loaded {n} rows, alignment OK\n")


def pearson(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    xc, yc = x - x.mean(), y - y.mean()
    return (xc @ yc) / np.sqrt((xc ** 2).sum() * (yc ** 2).sum())


def partial_corr(pc_col, y, control):
    r_xy = pearson(pc_col, y)
    r_xc = pearson(pc_col, control)
    r_yc = pearson(y, control)
    denom = np.sqrt((1 - r_xc ** 2) * (1 - r_yc ** 2))
    return (r_xy - r_xc * r_yc) / denom if denom > 1e-9 else np.nan


# ============================================================
# CHECK 1: length-controlled (partial) correlation, per layer
# ============================================================
print("=" * 70)
print("CHECK 1: partial correlation with iptm, controlling for length")
print("=" * 70)

for condition in CONDITIONS:
    print(f"\n--- {condition} ---")
    print(f"{'layer':6s} {'raw best r':12s} {'partial r (same PC)':20s} {'shrinkage':10s}")
    for layer in range(36):
        layer_dir = PCA_DIR / condition / f"layer_{layer:02d}"
        best_raw, best_partial, best_variant, best_pc = 0.0, None, None, None
        for variant in VARIANTS:
            scores = np.load(layer_dir / f"{variant}_scores.npy")[:, :N_PCS]
            for pc in range(scores.shape[1]):
                r_raw = pearson(scores[:, pc], iptm)
                if abs(r_raw) > abs(best_raw):
                    best_raw = r_raw
                    best_variant, best_pc = variant, pc

        scores = np.load(layer_dir / f"{best_variant}_scores.npy")[:, :N_PCS]
        best_partial = partial_corr(scores[:, best_pc], iptm, lengths)
        shrink = abs(best_raw) - abs(best_partial)
        marker = "  <-- layer 14" if layer == 14 else ""
        print(f"{layer:<6d} {best_raw:+.3f}       {best_partial:+.3f}                {shrink:+.3f}{marker}")

# ============================================================
# CHECK 2: train/test split replication
# ============================================================
print("\n" + "=" * 70)
print("CHECK 2: does the best-layer selection replicate on held-out data?")
print("=" * 70)

rng = np.random.default_rng(0)
perm = rng.permutation(n)
half_a, half_b = perm[: n // 2], perm[n // 2 :]


def search_best(select_idx):
    """Search all (condition, layer, variant, pc) for max |r| with iptm on select_idx rows."""
    best = (0.0, None, None, None, None)
    for condition in CONDITIONS:
        for layer in range(36):
            layer_dir = PCA_DIR / condition / f"layer_{layer:02d}"
            for variant in VARIANTS:
                scores = np.load(layer_dir / f"{variant}_scores.npy")[:, :N_PCS]
                for pc in range(scores.shape[1]):
                    r = pearson(scores[select_idx, pc], iptm[select_idx])
                    if abs(r) > abs(best[0]):
                        best = (r, condition, layer, variant, pc)
    return best


def evaluate(condition, layer, variant, pc, eval_idx):
    scores = np.load(PCA_DIR / condition / f"layer_{layer:02d}" / f"{variant}_scores.npy")[:, pc]
    return pearson(scores[eval_idx], iptm[eval_idx])


for select_name, select_idx, eval_name, eval_idx in [
    ("half A", half_a, "half B", half_b),
    ("half B", half_b, "half A", half_a),
]:
    r_select, condition, layer, variant, pc = search_best(select_idx)
    r_eval = evaluate(condition, layer, variant, pc, eval_idx)
    print(f"\nselected on {select_name}: best = {condition} layer {layer} {variant} PC{pc}, r={r_select:+.3f}")
    print(f"same axis evaluated on {eval_name} (held out): r={r_eval:+.3f}")
    print(f"is this layer 14? {'YES' if layer == 14 else f'NO — layer {layer} instead'}")

print("\ndone")
