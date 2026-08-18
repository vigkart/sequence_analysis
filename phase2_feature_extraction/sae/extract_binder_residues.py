"""One-time extraction: pulls binder-only residues (drops target + special
tokens) out of the big external memmap
(/Volumes/Extreme Pro/esmc_residue_embeddings/layers_7_8_9_joint.f32,
~175GB, all 80k designs x layers 7/8/9 x binder+target+specials) into
smaller per-layer local files sized for SAE training on this machine.

Also builds and saves the train/val/test SPLIT (at the design level, shared
identically across all 3 layers so the same design always lands in the same
split regardless of which layer's SAE is being trained) -- done ONCE here,
not re-derived per layer, so downstream training/eval is guaranteed
consistent across layers.

Split design (see docs/TODO.md's SAE-training discussion for the reasoning):
  - ALL 80,000 designs get a split label -- nothing is discarded. First,
    --test-frac (default 0.20) of the FULL 80k is held out as TEST, per PC7
    decile so it spans the full binding-relevant score range. The remaining
    80% ("trainval", 64,000 designs) is then split again by --val-frac
    (default 0.10, applied to the trainval pool, not the full 80k) into
    TRAIN (57,600) and VAL (6,400).
  - Why this two-stage shape, not a flat 3-way split: only TRAIN+VAL need to
    be RAM-resident during training (data.py's load_dataset only loads
    those two) -- 64,000 designs/layer is ~18.4GB (fp16), comfortable on
    this machine's 36GB with nothing else running. TEST (16,000 designs,
    a full 20% of the real data, not a small slice of a shrunken pool) is
    extracted here too so it's ready on disk, but is never loaded into RAM
    until feature_analysis.py's final evaluation step, so it never competes
    with training for memory.
  - Stratified by PC7 score DECILE (layer 9, joint, covariance_all_dims
    variant -- the axis established in phase1_layer_selection/screen_then_confirm.py
    as most predictive of real binding activity) at BOTH split stages, so
    train/val/test each spans the full binding-relevant score range rather
    than leaving that to chance. NOT stratified by design batch (Job Name,
    320 batches x 250 designs) -- batches are BoltzGen generation-job
    partitions, not distinct experimental conditions, so there's no reason
    to expect PC7 score to vary systematically by batch; random sampling
    within each PC7 decile already spreads batch membership widely given
    thousands of designs per decile.

Output (per layer, to the external drive -- plenty of free space there vs.
~44GB free locally):
    sae_binder_only/design_split.csv   -- id, row_index, pc7_score, split
                                           (shared across all 3 layers)
    sae_binder_only/layer_0N/activations.npy  -- (n_residues, 1152) float16,
                                           ALL 80k designs' binder residues
    sae_binder_only/layer_0N/index.csv        -- row_index, id, start, length, split

Usage:
    python extract_binder_residues.py --layers 7 8 9
"""

import argparse
import csv
from pathlib import Path

import numpy as np

EMBED_DIR = Path("/Volumes/Extreme Pro/esmc_residue_embeddings")
BIG_DATA_PATH = EMBED_DIR / "layers_7_8_9_joint.f32"
BIG_MANIFEST_PATH = EMBED_DIR / "manifest.csv"
OUT_DIR = EMBED_DIR / "sae_binder_only"
LAYERS_ALL = [7, 8, 9]  # matches embed_residue_layers_7_8_9.py's memmap axis order
D_MODEL = 1152

MAIN = Path(__file__).resolve().parent.parent.parent
PC7_SCORES_PATH = MAIN / "data" / "phase1_pca" / "joint" / "layer_09" / "covariance_all_dims_scores.npy"
# 0-indexed column 7 -- NOT 6. Verified directly against phase1_layer_selection/
# screen_then_confirm.py: its "PC{pc}" print uses the raw 0-indexed loop
# variable (not 1-indexed), so "PC7" throughout this project's docs/TODO.md
# means column 7. Confirmed empirically: column 7 reproduces the documented
# weighted_concordance=0.904 (joint, layer 9, covariance_all_dims) on the
# 14-binder screen set; column 6 gives an unremarkable 0.626. This was
# previously wrong here (was 6) -- caught before it affected any PC7-based
# feature selection, but note the currently-training layer 9 SAE's
# train/val/test split was stratified by the WRONG column (6) as a result.
# That doesn't invalidate the SAE itself (any reasonably-distributed
# decile-stratification still gives a representative split for training
# purposes) -- it just means the split isn't literally representative-by-
# real-PC7 as intended. Not worth regenerating mid-training; layers 7/8
# (not yet trained) will get the corrected stratification.
PC7_INDEX = 7


def load_manifest() -> list[dict]:
    with open(BIG_MANIFEST_PATH, newline="") as f:
        return [
            {k: (int(v) if k != "id" else v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def build_split(
    manifest: list[dict], val_frac: float, test_frac: float, seed: int,
    n_bins: int = 10, trainval_keep_frac: float = 1.0,
) -> list[dict]:
    """Returns one dict per design (ALL 80k, none dropped): {row_index, id,
    pc7_score, pc7_decile, split}. split in {"train", "val", "test", "unused"}.

    Two-stage, both stratified by PC7 decile: first carve test_frac of the
    FULL pool out as "test"; then split what's left (the "trainval" pool)
    by val_frac into "train"/"val". val_frac is relative to the trainval
    pool, not the full 80k -- see module docstring.

    trainval_keep_frac (default 1.0, no-op): after the train/val split,
    randomly demote (1 - trainval_keep_frac) of the "train" and "val" rows
    each to "unused" -- data.py's load_dataset only loads "train"/"val", so
    this directly controls the RAM footprint during training without
    touching test (which was never loaded into RAM in the first place, so
    shrinking it wouldn't help). Applied within each PC7 decile bin, so the
    shrunken train/val still spans the full score range. "unused" designs
    stay in activations.npy/index.csv (nothing is deleted) -- only their
    split label changes, so they're simply skipped, not lost."""
    pc7 = np.load(PC7_SCORES_PATH)[:, PC7_INDEX]
    assert pc7.shape[0] == len(manifest), "PC7 scores don't align with manifest row count"

    rng = np.random.RandomState(seed)
    row_indices = np.arange(len(manifest))
    bin_edges = np.quantile(pc7, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-6  # make the rightmost edge inclusive
    bin_ids = np.digitize(pc7, bin_edges[1:-1], right=False)  # 0..n_bins-1

    rows = []
    for b in range(n_bins):
        bin_rows = row_indices[bin_ids == b]
        rng.shuffle(bin_rows)

        n_test = round(len(bin_rows) * test_frac)
        trainval = bin_rows[n_test:]
        test = bin_rows[:n_test]

        n_val = round(len(trainval) * val_frac)
        val_rows = trainval[:n_val]
        train_rows = trainval[n_val:]

        splits_by_row = {}
        for row_idx in test:
            splits_by_row[row_idx] = "test"
        for label, group in [("val", val_rows), ("train", train_rows)]:
            n_keep = round(len(group) * trainval_keep_frac)
            for row_idx in group[:n_keep]:
                splits_by_row[row_idx] = label
            for row_idx in group[n_keep:]:
                splits_by_row[row_idx] = "unused"

        for row_idx, split in splits_by_row.items():
            rows.append({
                "row_index": int(row_idx),
                "id": manifest[row_idx]["id"],
                "pc7_score": float(pc7[row_idx]),
                "pc7_decile": b,
                "split": split,
            })

    n_by_split = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test", "unused")}
    print(f"split: {len(rows)}/{len(manifest)} designs -- counts: {n_by_split}")
    return rows


def extract_layer(layer: int, split_rows: list[dict], manifest: list[dict], total_residues_all: int) -> None:
    layer_axis = LAYERS_ALL.index(layer)
    manifest_by_row = {m["row_index"]: m for m in manifest}

    layer_dir = OUT_DIR / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    out_data_path = layer_dir / "activations.npy"
    out_index_path = layer_dir / "index.csv"

    lengths = [manifest_by_row[r["row_index"]]["binder_res_end"] - manifest_by_row[r["row_index"]]["binder_res_start"] for r in split_rows]
    total_out_residues = sum(lengths)
    print(f"layer {layer}: writing {total_out_residues:,} binder residues "
          f"({total_out_residues * D_MODEL * 2 / 1e9:.2f} GB fp16) -> {out_data_path}")

    big = np.memmap(BIG_DATA_PATH, dtype="float32", mode="r", shape=(total_residues_all, len(LAYERS_ALL), D_MODEL))
    out = np.lib.format.open_memmap(out_data_path, mode="w+", dtype="float16", shape=(total_out_residues, D_MODEL))

    index_rows = []
    cursor = 0
    for r, length in zip(split_rows, lengths):
        m = manifest_by_row[r["row_index"]]
        src_start = m["flat_start"] + m["binder_res_start"]
        out[cursor:cursor + length] = big[src_start:src_start + length, layer_axis, :].astype("float16")
        # NOTE: "id" (design_spec_N) is NOT unique across the 80k dataset --
        # it repeats identically across all 320 BoltzGen design batches (see
        # docs/TODO.md's "design_spec_N is NOT a unique ID" section). Every
        # downstream consumer (data.py, feature_analysis.py, ...) MUST key
        # off row_index, never "id" alone -- "id" is kept only for
        # human-readable display alongside it.
        index_rows.append({"row_index": r["row_index"], "id": r["id"], "start": cursor, "length": length, "split": r["split"]})
        cursor += length

    assert cursor == total_out_residues
    out.flush()
    del big, out

    with open(out_index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_index", "id", "start", "length", "split"])
        w.writeheader()
        w.writerows(index_rows)
    print(f"layer {layer}: done, wrote {out_index_path}")


def relabel_layer(layer: int, split_by_row: dict[int, str]) -> None:
    """Rewrites just the `split` column of an already-extracted layer's
    index.csv to match a new design_split.csv -- activations.npy is
    untouched (every design's residues are still at the same byte offset;
    only which split label a design carries changes). Used when
    trainval_keep_frac shrinks the training pool without re-running the
    expensive (~6 min/layer) copy out of the big external memmap."""
    layer_dir = OUT_DIR / f"layer_{layer:02d}"
    index_path = layer_dir / "index.csv"
    with open(index_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["split"] = split_by_row[int(row["row_index"])]
    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_index", "id", "start", "length", "split"])
        w.writeheader()
        w.writerows(rows)
    n_by_split = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test", "unused")}
    print(f"layer {layer}: relabeled {index_path} -- counts: {n_by_split}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS_ALL, choices=LAYERS_ALL)
    parser.add_argument("--test-frac", type=float, default=0.20, help="Fraction of the FULL 80k held out as test.")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Fraction of the remaining trainval pool (not the full 80k) used for val.")
    parser.add_argument(
        "--trainval-keep-frac", type=float, default=1.0,
        help="Fraction of the trainval pool actually kept as train/val (rest -> 'unused'). "
        "See build_split's docstring -- controls RAM footprint during training.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--relabel-only", action="store_true",
        help="Regenerate design_split.csv and patch each layer's index.csv split column "
        "WITHOUT re-copying activations.npy (fast -- seconds, not minutes). Use this to "
        "change --trainval-keep-frac (or other split params) after the initial extraction "
        "has already run once.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    total_residues_all = manifest[-1]["flat_start"] + manifest[-1]["total_length"]

    split_path = OUT_DIR / "design_split.csv"
    if split_path.exists() and not args.relabel_only:
        print(f"design_split.csv already exists, reusing it (delete or pass --relabel-only to regenerate): {split_path}")
        with open(split_path, newline="") as f:
            split_rows = [
                {"row_index": int(row["row_index"]), "id": row["id"], "pc7_score": float(row["pc7_score"]),
                 "pc7_decile": int(row["pc7_decile"]), "split": row["split"]}
                for row in csv.DictReader(f)
            ]
    else:
        split_rows = build_split(manifest, args.val_frac, args.test_frac, args.seed, trainval_keep_frac=args.trainval_keep_frac)
        with open(split_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["row_index", "id", "pc7_score", "pc7_decile", "split"])
            w.writeheader()
            w.writerows(split_rows)
        print(f"wrote {split_path}")

    if args.relabel_only:
        split_by_row = {r["row_index"]: r["split"] for r in split_rows}
        for layer in args.layers:
            relabel_layer(layer, split_by_row)
    else:
        for layer in args.layers:
            extract_layer(layer, split_rows, manifest, total_residues_all)


if __name__ == "__main__":
    main()
