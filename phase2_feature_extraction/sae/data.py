"""Load extracted binder-only per-residue activations (see
extract_binder_residues.py) and build the train/val tensors for SAE
training. Test is loaded separately (load_test_only, below) and never by
train.py -- see that function's docstring for why.

Expects a directory (e.g.
/Volumes/Extreme Pro/esmc_residue_embeddings/sae_binder_only/layer_09/, as
written by extract_binder_residues.py) containing:
    activations.npy: (n_residues, d_model) float16, ALL 80k designs' binder
        residues (target residues already excluded), one row per residue.
    index.csv: row_index, id, start, length, split -- activations[start:
        start+length] is that design's residues, in sequence order.
        split is one of "train"/"val"/"test": test is 20% of the full 80k;
        the remaining 80% ("trainval", 64k) is split again 90/10 into
        train/val. Assigned once at the design level in
        extract_binder_residues.py's build_split() (stratified by PC7
        decile) and shared identically across all 3 layers' index.csv
        files, so the same design lands in the same split for every layer.

Unlike igemmodel's version of this file (multiple training sources,
natural-binder eval-only carve-outs), this project has one homogeneous
corpus (BoltzGen designs against one TET1 site) and the split is already
resolved by the extraction step -- this module just loads it, no
re-derivation of train/val logic here.

IMPORTANT: "id" (design_spec_N) is NOT a unique identifier across the 80k
dataset -- it repeats across all 320 BoltzGen design batches (see
docs/TODO.md's "design_spec_N is NOT a unique ID" section, and
extract_binder_residues.py's index.csv comment). Use row_index for anything
that needs to resolve back to a specific design.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np  # noqa: F401 -- must import before torch, see train.py's import-order note
import pandas as pd
import torch


@dataclass
class SAEDataset:
    train_x: torch.Tensor  # (n_train, d_model) float16, RAW (uncentered, unscaled)
    val_x: torch.Tensor  # (n_val, d_model) float16, RAW
    train_row_index: np.ndarray  # (n_train,) int, the design each residue belongs to
    val_row_index: np.ndarray
    mean: torch.Tensor  # (d_model,) float32, computed from train_x only
    scale: float  # computed from train_x only


def _load_rows(acts: np.memmap, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """rows: dataframe with start/length/row_index columns, one row per design.
    Returns (activations, residue_row_index).

    Deliberately NOT `acts[big_fancy_index_array].copy()` -- that was the
    original approach and it was drastically slower in practice (confirmed:
    a single-layer load stalled at ~9GB/18GB after 39 minutes on the
    external drive). NumPy fancy indexing on a memmap doesn't recognize
    that the index array is really ~57k contiguous per-design runs
    concatenated together -- it gathers as if every index were an
    independent random access, which degrades into brutal small-block I/O
    on external-drive latency. Copying each design's own contiguous
    [start:start+length] slice directly (same pattern
    extract_binder_residues.py already uses successfully) makes every read
    a real contiguous block instead. Rows are also sorted by `start` first
    so the sequence of reads is monotonically increasing through the file,
    rather than jumping around in the decile-shuffled order index.csv rows
    happen to be in."""
    rows = rows.sort_values("start")
    total = int(rows["length"].sum())
    out = np.empty((total, acts.shape[1]), dtype=acts.dtype)
    row_index = np.empty(total, dtype=np.int64)

    cursor = 0
    for start, length, r_idx in zip(rows["start"], rows["length"], rows["row_index"]):
        out[cursor:cursor + length] = acts[start:start + length]
        row_index[cursor:cursor + length] = r_idx
        cursor += length
    assert cursor == total
    return out, row_index


def load_dataset(data_dir: Path) -> SAEDataset:
    """Loads ONLY train+val into RAM (~18.4GB/layer for the current 64k
    trainval pool) -- test is deliberately excluded here. test is a full 20%
    of the real 80k (not a small slice of a shrunken pool, see
    extract_binder_residues.py's module docstring for why), and loading it
    here too would put us right back at the ~23GB-per-layer RAM pressure
    this split was designed to avoid during training. Use load_test_only()
    (below) to load it separately, only when actually needed (e.g.
    feature_analysis.py's final evaluation)."""
    data_dir = Path(data_dir)
    index_df = pd.read_csv(data_dir / "index.csv")
    trainval_df = index_df[index_df["split"].isin(["train", "val"])]

    for name, rows in index_df.groupby("split"):
        print(f"{name}: {len(rows)} designs, {rows['length'].sum()} residues")

    # activations.npy is written via np.lib.format.open_memmap by the
    # extraction script, covering ALL 80k designs (train+val+test) --
    # memmap it and only materialize the train/val rows into RAM.
    acts = np.load(data_dir / "activations.npy", mmap_mode="r")

    splits = {}
    for split_name in ("train", "val"):
        rows = trainval_df[trainval_df["split"] == split_name]
        x, row_index = _load_rows(acts, rows)
        splits[split_name] = (torch.from_numpy(x), row_index)
    del acts

    train_x, train_row_index = splits["train"]
    val_x, val_row_index = splits["val"]

    mean = train_x.float().mean(dim=0)
    centered_norms = (train_x.float() - mean).norm(dim=-1)
    scale = float(train_x.shape[1] ** 0.5 / centered_norms.mean().item())
    print(f"train mean norm={mean.norm().item():.4f}, scale={scale:.6f}")

    return SAEDataset(
        train_x=train_x,
        val_x=val_x,
        train_row_index=train_row_index,
        val_row_index=val_row_index,
        mean=mean,
        scale=scale,
    )


def load_test_only(data_dir: Path) -> tuple[torch.Tensor, np.ndarray]:
    """Loads just the held-out test split into RAM -- for final evaluation
    (feature_analysis.py), never called from train.py. Returns (test_x,
    test_row_index)."""
    data_dir = Path(data_dir)
    index_df = pd.read_csv(data_dir / "index.csv")
    rows = index_df[index_df["split"] == "test"]
    acts = np.load(data_dir / "activations.npy", mmap_mode="r")
    x, row_index = _load_rows(acts, rows)
    del acts
    return torch.from_numpy(x), row_index


def iter_batches(x: torch.Tensor, batch_size: int, shuffle: bool = True, generator=None):
    """shuffle=True (training): random order, drops a ragged final batch.
    shuffle=False (val/test): original row order preserved (so a parallel
    row_index array stays aligned), includes a final partial batch so every
    row is covered exactly once."""
    n = x.shape[0]
    if shuffle:
        perm = torch.randperm(n, generator=generator)
        n_full = (n // batch_size) * batch_size
        for i in range(0, n_full, batch_size):
            yield x[perm[i : i + batch_size]]
    else:
        for i in range(0, n, batch_size):
            yield x[i : i + batch_size]


def center_scale(x: torch.Tensor, mean: torch.Tensor, scale: float) -> torch.Tensor:
    return (x.float() - mean) * scale


def uncenter_unscale(x_proc: torch.Tensor, mean: torch.Tensor, scale: float) -> torch.Tensor:
    return x_proc / scale + mean
