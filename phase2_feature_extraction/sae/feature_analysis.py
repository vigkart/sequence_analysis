"""Feature (semantic) analysis for a trained SAE: what does each of its
d_hidden features respond to? Standard SAE-interpretability recipe
(Cunningham et al. 2023; Bricken et al. 2023, "Towards Monosemanticity"):
for every feature, find the residues that make it fire hardest ("max-
activating examples") and track how often it fires at all ("density").
Reading the local sequence context around a feature's hardest-firing
residues is how a human (or an LLM, downstream in label_features.py)
assigns it a semantic label.

Adapted from igemmodel/sae/training/feature_analysis.py, with the linear
probe against downstream metrics REMOVED per this project's plan (see
docs/TODO.md's "Future direction 1") -- feature selection for InterProScan
here instead uses (a) which features fire on the real, wet-lab-tested
binders specifically, and (b) density extremes (highest and lowest
nonzero), matching the same "3 highest-density + 3 rare-but-strong" recipe
igEM's own team used in practice (see igemmodel/sae/RESULTS.md). See
select_interpro_candidates.py for that step.

Other differences from the igemmodel version:
  - Streams the FULL 80k-design pool, not a bounded subsample
    (--max-residues). Encode-only forward passes (no backward, no AuxK)
    are fast enough that the full pool is tractable in a few minutes --
    matches docs/TODO.md's stated goal of examples "across the 80k binder
    set", not a sample of it.
  - Every join keys on row_index, never "id" -- "id" (design_spec_N)
    repeats across all 320 BoltzGen batches (see extract_binder_residues.py's
    comment). Local sequence context is pulled from
    data/boltzgen_conf_1_batch_generation-scores.csv via row_index, since
    the big external manifest.csv doesn't carry sequence text.
  - No "source" column (single homogeneous corpus) -- split
    (train/val/test/unused) is carried through instead, purely for
    provenance/display in the output, not for any sampling logic.
  - Per-residue metadata during the streaming loop is (row_index, position)
    as plain int arrays, NOT a context string built for all ~10M residues
    up front -- context strings are only ever needed for the handful of
    residues that actually survive into a feature's top-N heap
    (<= d_hidden * top_n = 245,760 in the worst case, usually far fewer
    after dedup), so building all 10M up front (the first version of this
    script did that) was ~40x more string-building work than necessary and
    made the per-residue metadata pass the dominant cost. Context strings
    are computed in one pass over just the survivors at the end.
  - All prints are flushed immediately (flush=True) -- this project hit
    output-buffering confusion twice already tonight (the extraction
    script and the first version of this one both looked "stuck" under a
    redirect when they weren't), so this isn't optional here.

Usage:
    python feature_analysis.py --checkpoint checkpoints/layer_09_run1/best.pt \\
        --data-dir "/Volumes/Extreme Pro/esmc_residue_embeddings/sae_binder_only/layer_09" \\
        --output-dir feature_analysis_results/layer_09
"""

import argparse
import csv
import heapq
from pathlib import Path

import numpy as np  # noqa: F401 -- import-order note, see data.py
import pandas as pd
from data import center_scale
from sae_model import SparseAutoencoder
import torch

MAIN = Path(__file__).resolve().parent.parent.parent
SCORES_CSV = MAIN / "data" / "boltzgen_conf_1_batch_generation-scores.csv"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sae(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["mean"].to(device), ckpt["scale"]


def load_binder_sequences() -> dict[int, str]:
    """row_index -> designed binder sequence (from the scores CSV, the only
    place full sequence text lives -- the big external manifest.csv doesn't
    carry it)."""
    seqs = {}
    with open(SCORES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            seqs[row_index] = row["designed_sequence - boltzgen"]
    return seqs


def make_context(sequence: str, position: int, window: int) -> str:
    lo, hi = max(0, position - window), min(len(sequence), position + window + 1)
    return f"{sequence[lo:position]}[{sequence[position]}]{sequence[position + 1:hi]}"


class FeatureStats:
    """Streaming per-feature density + mean-activation-when-active. Never
    materializes the full (n_residues, d_hidden) code matrix."""

    def __init__(self, d_hidden: int):
        self.d_hidden = d_hidden
        self.fire_count = torch.zeros(d_hidden, dtype=torch.long)
        self.activation_sum = torch.zeros(d_hidden, dtype=torch.float64)
        self.n_seen = 0

    def update(self, codes: torch.Tensor) -> None:
        # NOTE: must .cpu() BEFORE .double() -- MPS has no float64 support,
        # unlike CUDA (where igemmodel's original version of this ran).
        fired = codes != 0
        self.fire_count += fired.sum(dim=0).cpu()
        self.activation_sum += codes.sum(dim=0).cpu().double()
        self.n_seen += codes.shape[0]

    def to_dataframe(self) -> pd.DataFrame:
        fire = self.fire_count.numpy()
        density = fire / max(self.n_seen, 1)
        mean_active = np.divide(
            self.activation_sum.numpy(), fire, out=np.zeros(len(fire)), where=fire > 0
        )
        return pd.DataFrame({
            "feature": np.arange(self.d_hidden),
            "density": density,
            "fire_count": fire,
            "mean_activation_when_active": mean_active,
            "dead": fire == 0,
        })


class TopActivations:
    """Streaming top-N max-activating-example tracker: one small min-heap
    per feature, so memory never scales with pool size. Heap entries are
    (activation, tie_breaker, row_index, position) -- plain ints, no
    context string built until to_dataframe() is called on the (small)
    survivor set."""

    def __init__(self, d_hidden: int, top_n: int = 15):
        self.d_hidden = d_hidden
        self.top_n = top_n
        self._heaps: list[list[tuple[float, int, int, int]]] = [[] for _ in range(d_hidden)]
        self._tie = 0

    def _offer(self, feature: int, activation: float, row_index: int, position: int) -> None:
        heap = self._heaps[feature]
        self._tie += 1
        entry = (activation, self._tie, row_index, position)
        if len(heap) < self.top_n:
            heapq.heappush(heap, entry)
        elif activation > heap[0][0]:
            heapq.heapreplace(heap, entry)

    def update_batch(self, codes: torch.Tensor, row_index_batch: np.ndarray, position_batch: np.ndarray) -> None:
        nz = torch.nonzero(codes, as_tuple=False)
        if nz.numel() == 0:
            return
        values = codes[nz[:, 0], nz[:, 1]].tolist()
        nz_list = nz.tolist()
        for (row, feat), val in zip(nz_list, values):
            self._offer(feat, val, int(row_index_batch[row]), int(position_batch[row]))

    def to_dataframe(self, sequences: dict[int, str], split_by_row: dict[int, str], context: int) -> pd.DataFrame:
        rows = []
        for feat in range(self.d_hidden):
            ranked = sorted(self._heaps[feat], key=lambda e: -e[0])
            for rank, (activation, _, row_index, position) in enumerate(ranked):
                rows.append({
                    "feature": feat,
                    "rank": rank,
                    "activation": activation,
                    "row_index": row_index,
                    "split": split_by_row[row_index],
                    "position": position,
                    "context": make_context(sequences[row_index], position, context),
                })
        return pd.DataFrame(rows)


def _build_residue_index_arrays(index_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized construction of (row_index_per_residue, position_per_residue)
    -- both length total_residues, in the SAME order as activations.npy.
    Pure numpy (np.repeat/np.arange), not a Python-level loop building
    10M objects -- this used to be the dominant cost in this script."""
    row_indices = index_df["row_index"].to_numpy()
    lengths = index_df["length"].to_numpy()
    row_index_per_residue = np.repeat(row_indices, lengths)
    position_per_residue = np.concatenate([np.arange(n) for n in lengths])
    return row_index_per_residue, position_per_residue


def run_full_pool_mining(
    model, mean, scale, acts: np.memmap, index_df: pd.DataFrame, sequences: dict[int, str],
    top_n: int, context: int, batch_size: int, device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Streams every residue in the pool (all 80k designs, all splits)
    through the frozen encoder, batched in large CONTIGUOUS chunks that
    cross design boundaries -- not one tiny forward pass per design (80k of
    those would be badly bottlenecked by Python/dispatch overhead at this
    scale, the same class of problem data.py's loader hit and was fixed
    for). activations.npy and index_df are in the same on-disk residue
    order, so a chunk of consecutive rows can be read as one plain slice."""
    stats = FeatureStats(model.d_hidden)
    top = TopActivations(model.d_hidden, top_n=top_n)

    print("  building per-residue index arrays (row_index/position, vectorized)...", flush=True)
    row_index_per_residue, position_per_residue = _build_residue_index_arrays(index_df)
    total_residues = len(row_index_per_residue)
    assert total_residues == acts.shape[0], (total_residues, acts.shape[0])
    split_by_row = dict(zip(index_df["row_index"], index_df["split"]))

    print(f"  starting encode pass over {total_residues:,} residues...", flush=True)
    for start in range(0, total_residues, batch_size):
        end = min(start + batch_size, total_residues)
        x = torch.from_numpy(acts[start:end].astype(np.float32)).to(device)
        x_proc = center_scale(x, mean, scale)
        with torch.no_grad():
            codes = model.encode(x_proc)

        stats.update(codes)
        top.update_batch(codes.cpu(), row_index_per_residue[start:end], position_per_residue[start:end])

        if (start // batch_size) % 20 == 0 or end == total_residues:
            print(f"  {end:,}/{total_residues:,} residues encoded ({100*end/total_residues:.1f}%)", flush=True)

    print("  building context strings for surviving top examples only...", flush=True)
    top_df = top.to_dataframe(sequences, split_by_row, context)
    return stats.to_dataframe(), top_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--context", type=int, default=6, help="Residues shown on each side of the activating residue.")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}", flush=True)

    model, mean, scale = load_sae(args.checkpoint, device)
    print(f"Loaded SAE: d_model={model.d_model} d_hidden={model.d_hidden} k={model.k}", flush=True)

    index_df = pd.read_csv(args.data_dir / "index.csv")
    print(f"Pool: {len(index_df):,} designs, {index_df['length'].sum():,} residues "
          f"(all splits: {index_df.groupby('split').size().to_dict()})", flush=True)

    print("Loading binder sequences from scores CSV...", flush=True)
    sequences = load_binder_sequences()

    acts = np.load(args.data_dir / "activations.npy", mmap_mode="r")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== Mining: density + max-activating examples across the FULL pool ===", flush=True)
    stats_df, top_df = run_full_pool_mining(
        model, mean, scale, acts, index_df, sequences, args.top_n, args.context, args.batch_size, device,
    )
    stats_df.to_csv(args.output_dir / "feature_stats.csv", index=False)
    top_df.to_csv(args.output_dir / "feature_top_examples.csv", index=False)
    n_dead = int(stats_df["dead"].sum())
    print(f"\nWrote feature_stats.csv ({n_dead}/{model.d_hidden} dead) and feature_top_examples.csv to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
