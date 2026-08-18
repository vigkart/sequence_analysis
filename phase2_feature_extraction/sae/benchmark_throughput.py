"""One-off: measure real SAE forward+backward throughput on this machine's
MPS backend, using actual binder-residue activations (not synthetic data),
to ground the SAE training pool-size decision in a real number instead of a
guess. See docs/TODO.md's "Future direction 1" and the sae/ README for why
this matters -- 10-fold CV or a full 80k-design training pool are only
worth committing to if the wall-clock time is actually tractable locally.

Reads a small real slice directly from the big external memmap (no need to
run extract_binder_residues.py first) -- just enough designs to fill several
batches, looped repeatedly to measure steady-state per-step time.

Usage:
    python benchmark_throughput.py --layer 9 --d-hidden 16384 --k 64 --steps 100
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from sae_model import SparseAutoencoder, loss_fn

EMBED_DIR = Path("/Volumes/Extreme Pro/esmc_residue_embeddings")
DATA_PATH = EMBED_DIR / "layers_7_8_9_joint.f32"
MANIFEST_PATH = EMBED_DIR / "manifest.csv"
LAYERS = [7, 8, 9]  # matches embed_residue_layers_7_8_9.py's memmap axis order
D_MODEL = 1152

# Whole-dataset totals, computed once from manifest.csv (see extraction
# script for the full per-design version) -- used to extrapolate this
# benchmark's measured throughput to real full-run wall-clock estimates.
N_DESIGNS_TOTAL = 80_000


def load_manifest_sample(n_designs: int) -> list[dict]:
    rows = []
    with open(MANIFEST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n_designs:
                break
            rows.append({k: int(v) if k != "id" else v for k, v in row.items()})
    return rows


def load_binder_activations(layer: int, n_designs: int) -> np.ndarray:
    """Reads real binder-residue activations for the first n_designs designs
    at the given layer, straight off the external memmap. Returns
    (n_residues, D_MODEL) float32."""
    layer_axis = LAYERS.index(layer)
    rows = load_manifest_sample(n_designs)
    last_end = rows[-1]["flat_start"] + rows[-1]["total_length"]
    # np.memmap needs the true total residue count for its shape; read it off
    # the manifest's last row rather than the (possibly truncated) sample.
    with open(MANIFEST_PATH, newline="") as f:
        last_row = None
        for last_row in csv.DictReader(f):
            pass
    total_residues = int(last_row["flat_start"]) + int(last_row["total_length"])
    mm = np.memmap(DATA_PATH, dtype="float32", mode="r", shape=(total_residues, len(LAYERS), D_MODEL))

    chunks = []
    for r in rows:
        start = r["flat_start"] + r["binder_res_start"]
        end = r["flat_start"] + r["binder_res_end"]
        chunks.append(np.array(mm[start:end, layer_axis, :]))  # copy out of the memmap
    del mm
    acts = np.concatenate(chunks, axis=0)
    print(
        f"loaded {len(rows)} designs, {acts.shape[0]:,} binder residues at layer {layer} "
        f"(read up to flat offset {last_end:,}/{total_residues:,})"
    )
    return acts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=9, choices=LAYERS)
    parser.add_argument("--d-hidden", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--auxk", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--sample-designs", type=int, default=500, help="Real designs read from the memmap to build benchmark batches from.")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--full-pool-residues", type=int, default=10_200_000,
        help="Estimated total binder residues across all 80k designs, single layer "
        "(~80.7%% of the 12,638,290 total residues per the manifest's binder_length "
        "vs total_length ratio) -- used only to extrapolate wall-clock estimates.",
    )
    parser.add_argument("--epochs-for-estimate", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != "mps":
        print("WARNING: MPS not available, falling back to CPU -- estimate will not reflect real training speed")

    acts = load_binder_activations(args.layer, args.sample_designs)
    if acts.shape[0] < args.batch_size * (args.warmup_steps + args.steps):
        # Reuse the same real residues repeatedly (shuffled) rather than reading
        # more designs -- fine for a throughput benchmark, since compute cost
        # per step depends on shape, not content.
        reps = int(np.ceil(args.batch_size * (args.warmup_steps + args.steps) / acts.shape[0])) + 1
        acts = np.tile(acts, (reps, 1))
    x_all = torch.from_numpy(acts).float()
    n = x_all.shape[0]

    mean = x_all.mean(dim=0).to(device)
    scale = float(x_all.shape[1] ** 0.5 / (x_all - x_all.mean(dim=0)).norm(dim=-1).mean().item())

    model = SparseAutoencoder(
        d_model=D_MODEL, d_hidden=args.d_hidden, k=args.k, auxk=args.auxk,
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=generator)

    def get_batch(step: int) -> torch.Tensor:
        idx = perm[(step * args.batch_size) % n : (step * args.batch_size) % n + args.batch_size]
        if idx.shape[0] < args.batch_size:
            idx = perm[:args.batch_size]
        batch = x_all[idx].to(device)
        return (batch - mean) * scale

    print(f"warming up ({args.warmup_steps} steps)...")
    for step in range(args.warmup_steps):
        x_proc = get_batch(step)
        recon, auxk_recon, _, _ = model(x_proc)
        mse_loss, auxk_loss = loss_fn(x_proc, recon, auxk_recon)
        loss = mse_loss + auxk_loss
        optimizer.zero_grad()
        loss.backward()
        model.norm_grad()
        optimizer.step()
        model.norm_weights()
    if device.type == "mps":
        torch.mps.synchronize()

    print(f"timing {args.steps} steps...")
    start = time.time()
    for step in range(args.warmup_steps, args.warmup_steps + args.steps):
        x_proc = get_batch(step)
        recon, auxk_recon, num_dead, _ = model(x_proc)
        mse_loss, auxk_loss = loss_fn(x_proc, recon, auxk_recon)
        loss = mse_loss + auxk_loss
        optimizer.zero_grad()
        loss.backward()
        model.norm_grad()
        optimizer.step()
        model.norm_weights()
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - start

    residues_per_step = args.batch_size
    total_residues_timed = residues_per_step * args.steps
    residues_per_sec = total_residues_timed / elapsed
    sec_per_step = elapsed / args.steps

    print(f"\n{args.steps} steps in {elapsed:.1f}s -> {sec_per_step * 1000:.1f}ms/step, "
          f"{residues_per_sec:,.0f} residues/sec (d_hidden={args.d_hidden}, k={args.k}, "
          f"batch_size={args.batch_size}, device={device})")

    for label, pool_residues in [
        ("this benchmark's pool size", n if n <= args.full_pool_residues else None),
        ("full 80k-design pool (est.)", args.full_pool_residues),
        ("30k-design subsample (~37.5%)", int(args.full_pool_residues * 0.375)),
        ("20k-design subsample (~25%)", int(args.full_pool_residues * 0.25)),
    ]:
        if pool_residues is None:
            continue
        steps_per_epoch = pool_residues // args.batch_size
        sec_per_epoch = steps_per_epoch * sec_per_step
        total_sec = sec_per_epoch * args.epochs_for_estimate
        print(
            f"  {label}: {pool_residues:,} residues -> {steps_per_epoch:,} steps/epoch, "
            f"{sec_per_epoch / 60:.1f} min/epoch, "
            f"{total_sec / 3600:.1f}h for {args.epochs_for_estimate} epochs"
        )


if __name__ == "__main__":
    main()
