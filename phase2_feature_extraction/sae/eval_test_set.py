"""One-off: evaluate a trained SAE checkpoint on the held-out TEST split
(the 16,000 designs never touched during training or val-based checkpoint
selection -- see extract_binder_residues.py's module docstring). train.py
never loads test_x at all (by design, to keep training's RAM footprint
bounded), so this is the first time these residues touch the model.

Usage:
    python eval_test_set.py --checkpoint checkpoints/layer_09_run1/best.pt \\
        --data-dir "/Volumes/Extreme Pro/esmc_residue_embeddings/sae_binder_only/layer_09"
"""

import argparse
from pathlib import Path

from data import center_scale, iter_batches, load_test_only
from sae_model import SparseAutoencoder
import torch


def fve(sq_err_sum: float, sq_x_sum: float) -> float:
    return 1.0 - sq_err_sum / max(sq_x_sum, 1e-12)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = ckpt["mean"].to(device)
    scale = ckpt["scale"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, layer {ckpt['layer']}, "
          f"reported val_fve={ckpt['val_fve']:.4f}")

    print("Loading test split...")
    test_x, test_row_index = load_test_only(args.data_dir)
    print(f"test_x: {test_x.shape[0]:,} residues across {len(set(test_row_index.tolist())):,} designs")

    sq_err, sq_x, l0_sum, n_rows, num_dead_seen = 0.0, 0.0, 0.0, 0, 0
    with torch.no_grad():
        for batch in iter_batches(test_x, args.batch_size, shuffle=False):
            x_proc = center_scale(batch.to(device), mean, scale)
            recon, _, num_dead, latents = model(x_proc)
            sq_err += (x_proc - recon).pow(2).sum().item()
            sq_x += x_proc.pow(2).sum().item()
            l0_sum += (latents != 0).sum().item()
            n_rows += batch.shape[0]
            num_dead_seen = num_dead  # dead-feature bookkeeping doesn't change in eval mode

    test_fve = fve(sq_err, sq_x)
    avg_l0 = l0_sum / max(n_rows, 1)
    print(f"\n=== Test set results ({n_rows:,} residues, never seen during training) ===")
    print(f"test_fve  = {test_fve:.4f}")
    print(f"avg_l0    = {avg_l0:.1f} (should equal k={model.k} exactly -- TopK is deterministic per residue)")
    print(f"num_dead (from training-time bookkeeping, not re-derived from test data) = {num_dead_seen}")
    print(f"\nFor comparison, checkpoint's own val_fve was {ckpt['val_fve']:.4f}.")


if __name__ == "__main__":
    main()
