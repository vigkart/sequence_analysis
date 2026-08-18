"""Train a sparse autoencoder on ESMC-600M layer 7/8/9 per-residue
activations (binder residues only, joint-condition embeddings) for the
TET1 binder-design pool. Run once per layer -- see docs/TODO.md's "Future
direction 1" for why layers 7-9 specifically, and this directory's README
for the full pipeline.

Ported from igemmodel/sae/training/train.py (architecture unchanged, see
sae_model.py's docstring), adapted for this project's scale/setup:
  - MPS device support added (igemmodel's version only ever ran on a CUDA
    cluster).
  - Loads only the train+val split (data.py's load_dataset skips test
    entirely -- see extract_binder_residues.py and data.py for why: test is
    a real 20% of the full 80k and would blow the RAM budget this split was
    designed around if loaded alongside train/val during a multi-hour
    unattended run). val is used here exactly like igemmodel used it
    (checkpoint selection via val_fve during training); test is never
    touched by this script.
  - No k-fold cross-validation on the autoencoder itself: with ~6M+
    residues in the training pool, a single held-out split already gives a
    stable FVE estimate (law-of-large-numbers), and 10-fold would multiply
    training time ~10x for negligible statistical benefit on this metric.
    k-fold/nested CV is reserved for feature_analysis.py's downstream
    linear probe against the PC7 binding-relevant score, where it's a real
    supervised claim and cheap to run properly.
  - LR schedule changed from igemmodel's original (linear warmup, then HELD
    CONSTANT at peak for the rest of training, forever) to linear warmup
    followed by cosine decay down to --min-lr-ratio of peak (see
    lr_lambda's docstring below) -- general practice favors decaying LR
    toward the end of training for tighter final convergence over holding
    it constant throughout.
  - Optimizer deliberately left as plain Adam (not AdamW), weight_decay=0,
    same as igemmodel: at weight_decay=0 the two are mathematically
    identical (AdamW only differs in how it applies decay, and there's no
    decay to apply), and this SAE's weight magnitude is already controlled
    by more targeted mechanisms than decay would provide -- sae_model.py's
    decoder unit-norm constraint (hard, not just a soft pull-to-zero) and
    the reconstruction loss's own implicit calibration pressure on the
    encoder. Adding decay on top would also add uniform downward pressure
    on encoder columns, working against the AuxK/k-annealing machinery
    that's already fighting dead-feature death.

Usage:
    python train.py --data-dir "/Volumes/Extreme Pro/esmc_residue_embeddings/sae_binder_only/layer_09" \\
        --output-dir checkpoints/layer_09_run1 --layer 9
"""

import argparse
import csv
import math
import time
from pathlib import Path

# NOTE: import order matters on this machine -- importing torch before numpy
# segfaults in numpy's macOS Accelerate-framework self-check (confirmed in
# igemmodel's train.py via `python -c "import torch; import numpy"` vs. the
# reverse order). data.py imports numpy/pandas before torch, so importing it
# first here guarantees numpy loads first process-wide.
from data import center_scale, iter_batches, load_dataset
from sae_model import SparseAutoencoder, loss_fn
import torch


def fve(sq_err_sum: float, sq_x_sum: float) -> float:
    """Fraction of variance explained: 1 - MSE(recon, x) / MSE(0, x). The
    "0" baseline is the do-nothing predictor in centered+scaled space, i.e.
    reconstructing the train-set mean in the original space -- standard
    convention for SAE reconstruction quality, and scale-invariant so no
    un-centering/un-scaling is needed to interpret it."""
    return 1.0 - sq_err_sum / max(sq_x_sum, 1e-12)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def validate(model, x, batch_size, mean, scale, device):
    model.eval()
    sq_err, sq_x, l0_sum, n_rows = 0.0, 0.0, 0.0, 0
    for batch in iter_batches(x, batch_size, shuffle=False):
        x_proc = center_scale(batch.to(device), mean, scale)
        recon, _, _, latents = model(x_proc)
        sq_err += (x_proc - recon).pow(2).sum().item()
        sq_x += x_proc.pow(2).sum().item()
        l0_sum += (latents != 0).sum().item()
        n_rows += batch.shape[0]
    num_dead = int(model.dead_feature_mask().sum().item())
    model.train()
    return {"fve": fve(sq_err, sq_x), "avg_l0": l0_sum / max(n_rows, 1)}, num_dead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True, choices=[7, 8, 9])
    parser.add_argument("--d-hidden", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument(
        "--k-start", type=int, default=None,
        help="Initial k for the annealing schedule (defaults to 4x --k). "
        "Linearly decayed down to --k over --k-anneal-frac of total steps, "
        "then held at --k. See sae_model.py's K-ANNEALING docstring.",
    )
    parser.add_argument("--k-anneal-frac", type=float, default=0.2)
    parser.add_argument("--auxk", type=int, default=256)
    parser.add_argument("--dead-tokens-threshold", type=int, default=400_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument(
        "--min-lr-ratio", type=float, default=0.1,
        help="Cosine decay floor as a fraction of peak lr (post-warmup). "
        "igemmodel's original schedule held constant at 1.0 forever after "
        "warmup instead of decaying -- see lr_lambda's docstring.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU device found, falling back to CPU (will be slow)")

    ds = load_dataset(args.data_dir)
    d_model = ds.train_x.shape[1]
    mean = ds.mean.to(device)
    scale = ds.scale

    model = SparseAutoencoder(
        d_model=d_model, d_hidden=args.d_hidden, k=args.k, auxk=args.auxk,
        dead_tokens_threshold=args.dead_tokens_threshold,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = ds.train_x.shape[0] // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step):
        """Linear warmup, then cosine decay down to --min-lr-ratio of peak
        over the remaining steps (not a hold-at-peak constant, which is
        what igemmodel's original schedule did -- see train.py's module
        docstring). Decaying to a small nonzero floor rather than exactly 0
        is deliberate: an exact-zero final LR means the last few steps do
        nothing, and it's brittle to any off-by-one in total_steps."""
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    k_start = args.k_start if args.k_start is not None else 4 * args.k
    k_anneal_steps = max(1, int(args.k_anneal_frac * total_steps))

    def current_k(step: int) -> int:
        if args.k_anneal_frac <= 0 or k_start <= args.k:
            return args.k
        frac = min(1.0, step / k_anneal_steps)
        return round(k_start + frac * (args.k - k_start))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training_log.csv"
    log_fields = ["epoch", "train_mse", "train_auxk_loss", "val_fve", "val_avg_l0", "num_dead"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(log_fields)

    print(
        f"{steps_per_epoch} steps/epoch x {args.epochs} epochs = {total_steps} total steps "
        f"(warmup: {warmup_steps} steps)"
    )

    generator = torch.Generator().manual_seed(args.seed)
    best_val_fve = float("-inf")
    step = 0
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        epoch_mse, epoch_auxk, n_batches = 0.0, 0.0, 0
        for batch in iter_batches(ds.train_x, args.batch_size, shuffle=True, generator=generator):
            x_proc = center_scale(batch.to(device), mean, scale)
            step_k = current_k(step)
            recon, auxk_recon, num_dead, _ = model(x_proc, k=step_k)
            mse_loss, auxk_loss = loss_fn(x_proc, recon, auxk_recon)
            loss = mse_loss + auxk_loss

            optimizer.zero_grad()
            loss.backward()
            model.norm_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            model.norm_weights()
            scheduler.step()

            epoch_mse += mse_loss.item()
            epoch_auxk += auxk_loss.item()
            n_batches += 1
            step += 1
            if step % args.log_every == 0:
                print(
                    f"  step {step}/{total_steps}: mse={mse_loss.item():.4f} "
                    f"auxk={auxk_loss.item():.4f} num_dead={num_dead} k={step_k} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        train_mse = epoch_mse / max(n_batches, 1)
        train_auxk = epoch_auxk / max(n_batches, 1)
        val_metrics, num_dead = validate(model, ds.val_x, args.batch_size, mean, scale, device)
        elapsed = time.time() - start

        print(
            f"epoch {epoch}: train_mse={train_mse:.4f} train_auxk={train_auxk:.4f} "
            f"val_fve={val_metrics['fve']:.4f} val_l0={val_metrics['avg_l0']:.1f} num_dead={num_dead} "
            f"end_k={current_k(step - 1)} ({elapsed:.1f}s)"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_mse, train_auxk, val_metrics["fve"], val_metrics["avg_l0"], num_dead]
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            # NOTE: keep this dict limited to SparseAutoencoder.__init__'s actual
            # params -- feature_analysis.py does SparseAutoencoder(**ckpt["config"]).
            "config": {
                "d_model": d_model,
                "d_hidden": args.d_hidden,
                "k": args.k,
                "auxk": args.auxk,
                "dead_tokens_threshold": args.dead_tokens_threshold,
            },
            "layer": args.layer,
            "k_start_resolved": k_start,
            "mean": ds.mean,
            "scale": scale,
            "args": vars(args),
            "epoch": epoch,
            "val_fve": val_metrics["fve"],
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if val_metrics["fve"] > best_val_fve:
            best_val_fve = val_metrics["fve"]
            torch.save(checkpoint, args.output_dir / "best.pt")
            print(f"  new best val_fve={best_val_fve:.4f}, saved best.pt")

    print(f"Done. Best val_fve={best_val_fve:.4f}. Checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
