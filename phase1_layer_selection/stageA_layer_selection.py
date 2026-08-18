import csv
import time
from pathlib import Path

import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "boltzgen_conf_1_batch_generation-scores.csv"
SEQ_COL = "designed_sequence - boltzgen"
ID_COL = "id - boltzgen"
JOBNAME_COL = "Job Name"
TARGET_SEQ = "DATHTQIEEDVATQLTQLASIIKINYIKPE"

OUT_DIR = ROOT / "data" / "phase1_mean_pooled"
OUT_DIR.mkdir(exist_ok=True)

N_LAYERS = 36
D_MODEL = 1152
LOG_EVERY = 200

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

# Same workaround as esmc_600M.py: ESMC.from_pretrained() is broken in esm==3.3.0.
weights_path = data_root("esmc-600") / "data" / "weights" / "esmc_600m_2024_12_v0.pth"
state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

model = ESMC(
    d_model=D_MODEL,
    n_heads=18,
    n_layers=N_LAYERS,
    tokenizer=get_esmc_model_tokenizers(),
    use_flash_attn=False,
).eval()
model.load_state_dict(state_dict, strict=True)
model = model.to(device)


def mean_pooled_layers(sequence: str) -> np.ndarray:
    """Per-layer mean-pooled embedding, shape (N_LAYERS, D_MODEL).
    Excludes the <cls>/<eos> special-token positions from the mean.
    """
    protein_tensor = model.encode(ESMProtein(sequence=sequence))
    output = model.logits(protein_tensor, LogitsConfig(return_hidden_states=True))
    hidden = output.hidden_states  # (N_LAYERS, 1, L, D_MODEL)
    residues_only = hidden[:, 0, 1:-1, :]  # drop batch dim, drop cls/eos
    return residues_only.mean(dim=1).float().cpu().numpy()  # (N_LAYERS, D_MODEL)


# Load all sequences + ids.
rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append((row[JOBNAME_COL], row[ID_COL], row[SEQ_COL]))

n = len(rows)
print(f"loaded {n} sequences")

# Save the id mapping once (row order is the join key back to the CSV).
ids_path = OUT_DIR / "ids.csv"
if not ids_path.exists():
    with open(ids_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "job_name", "id", "binder_length"])
        for i, (job_name, id_, seq) in enumerate(rows):
            w.writerow([i, job_name, id_, len(seq)])

binder_path = OUT_DIR / "binder_alone_mean_hidden.f32"
joint_path = OUT_DIR / "binder_target_joint_mean_hidden.f32"
progress_path = OUT_DIR / "progress.txt"

binder_mm = np.memmap(binder_path, dtype="float32", mode="r+" if binder_path.exists() else "w+", shape=(n, N_LAYERS, D_MODEL))
joint_mm = np.memmap(joint_path, dtype="float32", mode="r+" if joint_path.exists() else "w+", shape=(n, N_LAYERS, D_MODEL))

start_idx = int(progress_path.read_text().strip()) if progress_path.exists() else 0
if start_idx:
    print(f"resuming from index {start_idx}")

start_time = time.time()
for i in range(start_idx, n):
    _, _, binder_seq = rows[i]
    joint_seq = f"{binder_seq}|{TARGET_SEQ}"

    binder_mm[i] = mean_pooled_layers(binder_seq)
    joint_mm[i] = mean_pooled_layers(joint_seq)

    if (i + 1) % LOG_EVERY == 0 or (i + 1) == n:
        elapsed = time.time() - start_time
        done = i + 1 - start_idx
        rate = done / elapsed
        remaining = (n - (i + 1)) / rate if rate > 0 else float("inf")
        print(
            f"{i + 1}/{n}  elapsed={elapsed / 60:.1f}min  "
            f"rate={rate:.2f} seq/s  eta={remaining / 60:.1f}min",
            flush=True,
        )
        progress_path.write_text(str(i + 1))
        binder_mm.flush()
        joint_mm.flush()

print("done")
