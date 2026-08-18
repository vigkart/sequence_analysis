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
CSV_PATH = ROOT / "boltzgen_conf_1_batch_generation-scores.csv"
TARGET_SEQ = "DATHTQIEEDVATQLTQLASIIKINYIKPE"
LAYERS = [7, 8, 9]  # 0-indexed, matches every prior analysis in this project
D_MODEL = 1152
LOG_EVERY = 200

OUT_DIR = Path("/Volumes/Extreme Pro/esmc_residue_embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = OUT_DIR / "layers_7_8_9_joint.f32"
MANIFEST_PATH = OUT_DIR / "manifest.csv"
PROGRESS_PATH = OUT_DIR / "progress.txt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}", flush=True)

weights_path = data_root("esmc-600") / "data" / "weights" / "esmc_600m_2024_12_v0.pth"
state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
model = ESMC(
    d_model=D_MODEL, n_heads=18, n_layers=36,
    tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False,
).eval()
model.load_state_dict(state_dict, strict=True)
model = model.to(device)

# --- Load all sequences, precompute exact layout for the whole run ---
rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append((row["id - boltzgen"], row["designed_sequence - boltzgen"]))
n = len(rows)
print(f"loaded {n} designs", flush=True)

target_len = len(TARGET_SEQ)
manifest = []
flat_start = 0
for row_index, (design_id, binder_seq) in enumerate(rows):
    binder_len = len(binder_seq)
    total_len = binder_len + target_len + 3  # <cls> binder <chainbreak> target <eos>
    manifest.append({
        "row_index": row_index,
        "id": design_id,
        "binder_length": binder_len,
        "target_length": target_len,
        "total_length": total_len,
        "flat_start": flat_start,
        "binder_res_start": 1,
        "binder_res_end": 1 + binder_len,
        "target_res_start": binder_len + 2,
        "target_res_end": binder_len + 2 + target_len,
    })
    flat_start += total_len
total_residues = flat_start
print(f"total residues: {total_residues:,}  ->  {total_residues * len(LAYERS) * D_MODEL * 4 / 1e9:.1f} GB (fp32)", flush=True)

if not MANIFEST_PATH.exists():
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"wrote manifest: {MANIFEST_PATH}", flush=True)
else:
    print(f"manifest already exists, not overwriting: {MANIFEST_PATH}", flush=True)

mode = "r+" if DATA_PATH.exists() else "w+"
data = np.memmap(DATA_PATH, dtype="float32", mode=mode, shape=(total_residues, len(LAYERS), D_MODEL))
print(f"opened memmap ({mode}): {DATA_PATH}", flush=True)

start_idx = int(PROGRESS_PATH.read_text().strip()) if PROGRESS_PATH.exists() else 0
if start_idx:
    print(f"resuming from row {start_idx}", flush=True)

start_time = time.time()
for i in range(start_idx, n):
    design_id, binder_seq = rows[i]
    m = manifest[i]
    joint_seq = f"{binder_seq}|{TARGET_SEQ}"

    protein_tensor = model.encode(ESMProtein(sequence=joint_seq))
    output = model.logits(protein_tensor, LogitsConfig(return_hidden_states=True))
    hidden = output.hidden_states  # (36, 1, L, D)
    selected = hidden[LAYERS, 0, :, :]  # (3, L, D)
    selected = selected.permute(1, 0, 2).float().cpu().numpy()  # (L, 3, D)

    assert selected.shape[0] == m["total_length"], (
        f"row {i}: expected L={m['total_length']}, got {selected.shape[0]}"
    )
    data[m["flat_start"]: m["flat_start"] + m["total_length"]] = selected

    if (i + 1) % LOG_EVERY == 0 or (i + 1) == n:
        elapsed = time.time() - start_time
        done = i + 1 - start_idx
        rate = done / elapsed
        remaining = (n - (i + 1)) / rate if rate > 0 else float("inf")
        print(
            f"{i + 1}/{n}  elapsed={elapsed / 60:.1f}min  rate={rate:.2f} seq/s  eta={remaining / 60:.1f}min",
            flush=True,
        )
        PROGRESS_PATH.write_text(str(i + 1))
        data.flush()

PROGRESS_PATH.write_text(str(n))
data.flush()
print("done", flush=True)
