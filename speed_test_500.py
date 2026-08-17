import csv
import time

import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root

CSV_PATH = "boltzgen_conf_1_batch_generation-scores.csv"
SEQ_COL = "designed_sequence - boltzgen"
N = 500
FULL_RUN_SIZE = 80_000

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

# Same workaround as esmc_600M.py: ESMC.from_pretrained() is broken in esm==3.3.0
# (points load_torch_model at the wrong checkpoint path), so load directly.
weights_path = data_root("esmc-600") / "data" / "weights" / "esmc_600m_2024_12_v0.pth"
state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

model = ESMC(
    d_model=1152,
    n_heads=18,
    n_layers=36,
    tokenizer=get_esmc_model_tokenizers(),
    use_flash_attn=False,
).eval()
model.load_state_dict(state_dict, strict=True)
model = model.to(device)

sequences = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= N:
            break
        sequences.append(row[SEQ_COL])

print(f"loaded {len(sequences)} sequences")
lengths = [len(s) for s in sequences]
print(f"sequence length range: {min(lengths)}-{max(lengths)} (mean {sum(lengths)/len(lengths):.1f})")

# Warm-up: first MPS call includes one-time kernel compile/graph overhead.
warm_tensor = model.encode(ESMProtein(sequence=sequences[0]))
_ = model.logits(warm_tensor, LogitsConfig(sequence=True, return_embeddings=True))

start = time.time()
for i, seq in enumerate(sequences):
    protein_tensor = model.encode(ESMProtein(sequence=seq))
    output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed
        print(f"  {i + 1}/{len(sequences)}  elapsed={elapsed:.1f}s  rate={rate:.2f} seq/s")

total_time = time.time() - start
rate = len(sequences) / total_time

print()
print(f"Total time for {len(sequences)} sequences: {total_time:.1f}s")
print(f"Throughput: {rate:.2f} sequences/sec ({total_time / len(sequences) * 1000:.1f} ms/seq avg)")
print(
    f"Projected time for {FULL_RUN_SIZE:,} sequences: "
    f"{FULL_RUN_SIZE / rate / 60:.1f} min ({FULL_RUN_SIZE / rate / 3600:.2f} hr)"
)
