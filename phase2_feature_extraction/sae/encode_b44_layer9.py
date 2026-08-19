"""One-off: encode b44 (Binder3-4)'s layer-9 joint per-residue activations
fresh -- it's the one real, wet-lab-tested binder NOT in the 80k pool (see
docs/TODO.md), so it has no entry in the extracted layer_09 activations
file. Reuses the exact same model-loading pattern as
embed_residue_layers_7_8_9.py for consistency (same weights path, same
tokenizer, same layer-selection logic).

Usage:
    python encode_b44_layer9.py
"""

from pathlib import Path

import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root

B44_SEQ = "NENKKKFLEKLKEILKELEADLKEDAALKAKAEAGEDPAVSYAKIAAKYEVYAELFKEVAKLAKELGLEKVSKAAEELAKELKAAAKEAEAKLKAEYAKKPETKAAADKAYEETYKEYKEKAKKKIEELRKLVE"
TARGET_SEQ = "DATHTQIEEDVATQLTQLASIIKINYIKPE"
LAYER = 9
D_MODEL = 1152

OUT_PATH = Path(__file__).resolve().parent / "b44_layer9_binder_residues.npy"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

weights_path = data_root("esmc-600") / "data" / "weights" / "esmc_600m_2024_12_v0.pth"
state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
model = ESMC(
    d_model=D_MODEL, n_heads=18, n_layers=36,
    tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False,
).eval()
model.load_state_dict(state_dict, strict=True)
model = model.to(device)

joint_seq = f"{B44_SEQ}|{TARGET_SEQ}"
protein_tensor = model.encode(ESMProtein(sequence=joint_seq))
output = model.logits(protein_tensor, LogitsConfig(return_hidden_states=True))
hidden = output.hidden_states  # (36, 1, L, D)
selected = hidden[LAYER, 0, :, :].float().cpu().numpy()  # (L, D)

binder_len = len(B44_SEQ)
# <cls> binder <chainbreak> target <eos> -- same convention as
# embed_residue_layers_7_8_9.py: binder residues at [1, 1+binder_len)
binder_only = selected[1 : 1 + binder_len]
assert binder_only.shape == (binder_len, D_MODEL), binder_only.shape

np.save(OUT_PATH, binder_only.astype(np.float16))
print(f"wrote {OUT_PATH}: shape={binder_only.shape}")
