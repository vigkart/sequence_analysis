from pathlib import Path

import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root

ROOT = Path(__file__).resolve().parent.parent
BINDER3_4 = "NENKKKFLEKLKEILKELEADLKEDAALKAKAEAGEDPAVSYAKIAAKYEVYAELFKEVAKLAKELGLEKVSKAAEELAKELKAAAKEAEAKLKAEYAKKPETKAAADKAYEETYKEYKEKAKKKIEELRKLVE"
TARGET_SEQ = "DATHTQIEEDVATQLTQLASIIKINYIKPE"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
weights_path = data_root("esmc-600") / "data" / "weights" / "esmc_600m_2024_12_v0.pth"
state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
model = ESMC(d_model=1152, n_heads=18, n_layers=36, tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False).eval()
model.load_state_dict(state_dict, strict=True)
model = model.to(device)


def mean_pooled_layers(sequence: str) -> np.ndarray:
    protein_tensor = model.encode(ESMProtein(sequence=sequence))
    output = model.logits(protein_tensor, LogitsConfig(return_hidden_states=True))
    hidden = output.hidden_states  # (36, 1, L, 1152)
    residues_only = hidden[:, 0, 1:-1, :]
    return residues_only.mean(dim=1).float().cpu().numpy()  # (36, 1152)


binder_alone = mean_pooled_layers(BINDER3_4)
joint = mean_pooled_layers(f"{BINDER3_4}|{TARGET_SEQ}")

np.save(ROOT / "stageA_mean_pooled" / "binder3_4_binder_alone.npy", binder_alone)
np.save(ROOT / "stageA_mean_pooled" / "binder3_4_joint.npy", joint)
print("binder_alone shape:", binder_alone.shape)
print("joint shape:", joint.shape)
print("done")
