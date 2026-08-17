import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root

GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

# esm==3.3.0's ESMC.from_pretrained("esmc_600m") is broken: it points
# load_torch_model at the snapshot root instead of data/weights/*.pth.
# Load the checkpoint directly instead.
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

protein = ESMProtein(sequence=GFP)
protein_tensor = model.encode(protein)
output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))

print(f"logits shape: {tuple(output.logits.sequence.shape)}")
print(f"embeddings shape: {tuple(output.embeddings.shape)}")
