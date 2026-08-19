"""Narrower version of inspect_top_binder_features.py + discriminate_strong_vs_
weak_features.py: instead of ranking each strong binder's "distinctive"
features over its WHOLE sequence (which is dominated by generic scaffold/
packing features, per docs/sae_feature_labeling_findings.md and the
new157/new241 motif's failed structural check), restrict the search to only
the residues actually near the binder-target interface in the predicted
complex structure. The hypothesis: a real binding-relevant feature should
show up locally, at the interface, even if it's diluted into irrelevance
when pooled over the whole ~140-residue design.

Pipeline:
  1. For each of the 4 strongest real binders (top 3 of the 14-experiment:
     new157, new089, new241; top 1 of the 4-experiment: p144) and, for
     comparison, the 4 weak binders that have structures (new249, new211,
     new226, new163 -- b44 has no structure, not from the 80k pool),
     locate and verify (exact sequence match, not just filename) the
     correct predicted-complex structure file under Final_designs/.
  2. Per binder, compute each residue's CA-CA distance to the nearest
     target-chain residue, and define "interface residues" as the closest
     quartile (percentile <= 25) of that binder's own distribution --
     data-driven, not a fixed Angstrom cutoff, so it adapts to each
     structure's own geometry.
  3. Run the SAE per-residue (not pooled) and restrict the "surprise"
     ranking (rarity x intensity-relative-to-typical, same definition as
     inspect_top_binder_features.py) to interface residues only, for each
     strong binder.
  4. Check candidates against the weak binders two ways: (a) the strict
     whole-design test from discriminate_strong_vs_weak_features.py (fires
     anywhere at all, including b44) and (b) for weak binders with
     structures, whether it fires within their own interface region
     specifically.

Usage:
    python interface_restricted_feature_search.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import MMCIFParser

from data import center_scale
from sae_model import SparseAutoencoder

SAE_DIR = Path(__file__).resolve().parent
EMBED_DIR = Path("/Volumes/Extreme Pro/esmc_residue_embeddings/sae_binder_only/layer_09")
CHECKPOINT = SAE_DIR / "checkpoints" / "layer_09_run1" / "best.pt"
B44_ACTS_PATH = SAE_DIR / "b44_layer9_binder_residues.npy"
STRUCT_DIR = Path(
    "/Users/vigneshkarthik/Documents/liuLab/TET_binder_design/Final_designs/"
    "boltzgen_conf_1_batch_generation-top-1000-by-composite"
)

STRONG = {"new157": 9.19, "new089": 5.12, "new241": 4.92, "p144": 9.72}
WEAK = {"new249": 0.64, "new211": 0.92, "new226": 1.05, "new163": 1.41, "b44": 4.99}

# Verified via exact sequence match against designed_sequence in
# data/boltzgen_conf_1_batch_generation-scores.csv (row_index-indexed) --
# design_spec_N repeats across the 80k, so filename alone is not trustworthy.
STRUCT_FILES = {
    "new157": "rank0014_design_spec_157.cif",
    "new089": "rank0077_design_spec_089.cif",
    "new241": "rank0019_design_spec_241.cif",
    "p144": "rank0465_design_spec_144.cif",
    "new249": "rank0056_design_spec_249.cif",
    "new211": "rank0009_design_spec_211.cif",
    "new226": "rank0004_design_spec_226.cif",
    "new163": "rank0054_design_spec_163.cif",
}
INTERFACE_PERCENTILE = 25.0  # closest quartile of each binder's own distance distribution
TOP_N = 20

# designed_sequence, looked up by row_index from
# data/boltzgen_conf_1_batch_generation-scores.csv (row order == row_index).
BINDER_SEQS = {
    "new157": "SLAGMLAEAARAVLHAAATGKIAPCSSPIAQKLGEAAAKLCAKLKEDHPDLFDSIISLLSKIDPLDAATLARIAERVLADGVDWGRVLACFVLAYAAGKVLKEKGKEDLIPAVADALGDVLGERAGAWIEENGGIEGLVKFVEEAAS",
    "new089": "SEADFRALLRTAVKTLIDNAPESLAEEAADVRKNLDSIVDTVVDHLTANRDKLEKLTVDSLPRVVSRVLADGVDLGRIIACLGLGAYLCSLLIDQGRPELIAPVVAAVRDALVEHCPEELKKAIEAAGEALEAAARALAA",
    "new241": "GRLAEELVSAAVAAAVYGATGEPGPAATPAAARAGTAAGELGLKLLEEHPELVLAIAEKCLEIGPEGTARAVEKALAKDVSWATVAAALAAAGALGRYLAELGRADEAPAMGRAVGEGLVRGAGDWIAANGGIEGLAAKLEAAA",
    "p144": "SPLERARELGEELYPLLERYLDVLEEIVEKHRDLVLGAARKILELGRSEEELIEAGIRVGRLVGRRQPLDKALAIAAAAVGYAVGRLVGEERGLSREEAVRIGALVAGGLLVGAAEVHGISVEEALAILREGLEEVAK",
    "new249": "MSIAEEAREGLRVVLDHPAEVAALAREVLADAVAALLALPPAEAAEVAGRAAYRAVLKVKDGVGVAGAAAGGGVLAGEVRRRYGDAAADAVIRAVAEGIRRGLLELGVSEEEARELAELAEEAARAVAEVVAA",
    "new211": "MTDEELAKIIGDAVYSGASGRPAATPEGQRLGEYLRRLLEKMEKDAGELIQRIVDLVLAAGERAPEVLARAVRRVVSQEPMTWERIIAAMVAGAAVARALIAQGREAEARAVARAVGEAIAEVAGDWIRANGGIDGLIAALEEYAA",
    "new226": "SAAAAAAAARALVERHADAVVKLGQFALKEAPDIVEKTLALLRELGGDSPEGVGRALGRALARALARAGPDPWLLVAGFAVAAAHGLRLAGGDAEKAAEIAAEAAGTAVGVAAERFPEAAALLPEVAKLIEEIGKYLES",
    "new163": "SDVEKIAEAVGAGIYYAATGELPELPDPRVQATAKRLGEIAKKVLKDHPDITRRAADLIMALPGSPAEAAARALGRAVAQPGADWETVVAAAAVAAEVLRRNPEDRTFEAALLVGRGIVEGFAARRPEEATLEKFEEFVSSFVL",
}


def three_to_one(resname: str) -> str:
    return protein_letters_3to1[resname.capitalize()]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_structure_chains(path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(path.stem, path)
    chains = {}
    for model in structure:
        for chain in model:
            residues = []
            for res in chain:
                if res.id[0] != " ":
                    continue
                try:
                    aa = three_to_one(res.resname)
                except KeyError:
                    continue
                ca = res["CA"].coord if "CA" in res else None
                residues.append((aa, ca))
            if residues:
                chains[chain.id] = residues
        break
    return chains


def interface_mask_for(name: str) -> np.ndarray:
    """Returns a boolean array (length = binder length) marking the
    closest-quartile-to-target residues, in sequence order."""
    path = STRUCT_DIR / STRUCT_FILES[name]
    chains = load_structure_chains(path)

    binder_chain = None
    for cid, residues in chains.items():
        seq = "".join(aa for aa, _ in residues)
        if seq == BINDER_SEQS[name]:
            binder_chain = residues
            break
    assert binder_chain is not None, f"no exact sequence match for {name} in {path.name}"

    target_coords = np.array(
        [ca for cid, res in chains.items() if "".join(a for a, _ in res) != BINDER_SEQS[name]
         for _, ca in res if ca is not None]
    )
    dists = np.array(
        [np.min(np.linalg.norm(target_coords - ca, axis=1)) if ca is not None else np.inf
         for _, ca in binder_chain]
    )
    cutoff = np.percentile(dists, INTERFACE_PERCENTILE)
    mask = dists <= cutoff
    print(f"  {name}: {mask.sum()}/{len(mask)} interface residues "
          f"(dist <= {cutoff:.1f}A, full range {dists.min():.1f}-{dists.max():.1f}A)")
    return mask


def encode_full(name, real_binders, index_by_row, acts_file, model, mean, scale, device):
    r = index_by_row[real_binders[name]]
    acts = np.array(acts_file[r.start : r.start + r.length]).astype(np.float32)
    x = torch.from_numpy(acts).to(device)
    x_proc = center_scale(x, mean, scale)
    with torch.no_grad():
        codes = model.encode(x_proc)
    return codes.cpu().numpy()  # (L, 16384)


def main() -> None:
    device = pick_device()
    with open(SAE_DIR / "real_binder_row_indices.json") as f:
        real_binders = json.load(f)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean, scale = ckpt["mean"].to(device), ckpt["scale"]

    index_df = pd.read_csv(EMBED_DIR / "index.csv")
    index_by_row = {r.row_index: r for r in index_df.itertuples()}
    acts_file = np.load(EMBED_DIR / "activations.npy", mmap_mode="r")

    stats = pd.read_csv(SAE_DIR / "feature_analysis_results" / "layer_09" / "feature_stats.csv")
    density = stats.set_index("feature")["density"]
    mean_when_active = stats.set_index("feature")["mean_activation_when_active"]
    labels_df = pd.read_csv(SAE_DIR / "feature_analysis_results" / "layer_09" / "feature_labels.csv").set_index("feature")["label"]

    print("Computing interface masks from predicted structures...")
    interface_masks = {}
    for name in list(STRONG.keys()) + [n for n in WEAK if n != "b44"]:
        interface_masks[name] = interface_mask_for(name)

    print("\nEncoding all binders through the SAE...")
    full_codes = {}
    for name in list(STRONG.keys()) + [n for n in WEAK if n != "b44"]:
        full_codes[name] = encode_full(name, real_binders, index_by_row, acts_file, model, mean, scale, device)
    weak_whole_max = {}
    for name in WEAK:
        if name == "b44":
            weak_whole_max[name] = np.load(B44_ACTS_PATH).astype(np.float32)
            with torch.no_grad():
                x = torch.from_numpy(weak_whole_max[name]).to(device)
                x_proc = center_scale(x, mean, scale)
                weak_whole_max[name] = model.encode(x_proc).max(dim=0).values.cpu().numpy()
        else:
            weak_whole_max[name] = full_codes[name].max(axis=0)

    print(f"\n{'='*100}\nInterface-restricted top-{TOP_N} 'surprise' features per strong binder\n{'='*100}")
    top_features_by_binder = {}
    for name in STRONG:
        codes = full_codes[name]
        mask = interface_masks[name]
        max_act = codes[mask].max(axis=0)  # (16384,) restricted to interface residues

        active = np.nonzero(max_act)[0]
        dens = density.reindex(active).to_numpy()
        mwa = mean_when_active.reindex(active).to_numpy()
        mwa_safe = np.where(mwa > 0, mwa, 1.0)
        surprise = (1.0 / np.maximum(dens, 1e-6)) * (max_act[active] / mwa_safe)
        order = np.argsort(-surprise)
        top = active[order][:TOP_N]
        top_features_by_binder[name] = set(top.tolist())

        print(f"\n{name} (activity={STRONG[name]}%, {mask.sum()} interface residues, "
              f"{len(active)} features active at interface)")
        for feat in top:
            lbl = labels_df.get(feat, "?")
            print(f"  feature={feat:6d}  density={density.get(feat, float('nan')):.5f}  "
                  f"interface_max_act={max_act[feat]:.2f}")
            print(f"    {lbl[:180] if isinstance(lbl, str) else lbl}")

    print(f"\n{'='*100}\nOverlap among strong binders' interface-restricted top-{TOP_N} sets\n{'='*100}")
    from itertools import combinations
    names = list(STRONG.keys())
    any_overlap = False
    all_candidates = set()
    for n1, n2 in combinations(names, 2):
        shared = top_features_by_binder[n1] & top_features_by_binder[n2]
        if shared:
            any_overlap = True
            all_candidates |= shared
            print(f"  {n1} & {n2} share: {sorted(shared)}")
    if not any_overlap:
        print("  No pairwise overlap among interface-restricted top sets.")

    print(f"\n{'='*100}\nChecking EVERY strong binder's interface-restricted top-{TOP_N} feature "
          f"against all {len(WEAK)} weak binders (whole-design silence test)\n{'='*100}")
    survivors = []
    for name in STRONG:
        codes = full_codes[name]
        mask = interface_masks[name]
        max_act = codes[mask].max(axis=0)
        for feat in top_features_by_binder[name]:
            weak_whole = {wn: float(weak_whole_max[wn][feat]) for wn in WEAK}
            n_fires_whole = sum(1 for v in weak_whole.values() if v > 0)

            weak_interface = {}
            for wn in WEAK:
                if wn == "b44":
                    continue
                wmask = interface_masks[wn]
                weak_interface[wn] = float(full_codes[wn][wmask, feat].max())

            status = "SILENT on all weak (whole-design) -- survives" if n_fires_whole == 0 \
                else f"fires on {n_fires_whole}/{len(WEAK)} weak binders (whole-design) -- eliminated"
            if n_fires_whole == 0:
                survivors.append((name, feat, max_act[feat]))
            whole_str = ", ".join(f"{wn}={v:.2f}" for wn, v in weak_whole.items())
            iface_str = ", ".join(f"{wn}={v:.2f}" for wn, v in weak_interface.items())
            print(f"  {name} feature={feat:6d}  strong_iface_act={max_act[feat]:.2f}")
            print(f"    whole-design weak: {whole_str}  -> {status}")
            print(f"    interface-only weak: {iface_str}")

    print(f"\n{'='*100}\nSURVIVORS: interface-restricted, distinctive on a strong binder, "
          f"silent everywhere on all {len(WEAK)} weak binders\n{'='*100}")
    if not survivors:
        print("  None.")
    else:
        for name, feat, act in survivors:
            lbl = labels_df.get(feat, "?")
            print(f"  {name}: feature={feat}  interface_act={act:.2f}  "
                  f"density={density.get(feat, float('nan')):.5f}")
            print(f"    {lbl[:200] if isinstance(lbl, str) else lbl}")


if __name__ == "__main__":
    main()
