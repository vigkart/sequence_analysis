"""Does the new157/new241 shared motif (features 11736 @ ~residue 22-23,
12086 @ ~residue 118-120 -- see docs/SAE_full_writeup_for_presentation.md
section 7) sit near the binder-target interface, or is it just shared
scaffold architecture unrelated to binding? Checks by loading the actual
predicted complex structures and measuring distance from each candidate
residue to the nearest target-chain residue, compared against the
distribution across ALL binder residues as a calibration baseline.

Sequence is verified against the exact sequence used throughout this
project's SAE analysis before trusting any distance measurement -- see
docs/TODO.md's "design_spec_N is NOT a unique ID" note; filenames alone
are not trustworthy, since design_spec_N repeats across the 80k.

Usage:
    python check_motif_interface_proximity.py
"""

from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1

def three_to_one(resname: str) -> str:
    return protein_letters_3to1[resname.capitalize()]

STRUCT_DIR = Path("/Users/vigneshkarthik/Documents/liuLab/TET_binder_design/Final_designs/boltzgen_conf_1_batch_generation-top-1000-by-composite")

# rank key -> (expected sequence prefix for verification, positions of interest 0-indexed)
TARGETS = {
    "new157": {
        "file": "rank0014_design_spec_157.cif",
        "expected_seq": "SLAGMLAEAARAVLHAAATGKIAPCSSPIAQKLGEAAAKLCAKLKEDHPDLFDSIISLLSKIDPLDAATLARIAERVLADGVDWGRVLACFVLAYAAGKVLKEKGKEDLIPAVADALGDVLGERAGAWIEENGGIEGLVKFVEEAAS",
        "positions": {"feature_11736": 22, "feature_12086": 120, "feature_5521": 63},
    },
    "new241": {
        "file": "rank0019_design_spec_241.cif",
        "expected_seq": "GRLAEELVSAAVAAAVYGATGEPGPAATPAAARAGTAAGELGLKLLEEHPELVLAIAEKCLEIGPEGTARAVEKALAKDVSWATVAAALAAAGALGRYLAELGRADEAPAMGRAVGEGLVRGAGDWIAANGGIEGLAAKLEAAA",
        "positions": {"feature_11736": 23, "feature_12086": 118, "feature_5521": 61},
    },
}


def get_chain_residues(structure):
    """Returns {chain_id: [(res_id, resname_1letter, np.array ca_coord or None)]}."""
    chains = {}
    for model in structure:
        for chain in model:
            residues = []
            for res in chain:
                if res.id[0] != " ":  # skip heteroatoms/water
                    continue
                try:
                    aa = three_to_one(res.resname)
                except KeyError:
                    continue
                ca = res["CA"].coord if "CA" in res else None
                residues.append((res.id[1], aa, ca))
            if residues:
                chains[chain.id] = residues
        break  # first model only
    return chains


def main() -> None:
    parser = MMCIFParser(QUIET=True)

    for name, info in TARGETS.items():
        path = STRUCT_DIR / info["file"]
        print(f"\n{'='*80}\n{name}  ({info['file']})\n{'='*80}")
        structure = parser.get_structure(name, path)
        chains = get_chain_residues(structure)
        print(f"  chains found: {[(cid, len(res)) for cid, res in chains.items()]}")

        # Identify binder chain by sequence match against our known sequence.
        binder_chain_id = None
        for cid, residues in chains.items():
            seq = "".join(aa for _, aa, _ in residues)
            if seq == info["expected_seq"]:
                binder_chain_id = cid
                print(f"  chain {cid}: EXACT sequence match to our binder sequence ({len(seq)} residues) -- verified, not just filename-trusted")
                break
        if binder_chain_id is None:
            print("  WARNING: no chain matched our expected sequence exactly -- checking close matches")
            for cid, residues in chains.items():
                seq = "".join(aa for _, aa, _ in residues)
                print(f"    chain {cid} seq ({len(seq)} aa): {seq[:60]}...")
            continue

        binder_residues = chains[binder_chain_id]
        target_residues = [r for cid, res in chains.items() if cid != binder_chain_id for r in res]
        target_coords = np.array([ca for _, _, ca in target_residues if ca is not None])
        print(f"  target chain(s): {len(target_coords)} residues with CA coords")

        # Distance from every binder residue to nearest target residue --
        # this is our calibration baseline (what's a "typical" distance?).
        all_dists = []
        for res_id, aa, ca in binder_residues:
            if ca is None or len(target_coords) == 0:
                continue
            d = np.min(np.linalg.norm(target_coords - ca, axis=1))
            all_dists.append(d)
        all_dists = np.array(all_dists)
        print(f"  binder-to-target min-distance distribution: min={all_dists.min():.1f}A "
              f"median={np.median(all_dists):.1f}A max={all_dists.max():.1f}A")

        # Now check our specific candidate positions.
        pos_to_res = {res_id - min(r[0] for r in binder_residues): (res_id, aa, ca) for res_id, aa, ca in binder_residues}
        for label, pos0indexed in info["positions"].items():
            if pos0indexed not in pos_to_res:
                print(f"  {label} (0-indexed pos {pos0indexed}): position not found in structure residue list")
                continue
            res_id, aa, ca = pos_to_res[pos0indexed]
            if ca is None:
                print(f"  {label} (0-indexed pos {pos0indexed}, residue {aa}{res_id}): no CA coordinate available")
                continue
            d = np.min(np.linalg.norm(target_coords - ca, axis=1))
            percentile = (all_dists < d).mean() * 100
            print(f"  {label} (0-indexed pos {pos0indexed}, residue {aa}{res_id}): "
                  f"distance to nearest target residue = {d:.1f}A "
                  f"(percentile {percentile:.0f} of this binder's own distance distribution -- "
                  f"lower percentile = closer to target than typical)")


if __name__ == "__main__":
    main()
