"""Shared structure-derived interface masks for the real wet-lab-tested
binders, used by both correlate_features_with_binding_interface.py and
lasso_probe_binding_interface.py -- factored out so the two scripts (which
otherwise mirror correlate_features_with_binding.py and
lasso_probe_binding.py) don't duplicate the ~15-entry sequence/structure
lookup and mask computation.

Coverage: 14/14 screen binders have verified structures (exact sequence
match against the predicted complex under Final_designs/
boltzgen_conf_1_batch_generation-top-1000-by-composite/) -- the full
14-binder screen set is intact. Of the 4-binder confirm set, only `p144`
does; `p130` and `p154_prev` were searched for (exact AND substring/fuzzy
match against every rank*_design_spec_{130,154}.cif candidate in that
folder, plus Final_designs/binder_design_analysis's copy and
misc/Binder_Designs_initial) with no match found -- their predicted
structures are not present anywhere in this repo's checkout. `b44` was
never in the 80k pool at all (see encode_b44_layer9.py), so it has no
BoltzGen-predicted structure either. All three are therefore excluded
from any interface-restricted analysis; see the calling scripts'
docstrings for how each one handles the resulting reduced confirm set
(n=1, just p144).

Interface definition: a design-specific residue is "interface" if it's
in the closest quartile (percentile <= 25) of that binder's own
CA-to-nearest-target-residue distance distribution -- adapts to each
structure's own geometry rather than a fixed Angstrom cutoff, same
choice as interface_restricted_feature_search.py.
"""

from pathlib import Path

import numpy as np
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import MMCIFParser

STRUCT_DIR = Path(
    "/Users/vigneshkarthik/Documents/liuLab/TET_binder_design/Final_designs/"
    "boltzgen_conf_1_batch_generation-top-1000-by-composite"
)
INTERFACE_PERCENTILE = 25.0

# designed_sequence, looked up by row_index from
# data/boltzgen_conf_1_batch_generation-scores.csv (row order == row_index).
BINDER_SEQS = {
    "new149": "SAFAQAIIDAAVDTAKYLATGDPSVCKTEAGLLLAEAAARIMEAHAEEVAKIRELLRKIQPLTAEAVARIVEKLFASGEIDLGRLAAAFVLAGVAGRVAKEQGREDLIEEIGRAVGEVLVEKAGDWLKARGGVTAFGREMVKNTA",
    "new241": "GRLAEELVSAAVAAAVYGATGEPGPAATPAAARAGTAAGELGLKLLEEHPELVLAIAEKCLEIGPEGTARAVEKALAKDVSWATVAAALAAAGALGRYLAELGRADEAPAMGRAVGEGLVRGAGDWIAANGGIEGLAAKLEAAA",
    "new226": "SAAAAAAAARALVERHADAVVKLGQFALKEAPDIVEKTLALLRELGGDSPEGVGRALGRALARALARAGPDPWLLVAGFAVAAAHGLRLAGGDAEKAAEIAAEAAGTAVGVAAERFPEAAALLPEVAKLIEEIGKYLES",
    "new146": "SELAEQVGEAVYAALYYVMTGEVPTTSSAAAQRAGLAAGRLAKKVLEDHEELVRSAAEKILALNEDIAEALARVAAKVLSSGKADWATIAAIIAVAAEIAKLLKDTLTEEDLRKIGLAIGEVIEKYAGDLIVKNGGLDGFAAWLEEAAA",
    "new208": "SAAEAAGARIGEVLCEHLDLLRKLARDHADAVNAIVDTCLKLGPENALDCIGRGAGRALARSEGLERGAVAVAAGCAAIVLAERYPELGTPEEAIRRTAVAAGRGAADVLGERGPALVEQALKDTEEVLKKIEAS",
    "new211": "MTDEELAKIIGDAVYSGASGRPAATPEGQRLGEYLRRLLEKMEKDAGELIQRIVDLVLAAGERAPEVLARAVRRVVSQEPMTWERIIAAMVAGAAVARALIAQGREAEARAVARAVGEAIAEVAGDWIRANGGIDGLIAALEEYAA",
    "new249": "MSIAEEAREGLRVVLDHPAEVAALAREVLADAVAALLALPPAEAAEVAGRAAYRAVLKVKDGVGVAGAAAGGGVLAGEVRRRYGDAAADAVIRAVAEGIRRGLLELGVSEEEARELAELAEEAARAVAEVVAA",
    "new089": "SEADFRALLRTAVKTLIDNAPESLAEEAADVRKNLDSIVDTVVDHLTANRDKLEKLTVDSLPRVVSRVLADGVDLGRIIACLGLGAYLCSLLIDQGRPELIAPVVAAVRDALVEHCPEELKKAIEAAGEALEAAARALAA",
    "new174": "GAALAREIADLIMENADAIVAALRNLAPETYDAAVELLAQIPVEELAERAGRALARAINQLPRDPELIAAAIVAGGAVTYEVARARGLSDEEAREVVARAIEVAYQGSPVIQAISDAELLAAVREVVDALLK",
    "new248": "SREELLRTLESALLGIMRGSTEAETPVAARIAEGALAVFKKVEEDAKEIVDKIRKLLEQLAPITPERARRIVEKALDGGVDWGAIVAAFYLAALIGEVLRRHGLEDQLPAIARAIAEAIVEKAGDWILANGGAEGLAERLLSMA",
    "new163": "SDVEKIAEAVGAGIYYAATGELPELPDPRVQATAKRLGEIAKKVLKDHPDITRRAADLIMALPGSPAEAAARALGRAVAQPGADWETVVAAAAVAAEVLRRNPEDRTFEAALLVGRGIVEGFAARRPEEATLEKFEEFVSSFVL",
    "new108": "SELLEKIAENLKKVLEETLKLVKSPRAKELLETLLKNIKDIVEDNADQLEKIYELLKKVVKTREEGLAFFRRVADKLAELASDEAIIAIAFLAGIVATILSELIPEEEHEGLIKELAELVAERADAIVAAQE",
    "new228": "SEAQIEELADRLAEVMGPRVEQLIEKLKEVDKELYDQVLKLVKEAAPTAEEAAYFMGRAAGRAALRAPRDGDLLGALLAAAIAVATEIAAEKGVPPGPEATRRCALLVLRGLADASPEAFERLVALEDGVLRAIEEILS",
    "new157": "SLAGMLAEAARAVLHAAATGKIAPCSSPIAQKLGEAAAKLCAKLKEDHPDLFDSIISLLSKIDPLDAATLARIAERVLADGVDWGRVLACFVLAYAAGKVLKEKGKEDLIPAVADALGDVLGERAGAWIEENGGIEGLVKFVEEAAS",
    "new154": "GMADTLREAGAAFAVSYATGEVVPTASPDATAVAEAALRLLEKVKEEHKDLVDSIESLLAQIDPLTPERVARIVRKVLSSEPLDWGKIAAAFILAGIAGRILRRRGREDLIEPVGRAVGDVVAELAGDFIAENGGVAGLAAKLAEMAA",
    "p144": "SPLERARELGEELYPLLERYLDVLEEIVEKHRDLVLGAARKILELGRSEEELIEAGIRVGRLVGRRQPLDKALAIAAAAVGYAVGRLVGEERGLSREEAVRIGALVAGGLLVGAAEVHGISVEEALAILREGLEEVAK",
}

STRUCT_FILES = {
    "new149": "rank0034_design_spec_149.cif",
    "new241": "rank0019_design_spec_241.cif",
    "new226": "rank0004_design_spec_226.cif",
    "new146": "rank0037_design_spec_146.cif",
    "new208": "rank0003_design_spec_208.cif",
    "new211": "rank0009_design_spec_211.cif",
    "new249": "rank0056_design_spec_249.cif",
    "new089": "rank0077_design_spec_089.cif",
    "new174": "rank0002_design_spec_174.cif",
    "new248": "rank0057_design_spec_248.cif",
    "new163": "rank0054_design_spec_163.cif",
    "new108": "rank0005_design_spec_108.cif",
    "new228": "rank0001_design_spec_228.cif",
    "new157": "rank0014_design_spec_157.cif",
    "new154": "rank0016_design_spec_154.cif",
    "p144": "rank0465_design_spec_144.cif",
}

EXCLUDED_NO_STRUCTURE = ["p130", "p154_prev", "b44"]


def three_to_one(resname: str) -> str:
    return protein_letters_3to1[resname.capitalize()]


def _load_structure_chains(path: Path):
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


def compute_interface_masks(names) -> dict[str, np.ndarray]:
    """Returns {name: bool array, length = binder length, True = interface
    residue (closest quartile to target)}, in FASTA/sequence order."""
    masks = {}
    for name in names:
        path = STRUCT_DIR / STRUCT_FILES[name]
        chains = _load_structure_chains(path)
        binder_chain = None
        for cid, residues in chains.items():
            if "".join(aa for aa, _ in residues) == BINDER_SEQS[name]:
                binder_chain = residues
                break
        assert binder_chain is not None, f"no exact sequence match for {name} in {path.name}"

        target_coords = np.array(
            [ca for cid, res in chains.items() if res is not binder_chain
             for _, ca in res if ca is not None]
        )
        dists = np.array(
            [np.min(np.linalg.norm(target_coords - ca, axis=1)) if ca is not None else np.inf
             for _, ca in binder_chain]
        )
        cutoff = np.percentile(dists, INTERFACE_PERCENTILE)
        masks[name] = dists <= cutoff
        print(f"  {name}: {masks[name].sum()}/{len(masks[name])} interface residues "
              f"(dist <= {cutoff:.1f}A)")
    return masks
