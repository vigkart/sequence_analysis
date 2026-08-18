# ESMC interpretability — TET1 binder design

Interpretability analysis of ESMC-600M applied to 80,000 AI-designed protein
binders, searching for where (and whether) the model encodes binding-relevant
information, validated against real wet-lab activity data. Full narrative and
findings: [`TODO.md`](TODO.md) (running log) and
[`PI_presentation_summary.md`](PI_presentation_summary.md) (findings summary).

## Setup

Requires Python 3.12 (the `esm` package's hard requirement). Run from the
repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `esm` and `transformers` to specific commits of
Biohub's forks (`git+https://github.com/Biohub/esm.git@<commit>` etc.) —
these are required; stock PyPI `transformers` does not implement the `esmc`
architecture. See `TODO.md` for the full story on why, plus two upstream
bugs in `esm==3.3.0` that are worked around directly in this repo's scripts
(broken checkpoint path resolution, missing `return_mean_hidden_states`
support locally).

Every script resolves its data paths relative to the repo root (via
`Path(__file__).resolve().parent.parent`), not the current working
directory — so they can be run from anywhere, e.g. either of these work
identically:
```bash
python stage_a_pca/check_dim_variance.py
cd stage_a_pca && python check_dim_variance.py
```

## Directory structure

- **`stage_a_pca/`** — Stage A: embed all 80k designs (mean-pooled, both
  conditions), fit PCA per layer, correlate against BoltzGen's computational
  metrics, and the length-deconfounding check that debunked the early
  "layer 14" lead.
- **`real_data_validation/`** — validation against real wet-lab mCherry
  activity across two experiments (n=4 then n=19 real binders). This is
  where the layer 7–9 / PC7 finding was established and independently
  confirmed via a regularized linear probe.
- **`phase2_residue_embeddings/`** — Phase 2: residue-level (not
  mean-pooled) embeddings at layers 7–9 only, joint condition, saved to an
  external drive. Data foundation for the SAE and diffusion-head work
  logged in `TODO.md`.
- **`figures/`** — published Artifact visualizations (HTML) and their
  companion data.
- **`data/`** — small derived/lookup data: `rank_lookup_*.json` (resolves
  the 15-binder experiment's ambiguous `design_spec_N` labels to unique
  rows — see `TODO.md` for why that was necessary), `linear_probe_results.json`.
- **`docs/`** — this file, `TODO.md`, `PI_presentation_summary.md`.
- Root — `esmc_600M.py` (minimal sanity-test script) and
  `speed_test_500.py` (throughput benchmark) stay at the repo root since
  they're one-off utilities rather than part of a specific stage.

## Data not included in this repo

Large intermediate outputs are `.gitignore`d and live at the **repo root**
(one level up from this file) — must be regenerated locally or copied from
the working machine, GitHub doesn't accept files this large:

- `boltzgen_conf_1_batch_generation-scores.csv` (~72MB, 80k designed
  sequences + BoltzGen metrics — the source data everything else derives
  from; not included since it's project data, not code)
- `stageA_mean_pooled/*.f32` (2 × ~12GB — mean-pooled ESMC embeddings for
  all 80k designs, binder-alone and binder+target-joint conditions)
- `stageA_pca/` (~3.3GB — fitted PCA transforms and per-design scores for
  every layer × condition × preprocessing variant)

Phase 2's residue-level embeddings (~175GB) live entirely outside this repo,
on an external drive: `/Volumes/Extreme Pro/esmc_residue_embeddings/`.

## Script order (roughly chronological)

1. `esmc_600M.py` — minimal working ESMC-600M load + forward pass (GFP
   sanity test); also the reference for the `from_pretrained` checkpoint-path
   bug workaround.
2. `speed_test_500.py` — throughput benchmark on 500 sequences.
3. `stage_a_pca/stageA_layer_selection.py` — embeds all 80k designs, both
   conditions, mean-pooled per layer. Produces `stageA_mean_pooled/`. ~3 hours.
4. `stage_a_pca/check_dim_variance.py` — per-dimension variance audit
   (finds the "massive activation" outlier dimensions).
5. `stage_a_pca/run_pca_all_layers.py` — fits PCA (3 variants) per layer
   per condition, saves scores + the fitted transforms. Produces `stageA_pca/`.
6. `stage_a_pca/correlate_pca_with_binding.py` /
   `stage_a_pca/correlate_all_metrics.py` — correlate PCA axes against
   BoltzGen's computational metrics (iptm, ipsae, etc.).
7. `stage_a_pca/validate_layer14.py` — length-deconfounding + train/test
   replication check on the early "layer 14" lead (debunked — see
   `TODO.md` §"Layer-14 validation").
8. `real_data_validation/embed_binder3_4.py` — embeds the one real binder
   not present in the 80k set.
9. `real_data_validation/project_real_binders.py` /
   `real_data_validation/layer_hit_counts.py` /
   `real_data_validation/permutation_test.py` — early n=4 real-data
   analysis (superseded once more real data arrived; kept for the
   methodology, notably the exact permutation test).
10. `real_data_validation/project_19_binders.py` — projects all 19 real
    binders (both experiments) onto every PCA axis.
11. `real_data_validation/screen_then_confirm.py` /
    `real_data_validation/audit_screen_confirm.py` — the core result:
    screen on 14 binders from one experiment, confirm on 4 held-out
    binders from a separate experiment. Finds the layer 7–9 / PC7 signal.
12. `real_data_validation/linear_probe_per_layer.py` — independent
    confirmation via a regularized multivariate probe (RidgeCV +
    LogisticRegressionCV), rather than a single hand-picked axis. Agrees
    with the layer 7–9 finding.
13. `real_data_validation/cluster_layers_7_9.py` — HDBSCAN clustering test
    (null result, and why that's expected).
14. `figures/binding_evolution_heatmap.html`,
    `figures/linear_probe_chart.html`, `figures/combined_layer_evidence.html`
    — the visualizations.
15. `phase2_residue_embeddings/embed_residue_layers_7_8_9.py` — Phase 2:
    residue-level embeddings at layers 7–9, joint condition, all 80k
    designs. Saved to the external drive. ~113 minutes.

`data/rank_lookup_row_indices.json` / `data/rank_lookup_sequences.json`
resolve the 15-binder experiment's `design_spec_N` plasmid labels to unique
rows in the 80k dataset — necessary because `design_spec_N` is a per-batch
rank, not a globally unique ID (see `TODO.md` for how this was discovered
and fixed).
