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
`Path(__file__).resolve().parent.parent`, or `.parent.parent.parent` for
scripts one level deeper in a `utilities/` folder), not the current working
directory — so they can be run from anywhere, e.g. either of these work
identically:
```bash
python phase1_layer_selection/utilities/check_dim_variance.py
cd phase1_layer_selection/utilities && python check_dim_variance.py
```

## Directory structure

Organized by project phase, not by date or file type. Within each phase,
scripts split into the main directory (core data generation / the source of
a headline finding) and a `utilities/` subfolder (diagnostics, small
one-off helpers, superseded-but-kept-for-methodology analyses, or scripts
that audit another script's result rather than producing a new one):

- **`phase1_layer_selection/`** — everything about finding and validating
  which ESMC layer encodes binding-relevant information.
  - Main: `stageA_layer_selection.py` (generates the mean-pooled
    embeddings), `run_pca_all_layers.py` (fits PCA), `correlate_pca_with_binding.py`
    / `correlate_all_metrics.py` (correlate against BoltzGen's computational
    metrics), `validate_layer14.py` (debunks the early "layer 14" lead),
    `project_19_binders.py`, `screen_then_confirm.py` (the core layer 7–9 /
    PC7 result), `linear_probe_per_layer.py` (independent confirmation via
    a regularized probe), `cluster_layers_7_9.py` (the clustering
    null-result experiment).
  - `utilities/`: `check_dim_variance.py` (variance/outlier-dim audit),
    `embed_binder3_4.py` (one-off embedding helper), `project_real_binders.py`
    / `layer_hit_counts.py` / `permutation_test.py` (the superseded n=4
    real-data analysis, kept for the methodology), `audit_screen_confirm.py`
    (audits `screen_then_confirm.py`'s result), `esmc_600M.py` (minimal
    sanity-test script), `speed_test_500.py` (throughput benchmark).
- **`phase2_feature_extraction/`** — residue-level (not mean-pooled)
  embeddings at layers 7–9 only, joint condition, saved to an external
  drive. The planned SAE feature-extraction work (see `TODO.md`) lands
  here too once started; a `utilities/` subfolder will hold its small
  helper scripts the same way, once there's more than one main script to
  distinguish it from.
- **`phase3_binder_optimization/`** — not created yet. Future home for the
  generative/diffusion work once it exists (see `TODO.md`'s "Future
  direction 2").
- **`figures/`** — published Artifact visualizations (HTML) and their
  companion data.
- **`data/`** — all project data, also organized phase-wise:
  - `data/phase1_mean_pooled/`, `data/phase1_pca/` — Phase 1's large
    intermediate outputs (gitignored, see below).
  - `data/boltzgen_conf_1_batch_generation-scores.csv` — the source data
    everything else derives from (gitignored, see below).
  - `data/rank_lookup_*.json` — resolves the 15-binder experiment's
    ambiguous `design_spec_N` labels to unique rows (see `TODO.md` for why
    that was necessary).
  - `data/linear_probe_results.json` — raw output of
    `linear_probe_per_layer.py`.
- **`docs/`** — this file, `TODO.md`, `PI_presentation_summary.md`.

## Data not included in this repo

Large intermediate outputs are `.gitignore`d — must be regenerated locally
or copied from the working machine, GitHub doesn't accept files this large:

- `data/boltzgen_conf_1_batch_generation-scores.csv` (~72MB, 80k designed
  sequences + BoltzGen metrics — the source data everything else derives
  from; not included since it's project data, not code)
- `data/phase1_mean_pooled/*.f32` (2 × ~12GB — mean-pooled ESMC embeddings
  for all 80k designs, binder-alone and binder+target-joint conditions)
- `data/phase1_pca/` (~3.3GB — fitted PCA transforms and per-design scores
  for every layer × condition × preprocessing variant)

Phase 2's residue-level embeddings (~175GB) live entirely outside this repo,
on an external drive: `/Volumes/Extreme Pro/esmc_residue_embeddings/`.

## Script order (roughly chronological)

1. `phase1_layer_selection/utilities/esmc_600M.py` — minimal working
   ESMC-600M load + forward pass (GFP sanity test); also the reference for
   the `from_pretrained` checkpoint-path bug workaround.
2. `phase1_layer_selection/utilities/speed_test_500.py` — throughput
   benchmark on 500 sequences.
3. `phase1_layer_selection/stageA_layer_selection.py` — embeds all 80k
   designs, both conditions, mean-pooled per layer. Produces
   `data/phase1_mean_pooled/`. ~3 hours.
4. `phase1_layer_selection/utilities/check_dim_variance.py` —
   per-dimension variance audit (finds the "massive activation" outlier
   dimensions).
5. `phase1_layer_selection/run_pca_all_layers.py` — fits PCA (3 variants)
   per layer per condition, saves scores + the fitted transforms. Produces
   `data/phase1_pca/`.
6. `phase1_layer_selection/correlate_pca_with_binding.py` /
   `phase1_layer_selection/correlate_all_metrics.py` — correlate PCA axes
   against BoltzGen's computational metrics (iptm, ipsae, etc.).
7. `phase1_layer_selection/validate_layer14.py` — length-deconfounding +
   train/test replication check on the early "layer 14" lead (debunked —
   see `TODO.md` §"Layer-14 validation").
8. `phase1_layer_selection/utilities/embed_binder3_4.py` — embeds the one
   real binder not present in the 80k set.
9. `phase1_layer_selection/utilities/project_real_binders.py` /
   `phase1_layer_selection/utilities/layer_hit_counts.py` /
   `phase1_layer_selection/utilities/permutation_test.py` — early n=4
   real-data analysis (superseded once more real data arrived; kept for
   the methodology, notably the exact permutation test).
10. `phase1_layer_selection/project_19_binders.py` — projects all 19 real
    binders (both experiments) onto every PCA axis.
11. `phase1_layer_selection/screen_then_confirm.py` /
    `phase1_layer_selection/utilities/audit_screen_confirm.py` — the core
    result: screen on 14 binders from one experiment, confirm on 4
    held-out binders from a separate experiment. Finds the layer 7–9 /
    PC7 signal.
12. `phase1_layer_selection/linear_probe_per_layer.py` — independent
    confirmation via a regularized multivariate probe (RidgeCV +
    LogisticRegressionCV), rather than a single hand-picked axis. Agrees
    with the layer 7–9 finding.
13. `phase1_layer_selection/cluster_layers_7_9.py` — HDBSCAN clustering
    test (null result, and why that's expected).
14. `figures/binding_evolution_heatmap.html`,
    `figures/linear_probe_chart.html`, `figures/combined_layer_evidence.html`
    — the visualizations.
15. `phase2_feature_extraction/embed_residue_layers_7_8_9.py` — residue-level
    embeddings at layers 7–9, joint condition, all 80k designs. Saved to
    the external drive. ~113 minutes.
