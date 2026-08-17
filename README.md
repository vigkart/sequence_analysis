# ESMC interpretability — TET1 binder design

Interpretability analysis of ESMC-600M applied to 80,000 AI-designed protein
binders, searching for where (and whether) the model encodes binding-relevant
information, validated against real wet-lab activity data. Full narrative and
findings: [`TODO.md`](TODO.md) (running log) and
[`PI_presentation_summary.md`](PI_presentation_summary.md) (findings summary).

## Setup

Requires Python 3.12 (the `esm` package's hard requirement).

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

## Data not included in this repo

Large intermediate outputs are `.gitignore`d and must be regenerated locally
(see script order below) or copied from the working machine — GitHub
doesn't accept files this large:

- `boltzgen_conf_1_batch_generation-scores.csv` (~72MB, 80k designed
  sequences + BoltzGen metrics — the source data everything else derives
  from; not included since it's project data, not code)
- `stageA_mean_pooled/*.f32` (2 × ~12GB — mean-pooled ESMC embeddings for
  all 80k designs, binder-alone and binder+target-joint conditions)
- `stageA_pca/` (~3.3GB — fitted PCA transforms and per-design scores for
  every layer × condition × preprocessing variant)

## Script order (roughly chronological)

1. `esmc_600M.py` — minimal working ESMC-600M load + forward pass (GFP
   sanity test); also the reference for the `from_pretrained` checkpoint-path
   bug workaround.
2. `speed_test_500.py` — throughput benchmark on 500 sequences.
3. `stageA_layer_selection.py` — embeds all 80k designs, both conditions,
   mean-pooled per layer. Produces `stageA_mean_pooled/`. ~3 hours.
4. `check_dim_variance.py` — per-dimension variance audit (finds the
   "massive activation" outlier dimensions).
5. `run_pca_all_layers.py` — fits PCA (3 variants) per layer per condition,
   saves scores + the fitted transforms. Produces `stageA_pca/`.
6. `correlate_pca_with_binding.py` / `correlate_all_metrics.py` — correlate
   PCA axes against BoltzGen's computational metrics (iptm, ipsae, etc.).
7. `validate_layer14.py` — length-deconfounding + train/test replication
   check on the early "layer 14" lead (debunked — see `TODO.md` §"Layer-14
   validation").
8. `embed_binder3_4.py` — embeds the one real binder not present in the 80k
   set.
9. `project_real_binders.py` / `layer_hit_counts.py` / `permutation_test.py`
   — early n=4 real-data analysis (superseded once more real data arrived;
   kept for the methodology, notably the exact permutation test).
10. `project_19_binders.py` — projects all 19 real binders (both
    experiments) onto every PCA axis.
11. `screen_then_confirm.py` / `audit_screen_confirm.py` — the core result:
    screen on 14 binders from one experiment, confirm on 4 held-out binders
    from a separate experiment. Finds the layer 7–9 / PC7 signal.
12. `cluster_layers_7_9.py` — HDBSCAN clustering test (null result, and why
    that's expected).
13. `binding_evolution_heatmap.html` — the layer × PC evolution map figure.

`rank_lookup_row_indices.json` / `rank_lookup_sequences.json` resolve the
15-binder experiment's `design_spec_N` plasmid labels to unique rows in the
80k dataset — necessary because `design_spec_N` is a per-batch rank, not a
globally unique ID (see `TODO.md` for how this was discovered and fixed).
