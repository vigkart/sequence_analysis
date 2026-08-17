# ESMC embedding plan for binder designs

## End goal (full pipeline)
Train a diffusion model to design in ESMC's embedding space:
1. **Stage A — layer selection**: determine which ESMC layer most explicitly
   encodes binding, via mean-pooled per-layer embeddings correlated against
   existing binding-quality labels already in the scores CSV
   (`design_to_target_iptm`, `design_ipsae_min`, etc.). Run on a
   representative subset (a few thousand designs), not all 80k — pooling
   doesn't save forward-pass time (the model still runs all 36 layers
   either way), so subsetting is the only way to make this stage fast.
   Test **both** binder-alone and binder+target-joint conditions at each
   layer — binder-alone carries no target info, so this doubles as the
   test of whether joint attention (see below) adds real signal.
2. **Stage B — per-residue extraction at the chosen layer only**: once the
   best layer is identified, re-embed (subset or full 80k as needed) with
   `return_hidden_states=True, ith_hidden_layer=<chosen layer>` to get
   per-residue `[L, D]` tensors at just that one layer. Mean-pooled
   embeddings from Stage A cannot be reused here — diffusion needs
   per-residue structure to generate a variable-length sequence.
3. **Stage C — train a diffusion head** on those per-residue embeddings at
   the chosen layer.
4. **Stage D — generation**: insert a diffused embedding back into ESMC at
   that same layer's position and run the remaining (frozen) transformer
   blocks forward to decode it, then feed the result to ESMFold2 for
   structure design. Mechanically feasible since ESMC's transformer is
   just a plain stack of blocks (`model.transformer.blocks`) — resuming
   from partway through is just slicing that list and continuing the loop.

## Decision: embed two conditions per design (binder-alone, binder+target)
1. **Binder alone** — `ESMProtein(sequence=binder_seq)`
2. **Binder + target, joint** — `ESMProtein(sequence=f"{binder_seq}|{target_seq}")`
   using ESMC's real `chain_break_token` (`"|"`, vocab id 31).

Target is constant across all designs, so embed it alone once and reuse;
only conditions 1 and 2 need to be recomputed per design.

## Why both, not just one
Open question: does letting the binder and target attend to each other
(condition 2) capture real binding-relevant signal, or just add noise/
RoPE-distance artifacts? Verified by reading ESMC's source
(`esm/layers/attention.py`, `esm/models/esmc.py`):
- `|` is a real vocab token but does **not** block attention by default —
  `MultiHeadAttention` masks only where `sequence_id` differs, and
  `ESMC.forward()`'s default `sequence_id` is just a padding mask (same
  value for every real token). So condition 2 gives genuine full
  bidirectional attention across both chains, not a hard separator.
- Developer interpretability findings (SAE features, EC-number layer
  probing) are about single-chain intrinsic function, not pairwise
  binding — they don't settle whether joint attention helps here.
- This is empirically testable: compare the binder-alone embedding
  (condition 1) against the binder-span slice of the joint embedding
  (condition 2) to quantify how much the target's presence moves the
  binder's own representation.

## Practical notes
- Running both conditions ~doubles compute vs. binder-only alone.
- Env: pure-pip venv at `sequence_analysis/.venv` (Python 3.12.11), `esm`
  package installed from `Biohub/esm@main` (pulls Biohub's transformers
  fork — required, since `esmc` isn't a real architecture in stock
  PyPI transformers). MPS backend on M3 Pro.
- Known upstream bug: `ESMC.from_pretrained("esmc_600m")` in esm==3.3.0
  points at the wrong checkpoint path. Worked around in
  `esmc_600M.py` by loading `data/weights/esmc_600m_2024_12_v0.pth`
  directly and calling `model.load_state_dict(...)`.
- Sequences source (full 80k, unfiltered): `designed_sequence - boltzgen`
  column in `boltzgen_conf_1_batch_generation-scores.csv` (80,000 rows).
  Note: `boltzgen_conf_1_batch_generation-all-designs-metrics-passing.csv`
  in this same folder is a filtered subset (~47k) — not the right file
  for embedding all designs.

## Storage constraints (checked: only 66GB free on disk)
- All 37 layers, per-residue, float32: ~20MB/protein → ~1.6TB for 80k.
  **Does not fit. Not viable at full scale.**
- Mean-pooled, all 37 layers: ~170KB/protein → ~13.6GB for 80k. Fine —
  this is what Stage A uses.
- Per-residue, single chosen layer: ~576KB/protein (fp32) → ~46GB for 80k,
  ~23GB in fp16. Tight but workable — this is what Stage B uses.

## Status
- [x] Local ESMC-600M environment working (GFP test passes)
- [x] Speed benchmark on 500 sequences → estimate full-run time
      Result (`speed_test_500.py`, MPS, float32, no batching, single-sequence
      loop): 17.2 seq/s (58ms/seq avg) on binder-only sequences (100-148 aa).
      Projects to ~77.5 min for all 80k binder-alone embeddings.
      Caveat: condition 2 (binder+target joint) will cost more than 2x this,
      since attention is quadratic in length and the concatenated sequence
      is longer than the binder alone — need target sequence length to
      re-estimate condition 2's cost.
- [ ] Decide batching strategy based on benchmark result
- [x] Stage A embeddings: mean-pooled per-layer, binder-alone AND
      binder+target-joint, all 80k designs (`stageA_layer_selection.py`,
      output in `stageA_mean_pooled/`). Took 174.3 min at steady 7.65 seq/s
      (both conditions combined per design). 36 layers (not 37 — verified
      empirically, docstring's n_layers+1 doesn't apply to this local
      class), mean excludes <cls>/<eos>. 13.27GB per condition.
- [x] PCA across all 36 layers, both conditions, 3 variants (correlation;
      covariance all-dims; covariance outliers-excluded) — `run_pca_all_layers.py`,
      output in `stageA_pca/{condition}/layer_NN/`. Top 50 PCs + explained
      variance ratio saved per (layer, condition, variant).
      Findings: outlier dims (500/920/1091/269/522/... recurring set) are
      frequently highly correlated with each other (e.g. (920,1091) ~-0.9
      across layers 16-27) — largely redundant, not independent signals.
      Top 10 PCs already explain 55-93% of variance depending on layer —
      true dimensionality is well under 1152, supports using PCA (not a
      VAE) for the eventual diffusion latent, likely only need ~10-20 PCs.
      Layers 20-21 show a notable variance-concentration spike even after
      excluding known outlier dims — worth watching once correlated
      against binding labels.
- [x] Stage A analysis: correlated each layer/variant's PC scores (both
      conditions) against `design_to_target_iptm`, `design_to_target_ipsae`,
      `min_design_to_target_pae`, `pass_filters`, and binder length —
      `correlate_pca_with_binding.py`. Findings:
      - **Length dominates**: length correlates more strongly with top PCs
        (r up to 0.69) than any binding-quality label does (max ~0.48) in
        almost every layer. Real risk that binding correlations are partly
        confounded by length rather than purely about binding — not yet
        controlled for.
      - **Layer 14** is the most reproducible peak for `iptm` in both
        conditions (r=-0.462 binder-alone, r=-0.481 joint).
      - The layer 20-21 variance spike found in the PCA step does **not**
        correspond to a binding-relevant spike (iptm r ~-0.30 to -0.36,
        weaker than layer 14) — that concentrated variance likely isn't
        about binding.
      - binder-alone vs. joint: essentially tied across all 4 labels
        (differences <0.02 in r). No evidence joint attention adds signal
        here, at least via this linear-PC approach.
      - TODO (not yet done): control for length (partial correlation /
        residualize) before trusting layer 14's iptm correlation as
        genuinely about binding rather than partly length-driven.
- [ ] Stage B: per-residue extraction at the chosen layer only (subset or
      full 80k)
- [ ] Stage C: train diffusion head on Stage B embeddings
- [ ] Stage D: reinsert diffused embedding at chosen layer, run remaining
      ESMC blocks forward, feed to ESMFold2

## Real (wet-lab) validation data — IMPORTANT experimental-condition caveat
Two real datasets exist, and they are **not interchangeable** — different
experimental conditions, likely different mechanisms contributing to signal:

1. **No TET co-transfection** (the condition we're actually optimizing for):
   Binder1=7.26%, Binder2=9.73%, Binder3=6.44%, Binder3-4=4.99% mCherry
   activation (all four target the same site as the 80k design campaign),
   plus dCas9-TET=72.8% as a positive control.
   - n=4 real binders is too small for statistical validation — treat as a
     **directional sanity check only** (does a candidate axis rank
     Binder2 > Binder1 > Binder3 > Binder3-4 correctly?), not proof.
   - dCas9-TET is a large, structurally unrelated fusion protein (full
     Cas9 + TET catalytic domain, likely 1500+ residues) vs. ~100-150 aa
     designed binders. Given length dominates PCA structure (see above),
     dCas9-TET landing at an extreme PC value could just reflect its size,
     not a genuine "demethylation axis" — interpret with real caution.
2. **With TET co-transfection**: 29 binders across 6 different target
   sites (not this project's site), mCherry% + a second unexplained
   numeric column (~-3.5 to -8.0, meaning TBD).
   - **Not valid ground truth for the no-co-transfection optimization
     goal** — do not pool with dataset 1 or use to validate/calibrate an
     axis meant for the no-co-transfection condition.

Plan: test the **pre-derived layer 14 axis** (chosen from the 80k
computational iptm/ipsae correlation, before seeing this real data)
against dataset 1, rather than searching for a new "best" axis fit to
only 4 real points — avoids a multiple-comparisons/overfitting trap given
how many candidate layers/PCs/variants exist relative to n=4.

Sequences received for Binder1/2/3/3-4 (see below); dCas9-TET sequence
still needed if we want to include the positive control.
- Binder1 = design_spec_154, row 30263 (already in 80k dataset)
- Binder2 = design_spec_130, row 2007 (already in 80k dataset)
- Binder3 = design_spec_144, row 16250 (already in 80k dataset)
- Binder3-4 = not in dataset, embedded fresh (`embed_binder3_4.py`,
  output `stageA_mean_pooled/binder3_4_{binder_alone,joint}.npy`)

## Real-data ranking search (n=4, exploratory only — see caveats)
`project_real_binders.py` projected the 4 real binders onto all 36 layers
× 3 PCA variants × 20 PCs (4,320 candidates) and checked ranking
concordance against the true mCherry order. Raw |r| values are
meaningless at n=4 (trivially hits 0.99+ from noise) — the useful metric
is exact-ranking concordance (6/6 pairs correct). 267/4,320 candidates
matched perfectly (chance expectation ~180) — a real but modest excess,
and hits were scattered across all depths with no layer clustering.
`layer_hit_counts.py` found layer 3 and layer 7 tied for the most
per-layer hits (27 each) — looked significant under a naive binomial
null (z=5.61) but that null assumes independent candidates, which is
false here (dims are highly correlated with each other, established
earlier). The correct test — `permutation_test.py`, exact permutation
over all 4!=24 label assignments — put the TRUE assignment at rank 3 of
24 (p=3/24=0.125, not significant); two biologically meaningless
relabelings scored higher than the real one. **Conclusion: n=4 cannot
support any claim about where/whether ESMC encodes binding.** Need more
real data (tomorrow's 15 more binders) before revisiting this.

## Expanded metric correlation (all 26 CSV metrics, not just 4)
`correlate_all_metrics.py` — correlated all interface (20), intrinsic (5),
and composite (1) metrics against all PCA axes, both conditions.
- **Intrinsic metrics correlate far more strongly than interface ones**:
  hydrophobic_patch r=-0.868, liability_score r=-0.843 (R²~0.71-0.75) vs.
  best interface metric (delta_sasa) at only r=-0.546 (R²~0.30). ESMC
  embeddings clearly encode intrinsic sequence-composition properties
  much better than target-specific binding quality.
- binder-alone vs. joint: mean |r| difference across 20 interface metrics
  = -0.0032 (essentially zero); across 5 intrinsic metrics = -0.0094.
  **Joint condition shows no measurable benefit anywhere checked** —
  recommend binder-alone only going forward (cheaper, equally predictive).
- Some CSV columns are redundant: design_iiptm/design_residue_iptm
  produced identical numbers to design_to_target_iptm; design_ipsae_min
  identical to design_to_target_ipsae.

## Layer-14 validation (length control + holdout) — `validate_layer14.py`
- **Holdout replication: real, not overfitting.** Selected best axis
  (joint, layer 14, covariance_all_dims, PC1) on random half A of 80k,
  evaluated on held-out half B: r=-0.482 (select) vs r=-0.480 (holdout) —
  essentially identical both directions. The raw correlation is a stable,
  reproducible pattern, not a multiple-comparisons artifact.
- **Length control: layer 14 is NOT distinctively best once you remove
  length's contribution.** Partial correlation (controlling for length)
  at layer 14 drops from -0.462→-0.357 (binder-alone) and -0.481→-0.395
  (joint) — ~18-23% of the raw correlation was length-driven. Across all
  layers, layer 14 is no longer the peak post-correction: binder-alone
  layers 0 (-0.406) and 21 (-0.409) exceed it; joint layer 12 (-0.418)
  exceeds it. The stretch of layers 7-19 (where layer 14 sits) is
  specifically where length contributes most to the raw correlation.
- **Conclusion**: the iptm relationship is real (not noise) but spread
  fairly evenly across most of the network (partial r ~0.30-0.42
  everywhere) rather than concentrated in one "binding layer." No single
  layer is well-supported as THE layer from this computational analysis
  alone. Revisit layer choice after tomorrow's real data — do not commit
  to a layer for the diffusion pipeline based on this alone.

## CORRECTED mapping (previous 3-binder experiment, no-TET condition)
p130=7.26, p144=9.72, p154_prev=6.44, b44=4.99. (Earlier turns had this
scrambled — 130/144/154 swapped. Confirmed correct by user.)

## IMPORTANT: design_spec_N is NOT a unique ID across the 80k dataset
It's a per-batch rank, repeats dozens of times across different design
batches. Exact-sequence matching (not ID lookup) is required to resolve
any plasmid number to a specific row. For the 15-binder experiment, the
plasmid numbers were resolved via composite_rank in
`Final_designs/boltzgen_conf_1_batch_generation-top-1000-composite-rank.csv`,
cross-checked against actual .pdb filenames in
`Final_designs/boltzgen_conf_1_batch_generation-top-1000-by-composite/`.
Discovered: the new experiment's "154" and the previous experiment's
"154" (p154_prev) are DIFFERENT designs (139 vs 148 residues, no overlap)
— coincidental label collision, not a replicate. Row indices saved in
`rank_lookup_row_indices.json` / sequences in `rank_lookup_sequences.json`.

## 19-binder real dataset (no-TET condition) — `project_19_binders.py`
4 from previous experiment + 15 from new experiment (14 clean, 5 flagged:
new154 too-few-cells, new149/new226/new146 too-few-cells per Zuli,
new157 contamination-suspected duplicate).
- **Layer 14 confirmed dead**: |r| < 0.16 with real mCherry across all
  variants/PCs — independently confirms the length-control finding from
  the 80k computational analysis. Two convergent lines of evidence now
  agree layer 14 was never a real binding layer.
- Broader search (36 layers x 3 variants x 20 PCs x 2 conditions) turned
  up stronger hits (top: binder_alone layer 10, covariance_outliers_excluded,
  PC19, r_clean=-0.741) but most shrink substantially when flagged points
  are included (r_all) — sign of fragility/overfitting, not confirmed
  signal. Most robust candidates (small clean-vs-all gap): binder_alone
  layers 1, 7, 10, 11. Treat as leads for the next data round, not
  conclusions — n=14-19 with 4,320 candidates searched is still thin.

## Follow-up: variance share, clustering, evolution map
- PC7 (layers 7-9) explains only ~2% of variance (rank 8/50) — a real but
  minor sub-signal, not a dominant axis. Consistent with it tracking a
  narrow "binding" concept rather than generic bulk variance.
- HDBSCAN (top-20 PCs, layers 7/8/9, joint) found no meaningful
  binder-separating clusters — all 18 labeled real binders land in the
  same dominant cluster or scatter into noise regardless of activity.
  Confirmed this is expected, not a failure: PC7's marginal distribution
  across all 80k designs is a smooth unimodal bell curve at every layer —
  a continuous gradient, not a discrete group structure, so density-based
  clustering was never going to find it.
- Built a layer(0-35) x PC(0-19) weighted-concordance evolution map
  (`evolution_grid.npy`/`.json`, figure at
  `sequence_analysis/binding_evolution_heatmap.html`, published as an
  Artifact). Confirms peak at layer 9/PC7 (0.904); layers 7-9 hold 4 of
  the 5 strongest cells on the map, concentrated at PC7/PC8; elsewhere
  signal is diffuse across many components, none exceeding ~0.85.
  Known cosmetic bug: em-dash/middot characters render as mojibake
  (missing charset meta tag) — not fixed, user said figure is good enough
  as-is.

## Linear-probe verification (independent of the single-PC search)
`linear_probe_per_layer.py` — per layer, trained a proper regularized
multivariate probe (RidgeCV regression + L2 LogisticRegressionCV
classification, top-10 PCs as features, regularization tuned via
leave-one-out CV *within* the 14, never touching the 4) and evaluated on
the same 4 held-out confirmation points.
- Caught and fixed a real bug before trusting results: RidgeCV's default
  LOO scoring is R², which is undefined for single-point folds — alpha was
  being selected essentially at random (pinned at the grid minimum for
  nearly every layer). Fixed by scoring on MSE instead; alpha selection
  became sensible (varies meaningfully across layers) after the fix.
- **Result confirms the PC7 finding via a fully independent method**:
  joint-condition layers 7 and 8 achieve perfect regression concordance
  (1.000) AND perfect classification-probability concordance (1.000) on
  the 4 held-out points — the top two results in the whole 72-row table on
  both metrics simultaneously. Layers 9/10/11/13 (still joint) cluster
  right behind with perfect probability ranking.
- binder_alone's best result only reaches 0.833 concordance — never
  matches joint's perfect scores. Reinforces that the real signal is
  target-aware (joint), not binder-intrinsic.
- Nuance: probability *ranking* on the 4 is perfect, but the hard
  0.5-threshold *decision* only fires on 1/4 — expected when generalizing
  across separate experimental batches (relative ranking is more robust to
  systematic shifts than absolute calibration); the ranking claim is the
  meaningful one and it held up.
- **Two independent methods (single-PC search-then-confirm, and proper
  regularized multivariate probe) now agree: layers 7-9 (joint) is the
  best-supported layer choice for the diffusion pipeline.**
- Visualized as line charts (joint vs. binder-alone panels, regression +
  classification-probability concordance per layer) at
  `sequence_analysis/linear_probe_chart.html`, published as an Artifact.
- Combined with the PCA heatmap into one page,
  `sequence_analysis/combined_layer_evidence.html`, published as an
  Artifact — both lines of evidence shown together with a shared verdict.

## PAUSED here — resume after more real (no-co-transfection) data.
Do not start diffusion training before layer choice is on firmer ground.
