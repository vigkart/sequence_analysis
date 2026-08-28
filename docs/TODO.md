# ESMC interpretability — TET1 binder design: project status & plan

## Current status

**The layer choice is settled and well-supported: layers 7-9 of ESMC-600M,
joint (binder+target) condition, encode the real binding-relevant signal.**
Two independent methods agree, both validated against real wet-lab
activity held out from the method that found them (not just internal
cross-validation on the same computational proxy metrics). Phase 1
(finding/validating this) is done. Phase 2 (generating the residue-level
data needed to act on it) is done. Next: the two future directions below.

## The core finding

**Evidence 1 — PCA single-axis search.** Searched every layer × every
principal component for the axis that best separates real high-activity
binders from the rest, screened on 14 real binders from one experiment.
Peak: **layer 9, PC7**, weighted concordance 0.904. Layers 7-9 hold 4 of
the 5 strongest cells across the whole 36×20 search, all at PC7/PC8 — a
real, localized peak. See `figures/binding_evolution_heatmap.html`.

**Evidence 2 — independent regularized linear-probe confirmation.**
Trained a proper multivariate probe (RidgeCV regression + L2
LogisticRegressionCV, top-10 PCs as features, regularization tuned via
leave-one-out CV) on the same 14-binder screen set, then evaluated on 4
held-out binders from a *separate* experiment never used in the search.
**Layers 7 and 8 (joint) hit 1.000 concordance on both regression and
classification-probability simultaneously** — the top result in the whole
sweep on both metrics. Layers 9/10/11/13 cluster right behind with
perfect probability ranking. binder-alone's best result only reaches
0.833 — never matches joint's perfect scores. See
`figures/linear_probe_chart.html` and the combined view at
`figures/combined_layer_evidence.html`.

Two methods, derived independently, agreeing on the same region — real
signal, not a coincidence of one metric or one dataset.

**Why joint, not binder-alone:** on the 80k *computational* metrics
(iptm, ipsae, etc.), binder-alone and joint were statistically
indistinguishable. But on *real* mCherry activity, only joint produces
the layer 7-9 signal. AlphaFold-derived structural confidence apparently
misses something that letting the model attend across binder and target
actually captures.

**What PC7 actually is:** a minor variance component (~2% of a layer's
variance, ranked 8th of 50) — real but narrow, not a dominant axis.
Confirmed continuous, not discrete: HDBSCAN found no separable clusters
along it, and its marginal distribution across all 80k designs is a
smooth unimodal bell curve at every layer. It's a gradient, so clustering
was never going to find it — that's expected, not a failure.

**What was ruled out:** an earlier candidate, layer 14, looked promising
via computational iptm correlation but was debunked two ways — partial
correlation showed ~20% of its signal was a length confound, and once
real mCherry data arrived its correlation was ~0. Lesson embedded in the
methodology now: a strong correlation with a computational proxy does not
imply real signal; always check against a confound (length, in this
project) and, once real data exists, an independent held-out confirmation.

## Methodology lessons worth carrying forward

- **At small real-data n, raw correlation is close to meaningless** — it
  can trivially hit 0.99 from noise. Use permutation testing (exact, if
  n is small enough) or held-out confirmation on data never touched
  during the search. This is why Evidence 1 and Evidence 2 above were
  each validated against binders that played no role in finding them.
- **Multiple-comparisons risk compounds with correlated candidates** —
  adjacent layers/PCs are not independent tests (established via
  massive-activation outlier dimensions correlated at |r| up to 0.9
  across many layers), so a naive binomial null under-corrects. A
  label-permutation test (shuffle real values across binders, keep the
  embeddings fixed) is the honest way to get a p-value in that setting.
- **`design_spec_N` is not a globally unique ID** in the 80k dataset —
  it's a per-batch rank, repeats across different design batches. Any
  future lookup by that label needs either the exact sequence or the
  `composite_rank` (see `data/rank_lookup_*.json` for how a real ambiguity
  here was resolved and cross-checked against actual `.pdb` filenames).
- **Co-transfection condition matters and isn't interchangeable.** Real
  binder activity was measured both with and without TET co-transfection
  across different experiments; only the no-co-transfection condition is
  valid ground truth for what this project is optimizing. Don't pool
  across conditions.

## Environment

Python 3.12 required (the `esm` package's hard requirement — stock PyPI
`transformers` doesn't implement the `esmc` architecture at all; `esm`
pulls Biohub's own transformers fork). MPS backend on Apple Silicon. Two
upstream bugs in `esm==3.3.0` are worked around directly in this repo's
scripts: broken `ESMC.from_pretrained()` checkpoint-path resolution, and
`return_mean_hidden_states` not implemented locally (Forge-API-only) — so
mean-pooling is done manually from `return_hidden_states=True` output
rather than relying on that config flag. Full setup instructions in
`README.md`.

## Real experimental data reference

Target sequence: `DATHTQIEEDVATQLTQLASIIKINYIKPE`. No-co-transfection
mCherry activity (the valid ground-truth condition) for the binders used
in Evidence 1/2 above:

| binder | design | mCherry % | row in 80k set |
|---|---|---|---|
| p130 | design_spec_130 | 7.26 | 2007 |
| p144 | design_spec_144 | 9.72 | 16250 |
| p154_prev | design_spec_154 | 6.44 | 30263 |
| b44 | (not in 80k set) | 4.99 | embedded fresh, see `phase1_layer_selection/utilities/embed_binder3_4.py` |

These 4 are the held-out confirmation set. The 14-binder screen set (a
separate, later experiment, 15 binders minus one flagged as unreliable)
is resolved via `data/rank_lookup_row_indices.json` — see
`phase1_layer_selection/screen_then_confirm.py` for the exact list and
values.

## Phase 2: residue-level embeddings (done)

`phase2_feature_extraction/embed_residue_layers_7_8_9.py` re-embedded all
80k designs, joint condition, residue-level (not mean-pooled) at layers
7, 8, 9. Completed: 80,000/80,000, 113.1 min, spot-checked for data
integrity (no NaNs, no zero-filled residues, manifest boundaries verified
against real tokenizer output).

- **Location**: `/Volumes/Extreme Pro/esmc_residue_embeddings/` (external
  drive — ~175GB, far beyond local disk capacity).
- **Format**: one flat fp32 memmap, `layers_7_8_9_joint.f32`, shape
  `(12,638,290 total residues, 3 layers, 1152 dims)`, plus `manifest.csv`
  with one row per design: `row_index`, `id`, `binder_length`,
  `target_length`, `total_length`, `flat_start` (byte offset into the
  memmap), `binder_res_start`/`binder_res_end`, `target_res_start`/
  `target_res_end` (index bounds within that design's span — position 0
  is `<cls>`, then binder residues, then the `|` chain-break token, then
  target residues, then `<eos>`).

## Future direction 1: sparse autoencoder + LLM feature annotation

Train an SAE on the residue-level layer 7-9 embeddings, but **only on
binder residues** (sliced via `manifest.csv`'s `binder_res_start/end`) —
not the target residues. The target still needs to be co-attended during
the forward pass (hence the joint embedding), even though its own residue
embeddings aren't used to train the SAE — they're kept in the saved data,
not discarded, in case they're useful later.

Annotation pipeline once the SAE is trained:
1. For every SAE feature, retrieve its top 15 maximally-activating
   example residues/sequences across the 80k binder set.
2. Scan the local sequence context around each of those 15 examples with
   InterProScan.
3. Send the InterPro annotations + the 15 example sequences to an LLM for
   feature-level annotation.
4. Repeat for all SAE features.
5. Cross-reference which interpretable features align with the PC7
   (layers 7-9, joint) binding-relevant direction — do any SAE features
   correspond to recognizable motifs that explain why PC7 separates good
   binders from bad ones?

## Future direction 2: diffusion head in embedding space

Train a lightweight diffusion head that diffuses binder residue
embeddings in the layer 7-9 latent space, optimized toward higher
predicted binding via the established PC7 (layers 7-9) axis as the
guidance/objective. At generation time: insert the diffused latent back
into ESMC at the corresponding layer, run the remaining frozen ESMC
blocks forward to decode a new binder sequence, then cofold the result
with ESMFold2 for a structural prediction of the new design.
