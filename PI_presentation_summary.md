# ESMC Interpretability Project — Summary for PI Presentation

## 1. Goal

We generated 80,000 candidate protein binders (BoltzGen) against a single
target site on TET1. Rather than treating each design as a black box scored
only by AlphaFold-style structural confidence metrics, we asked a deeper
question: **does the protein language model ESMC (600M parameters) that we
use to embed these sequences actually "know" anything about binding, and if
so, where inside the model does that knowledge live?** The long-term goal is
to use that location as the latent space for a generative model (diffusion)
that designs *new*, improved binders directly — rather than continuing to
rely on brute-force generate-and-filter.

## 2. Pipeline built

- Local inference pipeline for ESMC-600M (Apple Silicon/MPS), including
  diagnosing and working around real bugs in the model's public release
  (broken checkpoint loading, environment conflicts).
- Embedded all 80,000 designs, per-layer, in two conditions: **binder
  sequence alone**, and **binder + target concatenated** ("joint") using
  ESMC's native chain-break token — verified via direct inspection of the
  attention-masking code that this genuinely allows full bidirectional
  attention between binder and target, not just a formatting separator.
- Ran PCA per layer (36 layers × 3 preprocessing variants) to get a
  compressed, interpretable coordinate system for each layer's
  representation.

## 3. Findings, in the order we found them

### 3.1 Computational metrics: intrinsic quality >> interface quality
Correlating PCA axes against BoltzGen's own predicted metrics (iptm, ipsae,
PAE, etc.) across all 80k designs, the **strongest signal by far was intrinsic
sequence quality**, not binding: largest hydrophobic patch size (R²≈0.75) and
liability/developability score (R²≈0.71) were far better predicted by the
embeddings than any interface-specific metric (best: delta-SASA, R²≈0.30).
Interpretation: ESMC is very good at reading off generic sequence-composition
properties, and comparatively weak at reading off target-specific interface
quality from structural-confidence proxies alone.

### 3.2 A promising early lead (layer 14) did not survive scrutiny
An early analysis flagged layer 14 as the best single-layer predictor of
`iptm`. Two independent checks killed this:
- **Length confound**: ~20% of that correlation was attributable to binder
  sequence length, not binding. After controlling for length, several *other*
  layers outperformed layer 14.
- **Real-data check**: once real wet-lab activity data came in, layer 14's
  correlation with actual mCherry activity was ~0 (|r| < 0.16) across every
  variant tested. Confirmed dead by two independent lines of evidence.

This is a methodologically important result in its own right: **a strong
correlation with a computational proxy metric does not imply a real
biological signal** — length, and possibly other confounds, can produce
misleading "hits" that only real experimental data can rule out.

### 3.3 The real finding: layers 7–9, joint condition
Using actual wet-lab mCherry activation data (demethylation reporter assay,
no-TET-co-transfection condition — the condition we're actually optimizing
for) from two independent experiments:

- **Screen set** (14 binders, one experiment) was used to search all 36
  layers × 3 PCA variants × 20 components for axes that (a) rank binders
  correctly by activity, with extra weight on distinguishing "meaningfully
  active" (≥4%) binders, and (b) cleanly separate active from inactive
  binders (AUC).
- A **specific, tight cluster** emerged: layers 7–9 (occasionally 11), PC7 or
  PC8, specifically in the **joint (binder+target) condition** — AUC 0.93–1.00
  for separating active/inactive, weighted concordance 0.83–0.90.
- **Confirmation set** (4 binders, a completely separate, independent
  experiment, never used in the search): 8 of the top 25 candidate axes
  achieved a *perfect* ranking (6/6 pairwise comparisons correct) — versus
  ~1 expected by pure chance. This is genuine cross-experiment validation,
  not overfitting to one dataset.
- **Precise interpretation** (important nuance, checked by inspecting raw
  scores, not just the summary statistic): this axis is a strong **binary
  classifier** (active vs. inactive) and additionally tracks **relative
  ranking within the active range** cleanly (confirmed on the independent
  set, which happens to span 5–10% activity). It does *not* yet show clean
  graded resolution among low-activity binders.
- **Robustness check**: the finding survives even when all flagged/noisy
  data points are excluded, and even strengthens slightly — ruling out data
  quality issues as the driver.

### 3.4 Characterizing the signal
- PC7 explains only ~2% of a layer's total variance (ranked 8th of 50
  components) — a real but minor, specific sub-signal, not a dominant
  direction. Consistent with "binding" being a narrow concept riding on top
  of a much larger generic-sequence-property signal.
- HDBSCAN clustering on this representation found **no discrete clusters**
  separating active from inactive binders — but this is expected, not a
  contradiction: PC7's distribution across all 80k designs is a smooth,
  unimodal bell curve at every layer. This is a **continuous gradient**, not
  a discrete group structure, so density-based clustering was never going to
  detect it. Confirms what kind of signal this is.
- Built a full evolution map (36 layers × 20 components, weighted
  concordance against real activity) showing the signal is diffuse and weak
  everywhere else in the network and concentrates sharply at layers 7–9 —
  a genuine localized feature, visualized as a heatmap
  (`sequence_analysis/binding_evolution_heatmap.html`).
- Notably, **binder-alone and joint conditions were statistically
  indistinguishable on every computational metric** (iptm, ipsae, etc.) —
  but the *real* functional signal only shows up in the joint condition.
  This suggests AlphaFold-derived structural confidence metrics may be
  missing something about real function that letting the model attend
  across binder and target actually captures.

## 4. Statistical rigor (worth stating explicitly to a PI)

Every finding above was pressure-tested before being trusted:
- Length-deconfounding via partial correlation.
- Train/test replication on random halves of the 80k computational dataset.
- Exact permutation testing (n=4 real data has only 24 possible orderings —
  calculated the real p-value rather than trusting raw correlation, which
  is close to meaningless at that sample size).
- Cross-experiment confirmation (screen on one real experiment, confirm on
  a completely separate one) — the strongest test available given current
  data.
- Robustness checks against excluding flagged/noisy data points.
- A caught and corrected data-integrity bug along the way (an early plasmid
  numbering mix-up), underscoring the value of auditing rather than trusting
  results that look exciting.

## 5. How this can improve binders against the current site

- **Computational pre-screen for future design rounds**: score new candidate
  designs on the layer 7–9/PC7 (joint) axis before synthesis, to prioritize
  likely-active binders and reduce wasted wet-lab cycles. Best used as a
  secondary filter layered on top of existing structural-confidence filters,
  since it's most reliable at distinguishing "meaningfully active" from not.
- **Combine with the intrinsic-quality signal**: hydrophobic patch size and
  liability score are even more strongly predicted by the embeddings
  (though not yet validated against real activity data the way the binding
  axis was) — worth validating next, and combining both signals into one
  design-quality score.
- **Reinforces target-aware design**: since the real signal only appears in
  the joint condition, any future scoring model or generative model should
  condition on the target, not just the binder sequence alone.
- **Directly informs the diffusion-model plan**: layers 7–9 are now a far
  better-supported choice of latent layer for a future generative model than
  the earlier (now-debunked) layer 14 guess.
- **Next validation step**: more real no-co-transfection data (already
  planned) to sharpen resolution in the low-activity range, where the axis
  is currently a good classifier but not yet a fine-grained ranker.

## 6. Multivalent binder design (Genie3)

Current best binders show modest activity (roughly 5–10%) relative to the
dCas9-TET direct-fusion positive control (36–73% across experiments) — real
function, but well short of maximal potency. A standard strategy for closing
that gap without needing higher per-domain affinity is **avidity**:
presenting multiple binding domains on one scaffold so simultaneous
multi-point engagement multiplies effective binding strength, even when each
individual interaction is modest.

Proposed approach:
- Use the existing computational pipeline (layer 7–9/PC7 score + intrinsic
  quality filters) to select the best-scoring individual binder domains as
  candidate "arms."
- Use Genie3 (motif-scaffolding protein backbone generation) to design a
  single larger scaffold presenting multiple copies of a strong binder
  domain, or combinations of domains, in a defined geometric arrangement.
- Multivalent designs could present multiple copies of the same binder
  (pure avidity), or combine binders against different sites on the same
  target for combined avidity and improved specificity.
- Re-apply the same embedding/scoring pipeline to the resulting multivalent
  candidates before synthesis — the infrastructure built here generalizes
  directly, it doesn't need to be rebuilt.

## 7. Targeting other sites

- The full methodology (embed → PCA → correlate against real data → validate
  rigorously) is target-agnostic and can be reapplied wholesale to a binder
  library against any new site.
- **Open, well-motivated question**: does the layer 7–9/PC7 axis found here
  reflect something general about how ESMC represents protein-protein
  binding, or is it specific to this target/binder family? The cleanest test
  is checking whether it also predicts real activity for binders against a
  *different* site once that data exists. If it transfers, that's strong
  evidence of a general binding-relevant feature inside the model — a much
  bigger finding. If it doesn't, it's still useful, just narrower in scope.
- We already have a 29-binder dataset spanning 6 different target sites from
  earlier screening (collected under TET-co-transfection conditions, so not
  directly comparable to the primary no-co-transfection results) that could
  be revisited specifically for this cross-site generalization question.

## 8. Suggested next steps

1. Incorporate more real no-co-transfection data (in progress) to sharpen
   the low-activity end of the axis and further validate layers 7–9.
2. Validate the intrinsic-quality axes (hydrophobic patch, liability score)
   against real activity data the same rigorous way.
3. Test cross-site generalization of the layer 7–9/PC7 finding.
4. Begin scoping the diffusion-model architecture using layers 7–9 as the
   target latent space.
5. Parallel track: use the scoring pipeline to select candidate arms for
   Genie3-based multivalent scaffold design.
