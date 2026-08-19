# SAE feature labeling — review findings

Layer 9 SAE, all 16,384 features, LLM-labeled (`label_features.py`) plus
InterProScan evidence for 95 candidate features. Reviewed for (1) anything
significant overall, (2) anything actionable for generating better
binders, (3) whether any single feature represents binding — checked with
the same screen-locked-confirm rigor established for PC7 (see
`docs/pc7_sign_reversal_findings.md`), not the sign-agnostic mistake that
investigation started from.

## 1. Labeling quality: healthy

- **16,377 / 16,384 features got a real description**; only 7 came back
  "No clear pattern." The LLM found something interpretable in
  essentially the whole dictionary.
- **36 features' labels explicitly cite Bcl-2-fold InterPro evidence** —
  consistent with the earlier InterProScan run (94% of all hits were
  Bcl-2-superfamily), confirming the evidence merge worked correctly and
  the LLM used it appropriately (citing it as fold-similarity evidence,
  not claiming functional relationship — per the system prompt's
  instruction).
- **Dominant theme across all 16,384 labels, by raw keyword frequency**:
  `hydrophobic` (11,498 mentions), `alanine` (7,630), `small` (4,475),
  `acidic` (3,728), `leucine` (3,522) — overwhelmingly local
  amino-acid-composition and packing motifs (e.g. "hydrophobic residue
  following an acidic stretch," "small nonpolar residues in a
  hydrophobic environment"), not target-interface or binding-specific
  language.

## 2. Does any single feature represent binding? No — checked rigorously, not just eyeballed

Every one of the 16,384 features tested against the real wet-lab activity
data (all 18 real binders, including `b44` encoded fresh since it isn't
in the 80k pool), using the exact methodology from the PC7 investigation:
- Direction determined **only** on the 14-binder screen set, then
  **locked** and applied unchanged to the 4-binder confirm set — no
  independent re-optimization of direction on "held-out" data.
- Activity **control-ratio-normalized** (X = mCherry / own experiment's
  positive control) before any pooled comparison.
- **Benjamini-Hochberg FDR correction** applied given 16,384 candidates
  against only 18 real points — an even more severe multiple-comparisons
  regime than PC7's 4,320-candidate sweep.

**Result: 0 of 16,384 features reach BH-corrected significance (q<0.05).**
No feature can currently be said to represent binding with statistical
confidence — same conclusion as PC7's own final state.

**A real bug caught and fixed along the way**: max/mean-pooled SAE
activations are frequently exactly 0.0 for many designs at once (these
are sparse features — most simply don't fire on a given ~150-residue
protein). The concordance formula's tie-handling turned "never fires" into
a fake "perfect" score (0 concordant pairs out of 0 non-tied pairs →
`max(0, 1-0) = 1.0`), which is why the first run of this analysis showed
an absurd 9,963/16,384 features "confirmed." Fixed by tracking effective
(non-tied) pair counts and excluding degenerate features from ranking —
see `correlate_features_with_binding.py`'s docstring for the exact
mechanism.

### Best-available (unconfirmed) leads
None reach significance, but for reference, the strongest candidates by
raw screen concordance (with reasonable, non-degenerate effective-pair
counts) are almost all describing the same theme as §1's keyword
analysis — generic hydrophobic-packing motifs, e.g. feature 1157
("small/hydrophobic amino acids... at or near the beginning of the
sequence context") and feature 14792 ("hydrophobic residues... following
acidic residue-rich positions") both rank near the top in *both*
independently-computed max-pool and mean-pool versions — a mild, not
statistically validated, point of agreement. Full ranked tables:
`feature_analysis_results/layer_09/binding_correlation_{max,mean}pool.csv`.

## 3. What this means for generating better binders — the coherent, useful part

Even without a validated single "binding feature," there's a real,
**three-way convergent finding** worth acting on:

1. **Phase 1** (early in this project): ESMC's mean-pooled embeddings
   predict *intrinsic* sequence-quality metrics (hydrophobic patch size,
   developability/liability score — R²≈0.71–0.75) far better than
   *target-interface* quality metrics (best R²≈0.30).
2. **InterProScan** (this week): 94% of all domain hits across the
   corpus's SAE features trace back to one generic architecture — the
   Bcl-2 helical-bundle fold — suggesting the design corpus structurally
   converges on one common scaffold shape.
3. **SAE feature labels** (this review): the dictionary's dominant
   vocabulary is hydrophobic packing / small-residue / alanine-rich
   motifs, not binding-specific language, and the closest-to-correlating
   (still unvalidated) features describe exactly this same theme.

**Actionable read**: ESMC's representation of this corpus — at every
level examined (mean-pooled computational correlation, InterPro fold
matching, and now individual sparse features) — is dominated by generic
*intrinsic* sequence/fold quality, not binding specificity. If future
binder generation is guided by an ESMC-space objective (the diffusion-head
plan), it's worth being aware this will likely push toward "well-packed,
generic-helical-bundle-shaped" designs by default, not toward the
target-specific interactions that actually matter for binding — unless
the objective is specifically anchored to something that isolates the
binding-relevant component (which is exactly the unresolved role PC7/
layers 7-9 was meant to fill, per the ongoing direction investigation).

## Files
- `phase2_feature_extraction/sae/correlate_features_with_binding.py` —
  the correlation analysis.
- `phase2_feature_extraction/sae/encode_b44_layer9.py` — fresh b44
  encoding.
- `feature_analysis_results/layer_09/binding_correlation_{max,mean}pool.csv`
  — full per-feature results (gitignored, regenerate locally).
