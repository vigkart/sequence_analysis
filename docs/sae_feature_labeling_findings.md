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

**A more direct, intuitive version of the same result** (the same
observed-vs-chance framing that made the PC7 finding legible): restricting
to features with a *fully* informative confirm test (all 6 real pairs
between the 4 confirm binders, no ties/degenerate cases), and counting how
many hit a "perfect" locked-confirm score —

| Pooling | Fully-informative features | Hit "perfect" | Expected by pure chance | Ratio |
|---|---|---|---|---|
| max | 1,101 | 51 | 45.9 | 1.11x |
| mean | 1,101 | 38 | 45.9 | 0.83x |

Compare to PC7's original screen, which showed a real ~4x excess over
chance (8 of 25 vs. ~2 expected) — still misleading due to the sign-agnostic
bug, but at least a genuine statistical signal. Here there's no excess in
either direction — the count of apparently-perfect features is exactly
what undirected noise across 16,384 tries would produce. **This is a
clean, unambiguous null result, not just an absence of evidence.**

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

## 4. Could a COMBINATION of features (not one universal feature) predict binding?

Real, distinct hypothesis worth testing on its own: if binding is achieved
via several structurally different solutions across the binder set (binder
A via one motif, binder B via another), no *single* feature would show a
global correlation even if each subgroup's own feature is a real, strong,
causal driver within that subgroup — §2's univariate test structurally
cannot detect this.

**Aggregate "distinctiveness" test** (`correlate_distinctiveness_with_binding.py`):
does having *some* unusually rare/intense feature active, regardless of
which one, correlate with activity? Weak, inconclusive (screen concordance
0.54–0.66, well below even the top single features) — likely diluted by
the fact that ~20% of the entire 16,384-feature dictionary is "active
somewhere" in any given ~140-residue design, swamping whatever signal a
handful of truly special features might carry.

**Lasso multivariate probe** (`lasso_probe_binding.py`), the properly-powered
version — same approach iGEM's own prior pipeline used: `LassoCV` across
all 16,384 pooled features at once, capable of selecting a genuinely
different combination than any single-feature test could. Fit on the
14-screen only; the fitted model is then applied **unchanged** to the 4
confirm binders (no refitting), scored with a genuinely **signed** metric
(Pearson/Spearman on real predictions) — not sign-agnostic concordance,
directly applying the PC7 lesson that a sign-agnostic metric can mask a
backwards relationship.

| Pooling | Nested LOO-CV on the 14 (training) | Nonzero features | Held-out on the 4 (never fit) |
|---|---|---|---|
| max | R²=+0.170, ρ=+0.574 (p=0.032) | 6 | **Pearson r=-0.087 (p=0.913), R²=-3.623** |
| mean | R²=-0.964, ρ=-0.780 (p=0.001) | **0** (regularization zeroed everything) | undefined (constant prediction), R²=-0.274 |

Max-pool's training-set result looks significant on a superficial read
(p=0.032) — classic p≫n overfitting with 16,384 candidate predictors for
14 points, and the held-out test catches it: R²=-3.623 is worse than
predicting the same constant for every confirm binder. Mean-pool's own
internal cross-validation correctly declined to fit anything at all,
rather than grabbing onto noise the way max-pool did — arguably the more
trustworthy of the two outcomes precisely because it didn't overfit. The
6 features max-pool selected don't tell a coherent story either (cysteine,
a Y/H-before-PE motif, a very common 21%-density generic hydrophobic
feature, the Bcl-2-fold feature again, asparagine, methionine-in-alanine
context) — a grab-bag, consistent with fitting noise.

**Conclusion**: this doesn't refute the "diverse mechanisms" hypothesis
biologically — it's still plausible. It means 14/4 real data points isn't
enough to detect it, even with a method specifically designed for that
structure. The bottleneck is real-data quantity, not the analysis
approach.

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
