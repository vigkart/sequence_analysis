# PC7 sign-reversal investigation — findings for review

Written up overnight while `phase2_feature_extraction/sae/checkpoints/layer_09_run1`
trained, after a question about the SAE feature-selection plan surfaced a
real problem with the underlying PC7 finding. Not merged into `TODO.md`
since that file has other uncommitted edits in progress elsewhere — fold
this in wherever makes sense once you've reviewed it.

**Bottom line up front:** the direction of the PC7/layer-7-9 finding (which
end predicts good binding) does not replicate between the two real-data
experiments, and this is not noise — it's a statistically significant,
systematic reversal. The *existence* of a real signal at these layers still
looks likely, but not its *direction*, and until that's resolved, nothing
built on "high PC7 = good binder" (or the reverse) should be trusted,
including the planned SAE best/worst-binder feature selection.

## How this was found

Reviewing the plan to select SAE features by which fire strongest on
PC7-extreme "best" vs. "worst" designs, I needed to know which end of PC7
is good. Checking this directly against the real 18-binder data (rather
than assuming a direction) surfaced two separate problems.

## Bug #1 (fixed): PC7 was the wrong column

`phase2_feature_extraction/sae/extract_binder_residues.py` had
`PC7_INDEX = 6`. `screen_then_confirm.py` prints `PC{pc}` using the raw
0-indexed loop variable, not 1-indexed — so "PC7" everywhere in this
project's docs actually means column **7**, not 6. Verified empirically:
column 7 reproduces the documented `weighted_concordance=0.904` (joint,
layer 9, covariance_all_dims) on the 14-binder screen; column 6 gives an
unremarkable 0.626.

**Fixed** in commit `59d4a06`. Doesn't invalidate the currently-training
layer 9 SAE (PC7 was only used to stratify train/val/test for
representativeness — any reasonable decile-stratification still gives a
valid split for training purposes) — not worth regenerating mid-training.
Layers 7/8, not yet extracted for training, will get the corrected
stratification.

## Bug #2 (found, not yet resolved — this is the important one): sign-agnostic confirmation

`screen_then_confirm.py`'s confirmation step scores the held-out "4" (the
previous-experiment binders) with `concordance_plain`, which — like the
screening step's `weighted_concordance` — is `max(frac, 1-frac)`:
**sign-agnostic**. No model is fit on the 4 (that part of the methodology
is correctly held out) — but the confirmation metric independently
re-optimizes *which of the two possible directions* scores best on the 4,
rather than testing whether the specific direction found during screening
on the 14 also holds. That's a one-bit form of fitting to "held-out" data,
hidden inside the metric rather than in the model.

Same issue exists in `linear_probe_per_layer.py`'s `concordance()` — same
`max(frac, 1-frac)` construction. There, though, we get a cleaner signal
because it also reports `reg_pearson`, a genuinely signed correlation
between the model's actual (never-refit) predictions and the true labels:

| Layer (joint) | "Perfect" concordance (sign-agnostic) | Actual signed correlation (`reg_pearson`) |
|---|---|---|
| 7 | 1.000 | **-0.965** |
| 8 | 1.000 | **-0.992** |
| 9 | 0.833 | **-0.973** |

The probe's honestly-computed, never-refit predictions are almost
perfectly *backwards* relative to truth on the 4 — not weak, not
noise-level, a near-perfect inverse, consistently across all three layers.

## The fix and what it shows

`phase1_layer_selection/screen_then_confirm_signlocked.py` (new script,
committed): locks the direction found during screening on the 14, scores
the 4 using that fixed direction, no re-optimization.

**Result: 0 of the top 50 candidates from the original screen genuinely
confirm.** Not just "doesn't confirm" — the pattern is systematically
skewed toward reversal, not scattered randomly around chance: 12/50 hit
the single worst possible score (all 6 confirm pairs backwards), far more
than the ~2/50 you'd expect if direction generalized (or failed to
generalize) at random.

## Is the reversal itself real, or an artifact of correlated candidates?

The top 50 candidates aren't independent trials — many are the same
underlying structure viewed through highly correlated layers/PCA variants.
A systematic-looking pattern could in principle emerge from that
correlation alone, without a real batch effect. Tested this directly.

**Exact permutation test** (`phase1_layer_selection/utilities/signlock_permutation_test.py`,
following the same exact-permutation methodology this project already used
in `permutation_test.py`): for all 4! = 24 possible ways to assign the 4
real labels to the 4 embedded points, recomputed the aggregate
reversal statistic across all 50 candidates (screening-derived directions
held fixed, since screening never touches the 4). Compared the TRUE
labeling's aggregate score against the full distribution over 24
permutations.

**The true labeling is the single most-reversed of all 24 possible
permutations** — lowest mean locked-concordance (0.263, next-closest
0.290) and highest reversal count (12, next-closest 10) of any relabeling.
Exact p = 1/24 = 0.042.

This is real evidence *against* "just correlated-candidate noise" — pure
chance wouldn't reliably land the true labels at the most extreme possible
rank out of 24.

## Ruled out: length confound

Checked directly since this is exactly the failure mode that killed the
earlier layer-14 lead. Binder length is essentially uncorrelated with the
mCherry labels when pooled across both experiments (Spearman ρ=0.020,
p=0.938), and the two experiments' binders are similar in length (mean
140.6 vs. 138.5 residues). Not the explanation this time.

## Ruled out: one-off embedding script artifact

One of the 4 confirm-set binders (`b44`, "Binder3-4") was embedded via a
separate one-off script (`embed_binder3_4.py`), not the main pipeline the
other 3 (and all 14 screen-set binders) went through — raising the
possibility of a code-path-specific technical artifact. Checked by
dropping `b44` and re-running the locked-direction scoring on just the
remaining 3 points (which used the identical pipeline as the 14). **The
reversal doesn't just persist — it strengthens**: 21/50 candidates hit
complete reversal (vs. 12/50 with `b44` included). Not a `b44`-specific
bug.

## What's left: leading hypothesis, unconfirmed

With length and the one-off-script explanation ruled out, and the
statistical evidence pointing at a real (not noise) reversal, the most
plausible remaining explanation is a **batch effect between the two
wet-lab experiments** — different assay date, calibration, or
normalization between the "previous" and "new" rounds of testing — that a
real ESMC-encoded signal interacts with differently in the two batches.
Both real-data sources are documented elsewhere as the same no-TET
co-transfection condition against the same site, so it isn't that known
condition difference; it would have to be something else about how the
two rounds were run. **This is a hypothesis, not a finding** — I don't
have wet-lab context (assay dates, protocols, calibration records) to
check it further; that's a question for you, not something computable
from this repo's data.

## What this does and doesn't affect

- **Does not affect the currently-training layer 9 SAE.** It's
  unsupervised reconstruction — never touches PC7 direction. Safe to let
  finish and to train layers 7/8 the same way.
- **Does affect** anything downstream that assumed a validated direction:
  the planned "best/worst PC7 binders" SAE feature selection, and TODO's
  step 5 ("cross-reference SAE features against PC7 to explain why it
  separates good from bad binders"). Both are on hold pending this.
- **Does not mean PC7/layers 7-9 have zero relationship to binding** — the
  rank-separation power (concordance meaningfully above chance) and the
  layer localization (7-9 keep surfacing via multiple independent methods)
  both still look real. What's unresolved is direction, not existence.

## Suggested next steps (your call)

1. Check with whoever ran the two experiments whether there's a known
   calibration/normalization difference between the "previous" and "new"
   mCherry rounds that could explain a real signal flipping sign.
2. Once resolved (or if it can't be), decide how to proceed with
   the best/worst-binder feature selection — possible options include:
   waiting for more real data to triangulate a stable direction, using
   within-batch-only comparisons (never mixing the two experiments'
   labels for a single directional claim), or dropping the
   binary-direction framing and selecting features by extreme-vs-extreme
   PC7 groups without labeling either side "best."
3. Worth deciding whether this changes how much weight the layer 7-9
   choice itself should carry going forward, independent of the SAE work.

## Files added this session (all on `sae-training`, pushed)

- `phase1_layer_selection/screen_then_confirm_signlocked.py` — the
  corrected re-analysis.
- `phase1_layer_selection/utilities/signlock_permutation_test.py` — the
  exact permutation test.
- PC7 index fix in `phase2_feature_extraction/sae/extract_binder_residues.py`
  (commit `59d4a06`).
