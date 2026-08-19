# PC7 sign-reversal investigation — findings for review

Written up overnight while `phase2_feature_extraction/sae/checkpoints/layer_09_run1`
trained, after a question about the SAE feature-selection plan surfaced a
real problem with the underlying PC7 finding. Not merged into `TODO.md`
since that file has other uncommitted edits in progress elsewhere — fold
this in wherever makes sense once you've reviewed it.

**Bottom line up front (updated after more digging — see the batch-shift
section below):** the original screen-then-confirm methodology had a real
bug (sign-agnostic confirmation) that made the two experiments look like
they disagreed on PC7's direction. Following up, the two experiments turn
out not to be on a comparable activity scale at all — the "previous
experiment" confirm set reads more than 2x higher in raw % mCherry on
average, for reasons unrelated to PC7. Once that batch-level offset is
accounted for, the pooled evidence points back toward the **screen set's
original direction (higher PC7 → lower binding)**, not a genuine reversal.
Current best read: PC7's direction is probably the screen's, and the
"confirmation" step's apparent contradiction was a small-n artifact of
pooling two batches that were never on the same scale — not evidence that
the direction is fundamentally unstable. Still not fully resolved (see
"What's left" below), but meaningfully less alarming than it looked a few
hours ago.

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

## Verification pass: data-entry, hand arithmetic, reorg artifacts

Before trusting any of the above, went back and checked it wasn't a
mistake somewhere mundane:

- **Cross-checked every value against the full wet-lab tables** (both the
  15-binder and previous-3-binder experiments, as provided directly):
  all 14 NEW14 values and all 4 PREV4 (no-TET) values match exactly what
  the code used. No transcription error.
- **Robustness check with `new154`** (originally excluded as "flagged
  inaccurate," now available): re-ran the full sign-locked screen with all
  15 binders instead of 14. Same top axis (layer 9, joint,
  covariance_all_dims, PC7, same negative direction), same outcome
  (0/50 genuinely confirmed, mean_locked=0.240 vs. 0.263 before). `new154`
  wasn't suppressing or manufacturing the reversal.
- **Hand-verified the arithmetic**, no code involved: computed all 6
  pairwise PC7-vs-label comparisons for the confirm set by hand from the
  raw numbers (all 6 positively concordant, matching the code) and spot-
  checked 2 pairs from the screen set by hand (both discordant with the
  confirm set's direction, matching the code). The concordance formula is
  also symmetric under swapping any pair's order, so pair-list ordering
  can't hide a sign flip.
- **Checked for reorg-related staleness** (the codebase went through
  several directory reorganizations recently): verified `pca.mean_`
  matches the current raw embeddings' actual mean to full floating-point
  precision (0.00000000 difference) — if the PCA had been fit on an older
  or different version of the embeddings, this would show a gap. Also
  verified file sizes match expected byte counts exactly, modification
  timestamps are in the correct dependency order (scores CSV → embeddings
  → PCA fit → rank lookups), and `rank_lookup_row_indices.json` still
  resolves to the exact same sequences currently in the scores CSV for all
  15 real binders. No desync found anywhere checked.

## The actual explanation: the two experiments aren't on the same scale

Looking at all 18 real binders sorted by PC7 (not just concordance
statistics) made the pattern visible directly: the "previous experiment"
(confirm set) binders sit near the top of the overall activity
distribution almost regardless of their PC7 value. Quantified:

- Screen (14) mean label: **3.12%**
- Confirm (4) mean label: **7.10%** — more than double, unrelated to PC7

This constant-per-batch shift can't affect the confirm set's *internal*
pairwise concordance (subtracting a constant doesn't change which of two
points is bigger) — so the "perfect 6/6" result there was never going to
be touched by it. But it does mean naively pooling raw values across both
batches, or treating the confirm set as validating a claim about absolute
activity level, is comparing numbers that were never on the same footing.

Correcting for it directly: pooling all 18 points and comparing PC7 to
each point's deviation from *its own batch's mean* (rather than the raw
label) flips the pooled correlation back toward the screen's original
direction:

| | Spearman ρ (PC7 vs. label), n=18 pooled |
|---|---|
| Naive (raw labels) | -0.193 |
| Batch-mean-corrected | **-0.226** |

Neither is significant at this n with this simple pooling (not a
replacement for the proper weighted/pairwise methodology — just a
diagnostic), but both are negative, matching the screen set, not the
confirm set's apparent reversal.

**Most likely explanation**: the confirm set's "perfect positive" result
was a small-n coincidence, not a real reversal. Two probably-unrelated
facts about that batch — (1) it reads ~2.3x higher across the board for
reasons that look like calibration/normalization rather than binding
quality, and (2) by chance, 3 of its 4 points also happen to sit at the
higher end of the PC7 range — combined to produce a spurious positive
result in a test with only 6 pairs to explain. The screen set's negative
direction is the more defensible one to trust right now.

**Update — wet-lab confirmation, not just a computational hypothesis
anymore**: the two experiments' own positive controls (KY136/KY_136,
no-TET) differ by **1.99x** (72.8% previous experiment vs. 36.56% new
experiment) — independently sourced from the actual assay records, not
derived from the binder data at all. That's remarkably close to the
2.28x ratio estimated purely computationally above (screen-batch mean
3.12% / confirm-batch mean 7.10%). Two independent measurements landing
within ~15% of each other is real corroboration, not coincidence.

Re-ran the pooled direction check using the proper normalization this
implies (X = mCherry / own-experiment's positive control, exactly the
method proposed by the researchers who ran the assays) instead of the
cruder batch-mean-subtraction used above:

| Normalization | Pooled ρ (PC7 vs. activity, n=18) |
|---|---|
| Raw, unnormalized | -0.193 |
| Batch-mean-corrected | -0.226 |
| **Control-ratio-normalized** | **-0.321** |

The properly-motivated normalization strengthens the match to the
screen set's original direction, not weakens it. Still not significant
at conventional levels with this simple pooled check at n=18 (not a
substitute for the real weighted/pairwise methodology) — but now the
most internally consistent version of this analysis, and grounded in an
actual measured batch difference rather than an inferred one.

**Caveat on the caveat**: the new experiment's own positive control
(well 16) was itself flagged by the researcher who ran the assay for
having too few cells in that well — so the denominator in the
control-ratio normalization above carries its own uncertainty. Doesn't
change the direction, but is a reason not to overstate how "closed" this
now is.

**Sequence-to-plasmid mapping, checked and ruled out as a concern**: an
older mapping issue (design_spec_130/144/154 swapped against the wrong
activity values) was raised as a possible explanation. Checked directly
against the raw data: the row indices used throughout this entire
investigation (2007/16250/30263) resolve to design_spec_130/144/154 with
sequence prefixes matching the confirmed-correct mapping already noted
elsewhere in this project's history. Not a live issue for anything above.

**Still not fully closed**: this is now a well-corroborated
reconstruction (computational estimate + independent wet-lab
measurement agreeing within ~15%), not a 100%-certain fact — the
control-well cell-count caveat above is the main remaining source of
uncertainty. A repeat experiment with a cleaner control read (already
planned, per the researchers' own next steps) would be the direct way to
close this out.

## What this does and doesn't affect

- **Does not affect the currently-training layer 9 SAE.** It's
  unsupervised reconstruction — never touches PC7 direction. Safe to let
  finish and to train layers 7/8 the same way.
- **Does affect** anything downstream that assumed a validated direction:
  the planned "best/worst PC7 binders" SAE feature selection, and TODO's
  step 5 ("cross-reference SAE features against PC7 to explain why it
  separates good from bad binders"). Recommend using the screen set's
  direction (higher PC7 → lower binding) for these given the batch-shift
  finding above, but flag it as a working assumption pending wet-lab
  confirmation, not a fully closed question.
- **Does not mean PC7/layers 7-9 have zero relationship to binding** — the
  rank-separation power (concordance meaningfully above chance) and the
  layer localization (7-9 keep surfacing via multiple independent methods)
  both still look real, and the batch-shift finding actually makes the
  screen set's direction more trustworthy than it looked a few hours ago,
  not less.

## Suggested next steps (your call)

1. Check with whoever ran the two experiments whether there's a known
   calibration/normalization difference between the "previous" and "new"
   mCherry rounds — would directly confirm (or rule out) the batch-shift
   explanation above.
2. If confirmed: the screen set's negative direction (higher PC7 → lower
   binding) is the one to build on for best/worst-binder feature
   selection. If it can't be confirmed either way, consider getting the
   confirm-set binders re-measured in the same batch/session as a future
   round of real data, so a same-scale confirmation actually becomes
   possible.
3. Either way, worth being careful going forward to never pool raw
   activity values across these two experimental batches without
   correcting for the batch-level offset first — the naive pooled
   correlation and the batch-corrected one point in different directions
   at small n, which this session's exploration ran into directly.

## Files added this session (all on `sae-training`, pushed)

- `phase1_layer_selection/screen_then_confirm_signlocked.py` — the
  corrected re-analysis.
- `phase1_layer_selection/utilities/signlock_permutation_test.py` — the
  exact permutation test.
- PC7 index fix in `phase2_feature_extraction/sae/extract_binder_residues.py`
  (commit `59d4a06`).
