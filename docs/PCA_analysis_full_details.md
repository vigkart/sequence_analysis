# PCA / layer-selection analysis — full details with metrics

Complete reference for the slideshow, covering the whole arc: methodology,
the original finding, the rigor-testing that followed, and the honest
final state. Supersedes/expands §2–3 of `PI_presentation_outline_v2.md`,
which predates the final sweep and wet-lab corroboration below.

---

## 1. Methodology

- **Embedding**: all 80,000 BoltzGen designs, embedded with ESM-C 600M at
  all 36 layers, in two conditions — binder alone, and binder+target
  concatenated via ESM-C's native chain-break token ("joint"). Verified in
  source that joint condition gives genuine bidirectional attention
  between binder and target.
- **PCA**: fit per layer × per condition × 3 preprocessing variants
  (`correlation_all_dims`, `covariance_all_dims`, `covariance_outliers_excluded`
  — the last excludes a small set of recurring "massive activation"
  outlier dimensions found during a separate variance audit). Top 50
  components kept per fit.
- **Real wet-lab activity data** (demethylation reporter, % mCherry,
  no-TET-co-transfection condition — the condition actually being
  optimized for):
  - **"The 14" (screen set)**: from a 15-binder plate (one design
    excluded — see Data Quality below), used to *search* for candidate
    axes.
  - **"The 4" (confirm set)**: p130, p144, p154_prev, b44 — from a
    separate, earlier 3(+1)-binder experiment — used to *test* whatever
    the screen found, held out from the search entirely.

---

## 2. Original screen-then-confirm result (as first reported)

- Screened **4,320 candidates** (36 layers × 3 variants × 20 PCs × 2
  conditions) against the 14, scored by **weighted pairwise concordance**:
  every pair of binders is checked for whether the PCA axis and the true
  activity agree on which is bigger; pairs get weight 9 (both binders
  ≥4% activity), 3 (one ≥4%), or 1 (neither) — upweighting comparisons
  among binders that show meaningful activity.
- **Top result**: layer 9, joint condition, `covariance_all_dims`, PC7 —
  **weighted_concordance = 0.904**, AUC(≥4% threshold) = 1.000.
- **Confirmation** (as originally computed): checked the top 25 screened
  candidates against "the 4" — **8 of 25 achieved a "perfect" 6/6**
  pairwise match.
- This, plus an independently-run **regularized multivariate probe**
  (RidgeCV regression + L2 LogisticRegressionCV classification on the
  top-10 PCs, trained only on the 14, evaluated without refitting on the
  4) landing on the same region — layers 7–8 joint, with reported
  "perfect" (1.000) concordance metrics — was the original basis for
  identifying layers 7–9 as the binding-relevant region.

### PC7's characterization (still stands, unaffected by everything below)
- Explains only **~2% of variance** at layer 9 (rank 8 of 50 components)
  — a narrow, specific sub-signal, not a dominant bulk-variance direction.
- **Continuous, unimodal distribution** across all 80,000 designs — no
  discrete clusters (checked via HDBSCAN on the top-20 PCs across layers
  7–9). Consistent with "binding" being a smooth gradient property, not
  something that partitions designs into groups.
- **Joint condition > binder-alone specifically for this real-data
  signal** — even though the two conditions are statistically
  indistinguishable on every *computational* metric (iptm, ipsae, etc.).

---

## 3. Rigor-testing the result: two bugs found

Triggered by needing to pin down PC7's *direction* (high vs. low = good
binding) for downstream feature-selection work — a question the original
metrics never had to answer, since they were direction-agnostic by
construction.

### Bug 1 (fixed, minor): off-by-one PC indexing
`screen_then_confirm.py` prints `PC{pc}` using the raw 0-indexed loop
variable — so "PC7" throughout this project means column 7, not 6.
Verified: column 7 reproduces the documented 0.904; column 6 gives an
unremarkable 0.626. A downstream script had this wrong; fixed, did not
affect anything already trained.

### Bug 2 (the important one): sign-agnostic confirmation
Both the screening metric (`weighted_concordance`) and the confirmation
metric (`concordance_plain`) are `max(frac, 1-frac)` — **sign-agnostic**.
No model was fit on "the 4" (that part of the methodology is sound), but
the confirmation step independently re-picks whichever of the two
possible directions scores best on the 4, rather than testing whether the
specific direction found during screening replicates. A one-bit form of
fitting to "held-out" data, hidden inside the evaluation metric rather
than the model.

**Mathematical consequence**: with n=4 (6 pairs, no ties), `max(frac,
1-frac) == 1.0` happens for **2 of the 24** possible label orderings (the
true order, or its exact reverse) — so the correct chance-probability for
a single candidate hitting "6/6" under this metric is **2/24 = 1/12 ≈
0.083**, not the 1/24 ≈ 0.042 informally assumed at the time — already
below the conventional significance bar on its own, before any empirical
re-check.

### Corrected (sign-locked) re-analysis
`screen_then_confirm_signlocked.py`: lock the direction found during
screening on the 14, score the 4 with that fixed direction, no
re-optimization.
- **Result: 0 of the top 50 candidates genuinely confirm.**
- Not partial disagreement — **all 8 of the original "confirmed" hits
  reverse completely** (locked score = 0.000, i.e., 0/6) with zero
  exceptions, including the top result (layer 9, PC7, joint).

### Is the reversal itself real, or an artifact of correlated candidates?
The top 50 candidates aren't independent (many are the same underlying
structure viewed through correlated layers/variants). Tested directly via
**exact permutation test** (`signlock_permutation_test.py`, following this
project's own established n=4 / 4!=24-permutation precedent): for all 24
possible ways to assign the 4 real labels to the 4 embedded points,
recomputed the aggregate reversal statistic across all 50 candidates.
- **The true label assignment is the single most-reversed of all 24
  possible relabelings** — lowest mean locked-concordance (0.263) and
  highest reversal count (12/50) of any relabeling.
- **Exact p = 1/24 = 0.042.**
- This is evidence *against* "just noise from correlated candidates" —
  pure chance wouldn't reliably land the true labels at the single most
  extreme rank out of 24.

---

## 4. Root-cause investigation — what was ruled out, what wasn't

| Candidate explanation | Test | Result |
|---|---|---|
| Length confound (same failure mode as the earlier layer-14 story) | Spearman ρ, length vs. activity, pooled n=18 | ρ=0.020, p=0.938 — not it. Batches similar length (140.6 vs 138.5 mean residues) |
| One-off embedding-script artifact (b44, embedded via a separate script) | Re-ran locked-direction scoring excluding b44 | Reversal **persists and strengthens** (21/50 full reversals vs. 12/50 with it) — not it |
| Codebase reorganization / stale data | Checked `pca.mean_` against current raw embeddings' actual mean | **Exact match, 0.00000000 difference** to full float precision — not it |
| Weighting scheme (9x/3x/1x pair weights) | Re-ran screen direction unweighted | Negative direction wins either way (0.904 weighted vs. 0.769 unweighted) — not it |
| Contamination-suspected new157 reading (9.19% vs. replicate's 4.99%) | Re-ran screen direction with both values | Negative direction wins either way (0.904 vs. 0.863 with corrected value) — not it |
| **Batch-scale calibration difference between the two experiments** | Checked the two experiments' own positive controls | **72.8% (previous) vs. 36.56% (new) = 1.99x ratio** — closely matches the 2.28x ratio independently estimated from the tested binders' own batch means. Independent, wet-lab-sourced corroboration, not just a computational inference |

**Caveat on the batch-scale explanation**: the new experiment's own
positive control well was itself flagged by the researcher who ran the
assay for having too few cells — so this isn't fully closed, just the
best-supported explanation found so far.

---

## 5. Pooled/normalized analysis — what it shows, and its real limits

Pooling all 18 real binders into one sample and correlating PC7 against
(normalized) activity — a fundamentally different, weaker question than
the held-out screen-then-confirm test (this discards the train/test
separation entirely):

| Normalization | Pooled Spearman ρ (n=18) |
|---|---|
| Raw, unnormalized | -0.193 |
| Batch-mean-corrected | -0.226 |
| Control-ratio-normalized (X = mCherry / own experiment's positive control) | **-0.321** |

All three are negative (matching the screen's direction), and the
properly wet-lab-motivated normalization is the strongest of the three —
but **none reach significance at this n** (p=0.194 for the best one).
This is a diagnostic consistent with the screen's direction, not proof of
it.

### The final honest check: does PC7 even win this alternative ranking?
Swept the same 4,320 candidates, ranked by |pooled control-normalized ρ|
instead of the original screen-then-confirm metric:
- **Layer 9/PC7/joint ranks 585th of 4,320** by this metric (ρ=-0.321,
  p=0.194 — not significant).
- The top candidate by this metric is a *different* axis entirely (layer
  2, PC14, joint: ρ=-0.851, p<0.001).

**Critical interpretation — do not present the new "winner" as the real
answer either.** This sweep has the identical structural weakness as the
original: 4,320 correlated candidates tested against only 18 real points.
A correlation this strong from *some* candidate out of 4,320 tries is not
surprising under pure noise. **Two different, reasonable scoring methods
picked completely different "best" candidates out of the same 4,320 —
that disagreement is itself the evidence that neither single-PC search
should be trusted to name one specific "best" axis at this sample size.**

---

## 6. Bottom line — what's actually defensible

- **Layer localization (7–9 region) remains the best-supported claim** —
  not because any single-PC search "won," but because **two structurally
  different methods agreed**: the original PCA single-axis screen, and
  separately a regularized multivariate probe using 10 PCs simultaneously
  (RidgeCV/LogisticRegressionCV), which is far less prone to
  cherry-picking one lucky axis out of a huge search space than either of
  the single-PC sweeps above.
- **The exact "best" PC and its direction are open questions**, not
  settled ones. There's a plausible, partially wet-lab-corroborated
  explanation for why the original confirmation test failed (batch-scale
  shift) — but that explanation has not been proven, and no specific PC
  (including PC7) should be presented as "the" validated axis.
- **What would actually resolve this**: either a genuine third,
  independent real dataset for proper held-out validation (ideally
  calibrated consistently with an existing batch), or moving past
  single-PC-axis analysis toward methods that use many dimensions at once
  (the multivariate probe, or the SAE feature work already in progress) —
  neither of which suffers from this specific fragility.

---

## Suggested slide framing

1. **Methodology slide**: search space (4,320 candidates), weighted
   concordance metric, screen/confirm split rationale.
2. **Original finding slide**: 0.904 weighted concordance, layer 9/PC7,
   corroborated by the independent multivariate probe — present as the
   original result.
3. **Rigor slide**: the sign-agnostic bug, corrected re-analysis (0/50
   confirm, p=0.042 for the reversal being real), root-cause table from
   §4 — frame as the same caliber of self-correction as the earlier
   layer-14 debunking.
4. **Honest current-state slide**: pooled/normalized evidence table,
   the 585th-of-4,320 finding, and the explicit "two methods disagree
   completely, neither should be trusted alone" framing — this is a
   *credibility* strength to present plainly, not a result to soften.
5. **Bottom line slide**: layer localization solid (multi-method
   convergence); exact axis/direction open; path forward is either new
   real data or moving to multivariate/SAE-based methods.
