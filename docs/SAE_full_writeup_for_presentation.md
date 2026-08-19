# Sparse Autoencoder (SAE) interpretability — full write-up for presentation

Covers everything from the SAE work's start through tonight's binder-level
motif investigation. Draws on `docs/sae_feature_labeling_findings.md` and
the underlying scripts in `phase2_feature_extraction/sae/` for full detail
and reproducibility.

---

## 1. Motivation

PC7 (layers 7-9) tells us a *direction* in ESMC's embedding space
correlates with binding — it says nothing about *what it represents*.
Two reasons that gap matters:
- **Scientific**: want to know what's mechanistically driving binding,
  not just that something is.
- **Practical**: "maximize PC7" is a narrow, single-axis proxy for any
  future generative/optimization step — worth knowing whether it's
  genuinely causal or entangled with something else before committing a
  generative model to that objective.

**Approach**: train a Sparse Autoencoder (SAE) on ESMC's residue-level
activations, decomposing the dense 1152-dim representation into a much
larger (16,384), *sparse* dictionary of features — each ideally more
monosemantic and interpretable than a raw dimension or a PCA axis.

---

## 2. Architecture

Adapted from a validated prior design (the lab's own SENTINEL/iGEM
interpretability work), not built from scratch:
- **Per-token TopK sparsity** (k=64 active features per residue, out of
  16,384) — matches the established recipe from Gao et al. 2024
  ("Scaling and evaluating sparse autoencoders").
- **AuxK dead-feature revival**: auxiliary loss that gives inactive
  features a path back to usefulness during training.
- **Fixed dataset-level centering + scaling** (not a learned/dynamic
  bias) — specifically avoids a known failure mode where outlier
  "massive activation" dimensions (already documented in this project's
  own earlier variance audit) cause permanent feature death when a
  learned bias has to absorb them.
- **Decoder unit-norm constraint**, enforced after every optimizer step
  — prevents the trivial trick of shrinking codes and growing decoder
  norm to fake low reconstruction loss.
- **Cosine LR decay after warmup** (adapted from the original recipe,
  which held LR constant after warmup for the whole run — decaying
  gives tighter final convergence).
- Deliberately **plain Adam, no weight decay**: this SAE's weight
  magnitude is already controlled by more targeted mechanisms (the
  decoder norm constraint, and the reconstruction loss's own implicit
  calibration pressure on the encoder) — added decay would work against
  the AuxK/k-annealing machinery already fighting dead-feature death.

---

## 3. Data

- **Residue-level** (not mean-pooled) ESMC activations at layers 7, 8, 9,
  **joint condition** (binder+target), all 80,000 designs — ~175GB,
  generated and integrity-verified in an earlier session.
- **Binder-only residues extracted for SAE training** (target excluded
  from training, though it's still attended to during the original
  forward pass — the target's presence affects the binder's own
  representation, which is exactly what "joint condition" is for).
- **Split**: all 80,000 designs get a split label, stratified by PC7
  decile so every split spans the full binding-relevant score range —
  20% (16,000) held out entirely as test, the remaining 80% split again
  90/10 into train (57,600) / val (6,400). Only train+val are loaded
  into RAM during training (~18GB), keeping the run tractable on local
  hardware; test is loaded separately only for final evaluation.

---

## 4. Training result — layer 9 (first of the 3 target layers)

- 50 epochs, ~4h46min on local Apple Silicon (MPS).
- **Reconstruction quality (fraction of variance explained, FVE):
  0.9017 on validation, and identically 0.9017 on a fully independent
  held-out test set** (~2M residues never touched during training or
  checkpoint selection) — no overfitting gap.
- **0 dead features out of 16,384** — every single dictionary feature is
  alive and firing. A fully healthy, fully-utilized dictionary.
- Layers 7 and 8 not yet trained (same pipeline, queued next).

---

## 5. Feature interpretation pipeline

1. **Full-pool mining**: for every one of the 16,384 features, found its
   ~15 hardest-firing residues across the *entire* 80,000-design pool
   (not a sample) — density (how often it fires) and its top
   max-activating examples with local sequence context.
2. **InterProScan candidate selection**: a bounded set of 95 features
   selected via two criteria — features that fire on the real,
   wet-lab-tested binders specifically (65), and density extremes
   (30: 15 most generic + 15 rarest-but-alive) — matching the same
   practical recipe validated in the lab's prior interpretability work.
   Run against ~1,100 unique associated proteins (scoped down from what
   would otherwise be tens of thousands of EBI API jobs for the full
   dictionary — a deliberate scope decision given InterProScan's
   fair-use limits as a shared free service).
   - **41% hit rate** (583/1,425 example residues) — notably high for a
     100% de novo, computationally-designed corpus with zero natural
     evolutionary history.
   - **94% of all hits (549/583) are one fold family: Bcl-2-like**
     (`Bcl-2-like_sf`, `Bcl-2_BH1-3`, `Bcl2-like`, `Bcl-2_fam`) — a
     classic all-helical-bundle architecture. Very unlikely to reflect
     real evolutionary relationship to Bcl-2; far more likely that this
     design corpus structurally converges on a generic helical-bundle
     scaffold that a broad, well-populated InterPro profile happens to
     catch.
3. **LLM labeling** (Claude Haiku via Batches API): every one of the
   16,384 features labeled from its examples (+ InterPro evidence where
   available). **16,377/16,384 (99.96%) got a real description**; only 7
   came back "No clear pattern." Dominant vocabulary across all labels:
   `hydrophobic` (11,498 mentions), `alanine` (7,630), `small` (4,475),
   `acidic` (3,728), `leucine` (3,522) — overwhelmingly local
   amino-acid-composition/packing motifs, not target-interface language.

**This is a coherent, three-way-convergent finding**, not three unrelated
observations:
1. Phase 1 (early in the project): ESMC's mean-pooled embeddings predict
   *intrinsic* sequence quality (hydrophobic patch, developability) far
   better than *target-interface* quality (R²≈0.71-0.75 vs. ≈0.30).
2. InterProScan: 94% of hits trace to one generic fold.
3. SAE feature labels: dominant vocabulary is packing/composition, not
   binding-specific language.

**Actionable read**: ESMC's representation of this corpus is dominated by
generic intrinsic sequence/fold quality at every level examined. A future
ESMC-space generative objective should be aware this will likely default
toward "well-packed, generic-helical-bundle-shaped" designs, not toward
target-specific binding, unless anchored to something that isolates the
binding-relevant component specifically.

---

## 6. Does any SAE feature represent binding? Tested rigorously — no

Correlated every one of the 16,384 features' pooled activation (max-pool
and mean-pool, computed independently) against the real wet-lab activity
data for all 18 real binders (including `b44`, encoded fresh since it
isn't in the 80k pool), using the **same rigor established for the PC7
investigation**:
- Direction determined **only** on the 14-binder screen set, then
  **locked** and applied unchanged to the 4-binder confirm set — no
  sign-agnostic re-optimization.
- Activity **control-ratio-normalized** before any pooled comparison.
- **Benjamini-Hochberg FDR correction** given 16,384 candidates against
  18 real points.

**A bug caught along the way**: SAE features are sparse (exactly 0.0 for
most designs), so the concordance formula's tie-handling initially
produced a fake "perfect" score for inactive features (0 concordant
pairs out of 0 non-tied pairs → `max(0,1)=1.0`) — the first run showed an
absurd 9,963/16,384 features "confirmed." Fixed by tracking effective
(non-tied) pair counts and excluding degenerate features.

**Result, most legible version**: restricting to features with a fully
informative confirm test (all 6 real pairs among the 4 confirm binders,
no ties), the count hitting a "perfect" locked score is statistically
indistinguishable from pure chance —

| Pooling | Fully-informative features | Hit "perfect" | Expected by chance | Ratio |
|---|---|---|---|---|
| max | 1,101 | 51 | 45.9 | 1.11x |
| mean | 1,101 | 38 | 45.9 | 0.83x |

Compare to PC7's original screen, which showed a real ~4x excess over
chance (later shown misleading, but at least a genuine statistical
signal). Here there's no excess in either direction. **A clean,
unambiguous null result, not just absence of evidence.**

### Could a combination of features (not one universal feature) predict binding?

Tested the hypothesis that binding might be achieved via multiple
structurally distinct solutions (different binders using different
motifs), which a single-feature test structurally cannot detect:

- **Aggregate "distinctiveness" test**: does having *some* rare/intense
  feature active, regardless of which one, correlate with activity?
  Weak, inconclusive (screen concordance 0.54-0.66) — likely diluted by
  ~20% of the entire dictionary being "active somewhere" in any given
  design.
- **Lasso multivariate probe** (`LassoCV` across all 16,384 pooled
  features, same approach the lab's prior pipeline used): fit on the
  14-screen with nested leave-one-out CV, then applied **unchanged** to
  the 4-confirm (no refitting), scored with a genuinely **signed**
  metric (Pearson/Spearman on real predictions) — directly applying the
  lesson from the PC7 investigation, where a sign-agnostic metric masked
  a backwards relationship.

| Pooling | Nested LOO-CV on the 14 (training) | Nonzero features selected | Held-out on the 4 (never fit) |
|---|---|---|---|
| max | R²=+0.170, ρ=+0.574 (p=0.032) | 6 | **Pearson r=-0.087 (p=0.913), R²=-3.623** |
| mean | R²=-0.964, ρ=-0.780 (p=0.001) | **0** (regularization zeroed everything) | undefined (constant prediction) |

Max-pool's apparent training-set significance is classic p≫n
overfitting (16,384 candidates for 14 points) — the held-out test
catches it decisively (R²=-3.623 is worse than predicting a constant for
every confirm binder). Mean-pool's own internal CV correctly declined to
fit anything, rather than grabbing onto noise.

**Conclusion**: doesn't refute the diverse-mechanisms hypothesis
biologically — still plausible — but 14/4 real data points isn't enough
to detect it, even with a method built for exactly that structure. The
bottleneck is real-data quantity, not analysis approach.

---

## 7. Qualitative deep-dive: what's distinctive about our actual strongest binders

Different question from §6 — not "does this generalize statistically,"
but "what does this specific successful design's SAE encoding look
like." For our 3 strongest binders from the 14-experiment (`new157`
9.19%, `new089` 5.12%, `new241` 4.92%) and the strongest from the
4-experiment (`p144` 9.72%):

- **No overlap at all** among their top-12 most distinctive (rare ×
  intense) features — each strong binder has its own signature, not a
  shared one, at first pass.
- Filtering those features to require **silence across 5 weak binders**
  (`new249`, `new211`, `new226`, `new163`, `b44`): 32/48 survive — but
  this alone isn't strong evidence, since rare features (by
  construction, density ~0.01-0.07%) are expected to often be silent on
  any additional small set of designs regardless of binding.
- **The sharper test**: of those 32 survivors, checking whether they're
  also silent on the *other* strong binders (not just weak ones) — 29
  are private (unique to one design, silent everywhere else, likely
  idiosyncratic quirks), but **3 features are shared specifically
  between `new157` and `new241`** — the 1st and 3rd strongest binders of
  the 14-experiment — while staying silent across every weak binder
  tested.

### The shared motif, in detail

`new157` and `new241` came from **different BoltzGen design batches**
(229 vs. 319) — not scaffold-mates from the same generation batch, which
makes the overlap more notable, not a trivial artifact.

Pulling the actual sequence windows (±15 residues) around where each
shared feature fires strongest in each design:

- **Feature 11736** — both fire on a hydrophobic residue shortly after a
  literal **"ATG" tripeptide**, at a similar relative position (~residue
  22-23 of ~145):
  ```
  new157: EAARAVLHAAATGKI[A]PCSSPIAQKLGEAAA
  new241: SAAVAAAVYGATGEP[G]PAATPAAARAGTAAG
  ```
- **Feature 12086** — the most striking: both windows end in the
  **exact same 5-residue string, "NGGIE"**, near the C-terminus (~26-27
  residues from the end in both):
  ```
  new157: KEDLIPAVADALGDV[L]GERAGAWIEENGGIE
  new241: RADEAPAMGRAVGEG[L]VRGAGDWIAANGGIE
  ```
- **Feature 5521** — weak; different activating residues, no clear
  shared substring. Likely not real.

**Honest interpretation**: two of three shared features show a genuine,
specific, recurring sequence pattern — not just "both hydrophobic." But
whether this is *binding-relevant* or just a shared *scaffold* feature
(a common C-terminal capping pattern BoltzGen's generator tends to
produce, unrelated to binding) is not yet resolved — that distinction
hinges on whether these positions are spatially near the binder-target
interface or on the face pointing away from it, which needs the
predicted 3D structures to check.

### Structural check: resolved — mostly against the binding-relevant hypothesis

Using the actual predicted complex structures
(`Final_designs/boltzgen_conf_1_batch_generation-top-1000-by-composite/`,
`rank0014_design_spec_157.cif` / `rank0019_design_spec_241.cif`), with
the binder chain in each verified by **exact sequence match** against the
sequences used throughout this analysis (not trusted by filename alone —
`design_spec_N` repeats across the 80k pool, see `docs/TODO.md`). For
each candidate residue, measured CA–CA distance to the nearest
target-chain residue, and expressed it as a percentile of that binder's
own full distance-to-target distribution (calibration baseline —
"typical" for that specific structure):

| Feature | new157 distance (percentile) | new241 distance (percentile) | Interpretation |
|---|---|---|---|
| 11736 ("ATG" motif) | 23.7 Å (96th) | 22.0 Å (92nd) | Among the **farthest** residues from target in both — not interface-relevant |
| 12086 ("NGGIE" exact match) | 14.4 Å (56th) | 12.2 Å (45th) | Roughly **median** distance in both — unremarkable, not interface-proximal |
| 5521 (weak sequence match) | 10.1 Å (32nd) | 7.9 Å (17th) | Relatively **closer** than the other two, though still not a direct contact (binder's own closest residue is ~4.4 Å in both structures) |

**Status**: resolved, mostly against the binding-relevant hypothesis. The
two motifs that looked most compelling on sequence alone (feature 11736's
exact "ATG" match and feature 12086's exact "NGGIE" match) both sit far
from the predicted interface — consistent with shared *scaffold*
architecture rather than a binding-relevant motif, and consistent with
this whole analysis's dominant theme (§6 keyword analysis, InterProScan's
Bcl-2-fold dominance) that ESMC/SAE features mostly capture generic fold
identity, not binding specificity. Feature 5521 — the weakest sequence
match — is ironically the most interface-proximal of the three, though
still 1.8–2.3x farther than each binder's own closest residue, so this is
not a strong lead either. **Net effect: this test did what it was
supposed to do — it downgrades rather than confirms this lead. Not
recommended as a design target without further, stronger evidence.**

### Narrowing the search to the interface itself — a better candidate emerges

The §7 search above ranked each strong binder's distinctive features over
its *whole* sequence, which — per the structural check just above — is
dominated by generic scaffold signal that swamps anything actually near
the interface. Repeating the same "surprise" ranking (rarity × intensity),
but restricted to each binder's own **closest quartile of residues to the
target** (by CA–CA distance, from the same predicted structures, ~35
residues out of ~140 per design), for all 4 strongest real binders (top 3
of the 14-experiment — `new157`, `new089`, `new241` — plus the top 1 of
the 4-experiment, `p144`), narrows the top-20-per-binder search space from
~140 to ~35 residues each. `phase2_feature_extraction/sae/interface_restricted_feature_search.py`.

**Result**: only two feature/binder-pair overlaps survive being silent on
all 5 weak binders (`new249`, `new211`, `new226`, `new163`, `b44`, checked
both whole-design and, where structures exist, interface-region only):

- **Feature 372**, shared `new157` & `new241` (same pair as §7's original
  finding, but a different feature) — sequence match at the raw-string
  level is weak (~27% aligned identity around the activating residue,
  roughly what generic composition bias would produce). Not compelling.
- **Feature 10685**, shared `new157` & `new089` — a pair **not** flagged
  by the whole-binder search in §7 at all. Their activating windows:
  ```
  new157: ERVLADGVDWGRVLA[C]FVLAYAAGKVLKEKG
  new089: SRVLADGVDLGRIIA[C]LGLGAYLCSLLIDQG
  ```
  Both share a literal, **exact 8-residue substring "RVLADGVD"**,
  confirmed by direct string search (not just SAE feature similarity) —
  and it does **not** appear anywhere in `new241`, `p144`, or any of the
  4 structurally-checked weak binders. Scanning the full 80,000-design
  pool for this exact substring: it appears in only **89/80,000 (0.11%)**
  designs — an uncommon, specific fragment, not a generic scaffold
  cliché like the "ATG"/"NGGIE" motifs above. Structurally, both
  occurrences sit close to the interface: **14th and 11th percentile**
  of each binder's own distance-to-target distribution (7.6 Å and 7.1 Å)
  — notably closer than any earlier candidate in this write-up.

**LLM labels**: feature 10685 — "Cysteine (C) identity at the activating
position, consistently appearing in a hydrophobic local context with
aliphatic residues (V, I, L, A) in surrounding positions." Feature 372 —
"Leucine (L) residues immediately following an acidic/basic-rich motif
(RAV[RKA][RA]) with a conserved G-V-D pattern in the following
positions."

**InterPro**: neither feature was in the curated 95-feature candidate set
(built from each feature's single highest-activating examples pool-wide,
not from this later interface-restricted search), so there was no
existing annotation. Submitting `new157` and `new089`'s full sequences
directly to InterProScan5 (`interpro_check_motif_candidate.py`) found
both hit the **same generic Bcl-2-like apoptosis-regulator superfamily
fold** (IPR036834 / IPR002475 / IPR026298) spanning almost the *entire*
protein in both (residues 7–128/140 and 6–147/147) — the identical
generic fold that dominates 94% of all InterPro hits across this whole
corpus (§1 above). The motif position falls inside this hit only because
the hit covers nearly the whole sequence — not distinguishing evidence
either way. **Net effect: InterPro neither strengthens nor undercuts
this candidate; the case rests entirely on the rare exact-substring match
and the interface-proximity result above.**

**Necessary statistical caveat, applying the same rigor as everywhere
else in this document**: the SAE *feature* 10685 itself (as opposed to
the literal 8-mer) fires somewhere in an estimated ~9% of the 80k pool
(extrapolated from its per-token density), because it also activates on
other, unrelated cysteine-in-hydrophobic-context positions elsewhere in
the corpus. At that base rate, the probability all 5 tested weak binders
would show zero activation **purely by chance** is ≈62% — not a rare
event. So "silent on the weak binders we happened to test" is **not**
independent statistical evidence here; the actual signal, if any, is the
rare literal-motif match itself, not the discrimination test used to
surface it.

**Status**: the single most concrete, novel, testable lead from the
whole SAE pipeline — a specific, uncommon (89/80,000) sequence motif
shared by 2 of the 3 top binders from the 14-experiment, sitting near
the predicted interface in both — but **not validated**, and not
expected to be validatable from the existing 18-binder dataset (n=2
positive occurrences is too small for any statistical test). The
concrete next step this points to: of the 87 other pool designs sharing
this exact motif that were never wet-lab tested, testing even a handful
would directly ask whether this motif associates with binding — a real,
informative experiment this analysis alone cannot answer.

---

## 8. Current status

- ✅ Layer 9 SAE trained and validated (test-set FVE matches val exactly)
- ✅ Full-pool feature mining for layer 9
- ✅ InterProScan annotation (95 candidate features, 1,100 proteins)
- ✅ LLM feature labeling (full dictionary, 16,384 features)
- ✅ Feature-vs-binding correlation, tested rigorously (null result)
- ✅ Lasso multivariate probe (null result)
- ✅ Structural check of the shared new157/new241 motif — resolved, mostly
  against the binding-relevant hypothesis (see §7)
- ✅ Interface-restricted feature search — found a new, more concrete
  candidate motif (feature 10685, new157/new089, 89/80k pool prevalence),
  flagged as an untested hypothesis for a future wet-lab round (see §7)
- ⬜ Layers 7/8 SAE training (not started)
- ⬜ Cross-referencing interpretable features against the PC7 direction
  (blocked on the PC7 direction question — see `docs/pc7_sign_reversal_findings.md`)
