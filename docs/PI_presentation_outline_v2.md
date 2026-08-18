# PI Presentation Outline — ESMC Interpretability + SAE Work

Built for a 60–90 min meeting. Supersedes `PI_presentation_summary.md` for
presentation purposes (that doc predates the SAE work entirely, and its
§3.3 states the original PC7 cross-experiment confirmation as settled —
tonight's work found a real bug in that validation and revised the
picture; corrected version is in §3 below). Suggested timing per section
in headers; adjust freely.

---

## 1. Opening / framing (5 min)

- Restate the core question: does ESMC-600M (the protein language model
  used to embed all 80,000 candidate binders) actually encode
  binding-relevant information, and if so, *where inside the model*?
- Why this matters beyond curiosity: BoltzGen already generates and scores
  80k designs by structural-confidence metrics alone (iptm, ipsae, etc.).
  If ESMC's internal representations also carry binding signal, that's a
  second, orthogonal signal for scoring/optimizing — and if we can find
  *where* and (via SAE work) *what* it represents, it becomes a
  legitimate target for a generative model instead of brute-force
  generate-and-filter.
- State the validation discipline up front, since it's the throughline of
  everything that follows: **computational metrics alone are not trusted
  as ground truth here** — every finding in this project has been
  checked against real wet-lab activity data before being acted on. This
  isn't boilerplate; it directly killed one early lead (§2) and forced a
  correction to what looked like a validated finding (§3).

---

## 2. Phase 1: finding the binding-relevant layer (20 min)

- **Pipeline**: embedded all 80,000 designs at all 36 ESMC layers, two
  conditions — binder alone, and binder+target concatenated via ESMC's
  native chain-break token ("joint"). Verified directly in ESMC's source
  that joint condition gives genuine bidirectional attention between
  binder and target, not just a formatting separator.
- **PCA per layer** as a compressed, workable coordinate system for
  correlating against activity data (36 layers × 3 preprocessing variants
  × 20 components = a real search space, not a single guess).
- **Early lead that didn't survive scrutiny — layer 14**: strongest single
  computational-metric (iptm) correlation, until two checks killed it:
  - Length confound: ~20% of the raw correlation was attributable to
    binder length, not binding.
  - Real-data check: correlation with actual mCherry activity was ~0
    (|r|<0.16) once real data came in.
  - **Worth stating explicitly**: a strong correlation with a
    computational proxy does not imply a real biological signal. This
    result is the reason the project insists on real-data validation for
    everything downstream, including the SAE work.
- **Side finding**: ESMC's embeddings predict intrinsic sequence-quality
  properties (hydrophobic patch size, developability/liability score)
  far better than target-interface quality — R²≈0.71–0.75 vs. best
  interface metric R²≈0.30. The model reads generic sequence composition
  much more readily than binding-specific structure from these proxies.
- **The real finding — layers 7–9, joint condition**, converged on by two
  independent methods:
  - PCA-axis screen-then-confirm: screened 36 layers × 3 variants × 20
    PCs × 2 conditions on 14 real binders (one wet-lab experiment),
    confirmed on 4 completely independent real binders (a separate
    experiment). Peak: layer 9, PC7, joint — weighted concordance 0.904
    on the screen set.
  - Independently, a regularized multivariate probe (RidgeCV /
    LogisticRegressionCV on the top-10 PCs, trained on the 14, evaluated
    — never refit — on the 4) landed on the same layers, 7–8 joint.
  - Two different methods, same narrow region of a 36-layer network — the
    core reason this location is trusted.
- **Characterizing the signal**: PC7 explains only ~2% of a layer's
  variance (rank 8/50) — a real but narrow, specific sub-signal, not a
  dominant bulk-variance direction. Distribution is a smooth continuous
  gradient across all 80k designs (confirmed via HDBSCAN finding no
  discrete clusters) — consistent with "binding" being a specific,
  narrow concept, not something that partitions designs into groups.
- **Joint > binder-alone, specifically for the real signal** — even
  though binder-alone and joint are statistically indistinguishable on
  every *computational* metric. Suggests structural-confidence proxies
  (iptm/ipsae) may be missing something about real function that letting
  the model actually attend across binder and target captures.

---

## 3. Testing our own result properly (15 min)

Frame this as a rigor/process section, not a setback — this is exactly
the same caliber of self-correction as the layer-14 debunking in §2, just
caught later and on a more central result.

- Setting up the SAE work (§4) required knowing the *direction* of PC7
  (does high or low PC7 mean better binding?) — a question the original
  screen-then-confirm result never actually needed to answer, since its
  metrics were direction-agnostic by construction.
- Checking this directly surfaced two real bugs:
  - A column-indexing bug (PC7 meant column 7, code used column 6) —
    caught and fixed; doesn't affect anything already trained.
  - **The more important one**: the original confirmation step scored the
    held-out 4 binders with a metric that silently picks whichever of two
    possible directions scores better — meaning "8/25 candidates achieved
    a perfect 6/6 confirmation" never actually tested whether the
    direction found during screening replicates. No model was fit on the
    held-out data (that part was fine) — the flaw was entirely in the
    evaluation metric.
- **Corrected re-analysis**: locked the direction found on the screen set,
  re-scored the confirm set with no re-optimization. Result: 0 of the top
  50 candidates genuinely confirm — and the miss isn't random, it's a
  systematic reversal (confirmed via an exact permutation test, following
  the project's own established n=4 exact-test methodology: the true
  label assignment is the single most-reversed of all 24 possible
  relabelings, p=0.042 — a real, structured effect, not noise).
- **Root-caused, not just flagged**: ruled out a length confound (checked
  directly — not the layer-14 story repeating), ruled out a one-off
  embedding-script artifact (reversal persists, actually strengthens,
  without the one point that used a different code path), ruled out
  recent codebase-reorganization data staleness (verified the PCA fit
  matches the current embeddings' mean to full floating-point precision).
  Landed on: **the two wet-lab experimental batches are not on a
  comparable activity scale** — the confirm-set batch reads ~2.3x higher
  in raw % activity on average, for reasons unrelated to PC7.
- Correcting for that batch-level offset before pooling flips the
  evidence back to match the *original screen direction*, not away from
  it.
- **Bottom line to state plainly**: layer localization (7–9) remains
  solid and multiply-corroborated — nothing here undermines *where* the
  signal lives. Direction is now a well-supported working hypothesis
  (matching the screen set), not a coin flip, but not fully closed either
  — pending one concrete, answerable question.
- **Direct ask for the PI**: was there a known calibration or
  normalization difference between the two mCherry experimental rounds?
  This is a real, specific question that could close this out.

---

## 4. Phase 2: mechanistic interpretability via sparse autoencoders (20 min)

- **Motivation, two reasons**:
  - Scientific: PC7 tells us *that* a direction in layer 7–9 space
    correlates with binding — it says nothing about *what it represents*
    mechanistically (motif? contact geometry? a specific property?).
  - Practical: "maximize PC7" is a narrow, single-axis proxy for any
    future generative/optimization step. Before committing a generative
    model to that objective, want to know if it's genuinely causal or
    entangled with something else.
- **Approach**: train a Sparse Autoencoder (SAE) on ESMC's residue-level
  activations — decomposes the dense 1152-dim representation into a much
  larger (16,384), *sparse* (only ~64 active per residue) dictionary of
  features, each ideally more monosemantic and human-interpretable than a
  raw dimension or a single PCA axis.
- **Not built from scratch**: adapted a validated architecture from the
  lab's own prior interpretability work (the SENTINEL/iGEM project) —
  TopK sparse autoencoder, dead-feature-revival mechanism (AuxK), and a
  fixed-centering scheme that specifically avoids a known failure mode
  (outlier "massive activation" dimensions killing off dictionary
  features during training). Reused proven methodology, not a new
  from-scratch design.
- **Data**: residue-level (not pooled) embeddings at layers 7, 8, 9,
  joint condition, all 80,000 designs — ~175GB, generated and
  integrity-verified.
- **Engineering required to make this project's scale tractable** on
  available hardware (brief, don't over-dwell): binder-only residue
  extraction (target excluded from SAE training but retained for
  context), a RAM-aware stratified train/val/test split, data-loading
  throughput fixes.
- **Training result — layer 9 (first of the 3 target layers)**:
  - 50 epochs; reconstruction quality (fraction of variance explained,
    FVE) = **0.902 on validation, and identically 0.902 on a fully
    independent held-out test set** (~2M residues never touched during
    training) — no overfitting gap.
  - **0 dead features out of 16,384** — every single dictionary feature
    is alive and firing. A fully healthy, fully-utilized dictionary.
- **Status**: layer 9 done; layers 7 and 8 queued next (same pipeline,
  ~4–5h each, unattended).

---

## 5. Feature interpretation — in progress (10 min)

- For every one of the 16,384 features: mined its ~15 hardest-firing
  residues across the *entire* 80,000-design pool (not a sample).
- Selected a bounded set of features (~95) for deeper annotation via two
  criteria: (a) features that fire specifically on the real,
  wet-lab-tested binders, and (b) features at the density extremes (most
  generic + rarest-but-real) — the same practical selection strategy
  already validated in the lab's prior interpretability work.
- Running these through InterProScan (real protein domain/family
  database lookup) against ~1,100 associated designs — notable given
  these are 100% de novo, computationally generated sequences with zero
  evolutionary history, so hits are not guaranteed at all.
- **Early result** (in progress as of this meeting, run live and
  updating): hit rate running around **~38%** — well above the
  conservative expectation for a purely de novo corpus. A specific
  pattern is already visible: multiple independent features are
  converging on the **Bcl-2 family fold** (a classic helical-bundle
  architecture) — suggesting BoltzGen's generated binders structurally
  converge on recognizable general folds even without literal
  evolutionary relationship.
- **Next step**: LLM-assisted labeling (batch API) of all 16,384 features
  from their examples + InterPro evidence, then cross-reference which
  interpretable features align with the PC7 direction — the actual
  mechanistic payoff this whole phase is aimed at.

---

## 6. Current status snapshot (5 min)

Simple checklist slide:

- ✅ Layer/axis selection, with corrected direction understanding
- ✅ Residue-level embedding data generation (all 80k, 3 layers)
- ✅ Layer 9 SAE trained and validated (test-set FVE matches val)
- ✅ Full-pool feature mining for layer 9
- 🔄 InterProScan annotation (in progress, live during this meeting)
- ⬜ LLM feature labeling (not started)
- ⬜ Layers 7/8 SAE training (not started)
- ⬜ Cross-referencing interpretable features against PC7 (blocked on
  above)
- ❓ PC7 direction — working hypothesis, pending the calibration
  question above

---

## 7. Future directions (15 min)

### 7.1 What the current site's data tells us about strategy
- Real binder activity here tops out around 9–10% (best measured
  binders), against a dCas9-TET direct-fusion positive control at
  36–73% across the two experiments — real, measurable function, but
  well short of what's needed for standalone therapeutic-grade efficiency
  (>60%).
- **This is not a negative result — it's a useful, expected data point**:
  this site can contribute substantial binding, but the right strategic
  read is that it belongs as *one component* of a multivalent binder, not
  as a standalone therapeutic candidate on its own.

### 7.2 Optimize binding at this site using what's already built
- Once the PC7 direction question is closed, the layer 7–9 axis (and/or
  SAE-derived mechanistic features) becomes a legitimate scoring/guidance
  signal for a next round of design at this site.
- Longer-term, the originally-scoped path: a lightweight diffusion head
  trained to generate new binder embeddings directly in this validated
  latent space, decoded via the frozen ESMC transformer and structurally
  validated via ESMFold2 cofolding.
- The SAE's interpretable features (once labeled) give this a more
  mechanistically-grounded objective than "maximize PC7" alone — e.g., if
  specific features turn out to correspond to real motifs that explain
  *why* PC7 separates good from bad binders, those become an
  interpretable, checkable target for generation rather than a single
  opaque axis.

### 7.3 Expand to other TET1 sites → multivalent recruiter
- Start new BoltzGen design campaigns against additional sites on TET1.
- **Efficiency point worth emphasizing**: the full methodology built here
  (embed → layer/axis search → real-data validation → SAE
  interpretability) is now a largely reusable, already-debugged pipeline
  — a new site doesn't mean rebuilding from scratch, it means re-running
  validated tooling on new data.
- **Open scientific question, worth flagging directly**: does the layer
  7–9 signal found here generalize to a *different* site, or is it
  specific to this target/binder family? Directly testable once new-site
  real data exists — if it transfers, that's evidence of a general
  protein-protein-binding feature inside ESMC (a substantially bigger
  finding); if not, still useful, just narrower in scope than hoped.
- Use the best-validated binder per site as an "arm" — combine several
  modest-affinity, site-specific binders onto one multivalent scaffold
  (e.g. via Genie3 motif-scaffolding, already discussed as an approach in
  earlier project notes) so that simultaneous multi-point engagement
  (avidity) closes the potency gap that no single site's binder can close
  alone.
- This — not any single site's binder — is the actual path to a
  therapeutically-relevant TET1/TET recruiter.

---

## 8. Closing — asks for the PI (5–10 min)

- **Concrete, answerable question**: calibration/normalization difference
  between the two mCherry experimental rounds (§3)?
- **Priority/resourcing question**: finish out the current SAE track
  (layers 7/8 + full labeling + PC7 cross-reference) before starting new
  design campaigns, or begin new-site BoltzGen campaigns in parallel now?
- **Feedback on the multivalent-scaffold direction** (§7.3): Genie3
  specifically, or an alternative scaffolding strategy they'd prefer?
- Open floor for questions on any section above.
