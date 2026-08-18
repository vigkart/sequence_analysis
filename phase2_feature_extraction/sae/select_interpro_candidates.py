"""Selects a bounded set of SAE features to send through InterProScan
(fetch_interpro.py), combining two criteria -- see docs/TODO.md's "Future
direction 1" and the review discussion that led here:

  (a) Features that fire on the real, wet-lab-tested binders specifically
      (real_binder_row_indices.json -- 18 designs with actual mCherry
      measurements, cross-referenced against feature_top_examples.csv:
      any feature with one of these designs in its own top-N list).
  (b) Density extremes: the highest-density (most generic) and lowest-
      density-but-nonzero (rarest still-alive) features, matching the same
      "highest-density + rare-but-strong" recipe igEM's own team used in
      practice for exactly this bottleneck (see igemmodel/sae/RESULTS.md
      and FEATURE_LABELING_SETUP.md) -- not the full dictionary, which
      would need far more InterProScan jobs than fair-use for a shared
      free service allows (see fetch_interpro.py's module docstring).

The union of (a) and (b), deduplicated, is small enough (typically a few
dozen to ~100 features) to stay within that fair-use bound while still
covering both "does this feature matter for binders we've actually
measured" and "what does the dictionary look like at its extremes."

Usage:
    python select_interpro_candidates.py \\
        --feature-stats feature_analysis_results/layer_09/feature_stats.csv \\
        --feature-examples feature_analysis_results/layer_09/feature_top_examples.csv \\
        --real-binders real_binder_row_indices.json \\
        --output-dir feature_analysis_results/layer_09 \\
        --n-density-high 15 --n-density-low 15
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--feature-examples", type=Path, required=True)
    parser.add_argument("--real-binders", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-density-high", type=int, default=15)
    parser.add_argument("--n-density-low", type=int, default=15)
    args = parser.parse_args()

    stats_df = pd.read_csv(args.feature_stats)
    examples_df = pd.read_csv(args.feature_examples)
    with open(args.real_binders) as f:
        real_binders = json.load(f)  # name -> row_index
    real_row_indices = set(real_binders.values())

    # (a) Real-binder criterion.
    hits = examples_df[examples_df["row_index"].isin(real_row_indices)]
    real_binder_features = sorted(hits["feature"].unique().tolist())
    print(f"(a) Real-binder criterion: {len(real_binder_features)} features have >=1 of the "
          f"{len(real_row_indices)} wet-lab-tested binders in their own top-{examples_df['rank'].max()+1} list")
    # Show which real binder(s) drove each hit, for transparency.
    row_to_name = {v: k for k, v in real_binders.items()}
    hit_summary = (
        hits.assign(binder_name=hits["row_index"].map(row_to_name))
        .groupby("feature")["binder_name"].apply(lambda s: sorted(set(s)))
        .reset_index()
        .rename(columns={"binder_name": "matching_real_binders"})
    )

    # (b) Density extremes -- exclude dead features (density==0) entirely,
    # both from the "high" and "low" ends (a dead feature isn't "rare but
    # strong", it's just dead, and there's nothing to send to InterProScan
    # for it since it has zero max-activating examples).
    alive = stats_df[~stats_df["dead"]].sort_values("density")
    low_density_features = alive.head(args.n_density_low)["feature"].tolist()
    high_density_features = alive.tail(args.n_density_high)["feature"].tolist()
    print(f"(b) Density extremes: {len(low_density_features)} lowest-density (rarest) + "
          f"{len(high_density_features)} highest-density (most generic) alive features")

    all_candidates = sorted(set(real_binder_features) | set(low_density_features) | set(high_density_features))
    print(f"\nUnion, deduplicated: {len(all_candidates)} candidate features for InterProScan")

    # How many unique proteins would this actually require? (fair-use check
    # -- see fetch_interpro.py's module docstring.)
    candidate_examples = examples_df[examples_df["feature"].isin(all_candidates)]
    n_unique_proteins = candidate_examples["row_index"].nunique()
    print(f"-> {n_unique_proteins} unique proteins across these features' top examples "
          f"(fetch_interpro.py submits one InterProScan job per unique protein)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "interpro_candidates.csv"
    out_df = pd.DataFrame({"feature": all_candidates})
    out_df["criterion"] = out_df["feature"].apply(
        lambda f: "+".join(filter(None, [
            "real_binder" if f in real_binder_features else "",
            "high_density" if f in high_density_features else "",
            "low_density" if f in low_density_features else "",
        ]))
    )
    out_df = out_df.merge(stats_df[["feature", "density", "fire_count"]], on="feature", how="left")
    out_df = out_df.merge(hit_summary, on="feature", how="left")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
