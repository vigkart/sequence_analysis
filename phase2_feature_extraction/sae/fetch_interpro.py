"""Fetch InterPro domain/family/site annotations for the proteins behind a
feature's max-activating examples, via EBI's InterProScan5 REST API, and
cross-reference them against the SPECIFIC residue position that fired the
feature (not just "does this protein have any annotation at all").

Adapted from igemmodel/sae/training/fetch_interpro.py -- same API
mechanics (verified against the live service there, unchanged here), but
joins on row_index instead of "id" (see extract_binder_residues.py's
comment on why "id"/design_spec_N repeats across the 80k), and pulls
sequences from data/boltzgen_conf_1_batch_generation-scores.csv instead of
a manifest_combined.csv.

STREAMING + RESUMABLE, unlike the igemmodel version (which submits
everything, waits for everything, then writes one output file at the
end) -- this run's candidate set (~1100 unique proteins) will take a long
while, and the point of that scale was to have real coverage to look at,
not to stare at a blank terminal until the last job finishes:
  - interpro_annotations.csv is written incrementally, one protein's
    result rows appended as soon as that protein's InterProScan job
    finishes -- safe to open and inspect at any point mid-run.
  - jobs_submitted.json checkpoints (row_index -> job_id) as submission
    happens, so a crash/interrupt doesn't mean re-submitting jobs already
    in EBI's queue on restart -- already-submitted and already-resulted
    rows are both skipped automatically.
  - Prints a live ETA: deterministic for the remaining submission phase
    (fixed per-job delay), extrapolated from the observed completion rate
    for the polling phase (EBI's queue behavior isn't something this
    script can predict in advance).

Coverage caveat, same as igemmodel's version and even more pronounced
here: this corpus is 100% de novo BoltzGen-designed binders against one
TET1 site -- zero natural proteins (unlike igemmodel's corpus, which had a
natural-binder subset InterPro coverage could concentrate on). Most
examples are expected to come back with NO hit; that's not a bug.
label_features.py is meant to use a hit as strong evidence when present
and fall back to sequence-context-only reasoning when absent.

Fair-use note: this candidate set (~1100 unique proteins) is well beyond
the "few dozen" scope this script was originally sized for -- a
deliberate choice made explicitly aware of the tradeoff, not an
oversight; see the conversation this was decided in.

Usage:
    python fetch_interpro.py --examples-csv feature_analysis_results/layer_09/feature_top_examples.csv \\
        --output-dir feature_analysis_results/layer_09 \\
        --email you@example.com --candidates-csv feature_analysis_results/layer_09/interpro_candidates.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
MAIN = Path(__file__).resolve().parent.parent.parent
SCORES_CSV = MAIN / "data" / "boltzgen_conf_1_batch_generation-scores.csv"


def load_binder_sequences(row_indices: set[int]) -> dict[int, str]:
    seqs = {}
    with open(SCORES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            if row_index in row_indices:
                seqs[row_index] = row["designed_sequence - boltzgen"]
            if len(seqs) == len(row_indices):
                break
    return seqs


def submit_job(sequence: str, email: str, title: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/run",
        data={"email": email, "sequence": sequence, "title": title, "goterms": "false", "pathways": "false"},
    )
    resp.raise_for_status()
    return resp.text.strip()


def check_status(job_id: str) -> str:
    resp = requests.get(f"{BASE_URL}/status/{job_id}")
    return resp.text.strip()


def fetch_matches(job_id: str) -> list:
    resp = requests.get(f"{BASE_URL}/result/{job_id}/json")
    resp.raise_for_status()
    return resp.json()["results"][0]["matches"]


def annotate_position(matches: list, position_0indexed: int) -> str | None:
    """InterPro locations are 1-indexed inclusive; our position column is
    0-indexed (see feature_analysis.py) -- convert before comparing."""
    pos_1indexed = position_0indexed + 1
    for match in matches:
        entry = match["signature"].get("entry")
        if entry is None:
            continue
        for loc in match["locations"]:
            if loc["start"] <= pos_1indexed <= loc["end"]:
                return f"{entry['name']} ({entry['accession']})"
    return None


def load_candidate_features(args) -> list:
    if args.features:
        return [int(f.strip()) for f in args.features.split(",")]
    if args.candidates_csv:
        candidates_df = pd.read_csv(args.candidates_csv)
        assert "feature" in candidates_df.columns, f"{args.candidates_csv} has no 'feature' column"
        return sorted(candidates_df["feature"].unique().tolist())
    return None


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def write_result_rows(writer, examples_by_row: dict[int, list[dict]], row_index: int, matches: list | None) -> None:
    for ex in examples_by_row.get(row_index, []):
        annotation = annotate_position(matches, ex["position"]) if matches is not None else None
        writer.writerow({
            "feature": ex["feature"], "rank": ex["rank"], "row_index": row_index,
            "position": ex["position"], "interpro_annotation": annotation,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--email", type=str, required=True)
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=int, default=20, help="Delay between polling passes over pending jobs.")
    parser.add_argument("--submit-delay-seconds", type=float, default=3.0)
    parser.add_argument("--submit-chunk", type=int, default=25, help="Submit this many, then do one polling pass over everything pending, before continuing to submit -- interleaves submission and results instead of submit-everything-then-wait.")
    args = parser.parse_args()

    examples = pd.read_csv(args.examples_csv)
    feature_ids = load_candidate_features(args)
    if feature_ids is not None:
        examples = examples[examples["feature"].isin(feature_ids)]
        print(f"Scoped to {len(feature_ids)} feature(s), {len(examples)} example rows", flush=True)
    else:
        print(f"No feature filter -- using all {examples['feature'].nunique()} features, {len(examples)} example rows", flush=True)

    examples_by_row: dict[int, list[dict]] = {}
    for row in examples.itertuples():
        examples_by_row.setdefault(row.row_index, []).append(
            {"feature": row.feature, "rank": row.rank, "position": row.position}
        )
    unique_row_indices = sorted(examples_by_row.keys())
    n_total = len(unique_row_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "interpro_annotations.csv"
    jobs_path = args.output_dir / "jobs_submitted.json"

    # Resume support: skip rows already fully written out, and reuse job ids
    # already submitted for rows that were submitted but not yet resulted.
    already_written = set()
    if out_path.exists():
        already_written = set(pd.read_csv(out_path)["row_index"].unique().tolist())
        print(f"Resuming: {len(already_written)} proteins already have results in {out_path}", flush=True)
    submitted_jobs: dict[int, str] = {}
    if jobs_path.exists():
        submitted_jobs = {int(k): v for k, v in json.loads(jobs_path.read_text()).items()}
        print(f"Resuming: {len(submitted_jobs)} jobs already submitted per {jobs_path}", flush=True)

    to_submit = [r for r in unique_row_indices if r not in already_written and r not in submitted_jobs]
    pending = {r: jid for r, jid in submitted_jobs.items() if r not in already_written}
    print(f"{n_total} unique proteins total: {len(already_written)} done, {len(pending)} awaiting results, "
          f"{len(to_submit)} still to submit", flush=True)

    print(f"Loading sequences for {len(to_submit)} proteins to submit...", flush=True)
    sequences = load_binder_sequences(set(to_submit))

    out_is_new = not out_path.exists()
    out_f = open(out_path, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=["feature", "rank", "row_index", "position", "interpro_annotation"])
    if out_is_new:
        writer.writeheader()
        out_f.flush()

    def poll_pending_once() -> int:
        """One status-check pass over everything pending. Streams out any
        that finished. Returns count newly resolved (FINISHED or failed)."""
        resolved = 0
        for row_index, job_id in list(pending.items()):
            status = check_status(job_id)
            if status == "FINISHED":
                try:
                    matches = fetch_matches(job_id)
                except Exception as e:
                    print(f"  failed to fetch results for row {row_index}: {e}", flush=True)
                    matches = None
                write_result_rows(writer, examples_by_row, row_index, matches)
                out_f.flush()
                del pending[row_index]
                resolved += 1
            elif status in ("ERROR", "FAILURE", "NOT_FOUND"):
                write_result_rows(writer, examples_by_row, row_index, None)
                out_f.flush()
                del pending[row_index]
                resolved += 1
        return resolved

    start_time = time.time()
    n_done_at_start = len(already_written)
    n_resolved_total = 0

    submit_idx = 0
    while submit_idx < len(to_submit) or pending:
        # Submit up to --submit-chunk more jobs.
        chunk_end = min(submit_idx + args.submit_chunk, len(to_submit))
        for row_index in to_submit[submit_idx:chunk_end]:
            job_id = submit_job(sequences[row_index], args.email, title=f"sae_row_{row_index}"[:100])
            pending[row_index] = job_id
            submitted_jobs[row_index] = job_id
            jobs_path.write_text(json.dumps(submitted_jobs))
            time.sleep(args.submit_delay_seconds)
        submit_idx = chunk_end

        # One polling pass over everything currently pending.
        resolved = poll_pending_once()
        n_resolved_total += resolved

        elapsed = time.time() - start_time
        n_done = n_done_at_start + n_resolved_total
        remaining_to_submit = len(to_submit) - submit_idx
        submit_eta = remaining_to_submit * args.submit_delay_seconds
        if n_resolved_total > 0:
            rate = n_resolved_total / elapsed  # completions/sec, observed
            resolve_eta = len(pending) / rate if rate > 0 else float("inf")
        else:
            resolve_eta = float("nan")  # not enough data yet
        print(
            f"[{elapsed/60:.1f}min elapsed] done={n_done}/{n_total}  pending={len(pending)}  "
            f"still-to-submit={remaining_to_submit}  "
            f"ETA(submit)={fmt_eta(submit_eta)}  "
            f"ETA(pending resolve, observed rate)={'n/a yet' if resolve_eta != resolve_eta else fmt_eta(resolve_eta)}",
            flush=True,
        )

        if submit_idx >= len(to_submit) and pending:
            time.sleep(args.poll_seconds)

    out_f.close()
    final_df = pd.read_csv(out_path)
    n_hits = final_df["interpro_annotation"].notna().sum()
    print(f"\nDone. {n_hits}/{len(final_df)} example residues fell inside an InterPro-annotated region. "
          f"Full results in {out_path}", flush=True)


if __name__ == "__main__":
    main()
