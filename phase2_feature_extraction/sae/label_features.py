"""LLM auto-labeling of SAE features from their max-activating examples,
adapted from igemmodel/sae/training/label_features.py.

Per the project decision: runs on EVERY feature with at least one recorded
example (all 16,384 here -- 0 dead in the layer 9 checkpoint), not just
the InterPro-candidate subset -- Batches API cost scales fine to the full
dictionary (~$12 estimated, linear from igemmodel's own $3-at-4096
figure), unlike InterProScan which had to be scoped down. Every feature
gets an LLM-written description from its sequence-context examples alone
if InterPro found nothing there (the InterPro caveat in the system prompt
below already handles "no annotation" gracefully -- most examples in this
100%-de-novo corpus are still expected to have none).

Difference from igemmodel's version, per this project's explicit request:
the InterPro label tally (how many of a feature's top-N examples hit each
annotation) is computed DETERMINISTICALLY IN CODE and saved as its own
output column (interpro_tally), not left for the LLM to count and restate
in its own text -- LLMs are an unreliable way to reproduce an exact count
that code can always get right. The tally is still shown to the LLM as
evidence (same as igemmodel's version), so its description can reference
it in prose, but the authoritative counts live in a column no one has to
trust the model's arithmetic for.

Uses the Message Batches API (bulk, non-interactive, 50% cheaper than
per-feature synchronous calls) -- same as igemmodel's version. Requires
`pip install anthropic` and ANTHROPIC_API_KEY set (or `ant auth login`) --
NOT covered by a Claude.ai Pro/Max subscription.

Usage:
    # Dry run first -- no API call, no cost, see the exact prompt:
    python label_features.py --examples-csv feature_analysis_results/layer_09/feature_top_examples.csv \\
        --output-dir feature_analysis_results/layer_09 --features 189 --dry-run

    # Full dictionary, with InterPro evidence merged in where available:
    python label_features.py --examples-csv feature_analysis_results/layer_09/feature_top_examples.csv \\
        --output-dir feature_analysis_results/layer_09 \\
        --interpro-csv feature_analysis_results/layer_09/interpro_annotations.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd

DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You are analyzing features from a sparse autoencoder (SAE) trained on ESM-C \
protein language model activations. Each feature is a learned direction that fires on certain \
residues. You will see the residues that activate ONE specific feature most strongly, as short \
local sequence windows with the activating residue in [brackets] -- e.g. "ABCD[E]FGHI" means \
residue E fired the feature, with D/C/B/A before it and F/G/H/I after.

You may also see an "InterPro evidence" section: a tally of real database domain/family \
annotations, one per example, each looked up at that residue's exact position in its full \
protein. Every protein in this dataset is a de novo, computationally designed binder with NO \
evolutionary relationship to any characterized protein family, so most or all examples showing \
"No InterPro annotation" is expected and NOT evidence of anything -- do not comment on its \
absence. If one InterPro annotation clearly dominates the tally (covers a majority of the \
annotated examples), treat it as strong, verified structural evidence (e.g. "resembles the X \
fold") and prefer describing the pattern in those terms over guessing from sequence alone -- but \
note explicitly that a domain hit on a de novo design indicates fold/architecture similarity, \
not evolutionary or functional relationship to that domain's normal biology. A minority or \
scattered set of different annotations is weak evidence -- weigh it accordingly.

Describe in ONE sentence the shared pattern across these examples, if any: a dominant InterPro \
domain/fold match, an amino acid identity, a local motif (e.g. adjacent to a specific residue or \
a small recurring subsequence), or a position tendency. Base your answer strictly on the evidence \
given -- do not speculate about biological function beyond what the evidence shows, and do not \
restate or recompute the InterPro counts yourself (they're provided separately, exactly). If the \
examples show no consistent pattern, say exactly: "No clear pattern." Do not add any other text."""


def aggregate_interpro(rows: pd.DataFrame) -> tuple[str | None, dict]:
    """Deterministic tally of InterPro annotations across a feature's
    examples. Returns (prompt_text, tally_dict) -- tally_dict is also
    saved verbatim to the output CSV so the counts never depend on
    trusting the LLM's own arithmetic."""
    if "interpro_annotation" not in rows.columns:
        return None, {}
    n = len(rows)
    counts = rows["interpro_annotation"].value_counts(dropna=True)
    n_annotated = int(counts.sum())
    if n_annotated == 0:
        return None, {}
    lines = [f"InterPro evidence across the {n} examples (each looked up at that residue's exact position in its full protein):"]
    for annotation, count in counts.items():
        lines.append(f"- {annotation}: {count}/{n} examples")
    n_missing = n - n_annotated
    if n_missing:
        lines.append(f"- No InterPro annotation: {n_missing}/{n} examples")
    tally = {str(k): int(v) for k, v in counts.items()}
    tally["_no_annotation"] = n_missing
    return "\n".join(lines), tally


def build_user_prompt(feature_id: int, examples: pd.DataFrame, top_n: int) -> tuple[str, dict]:
    rows = examples[examples["feature"] == feature_id].sort_values("rank").head(top_n)
    lines = [f"Feature {feature_id}, top {len(rows)} activating residues (sequence context):"]
    for i, row in enumerate(rows.itertuples(), start=1):
        lines.append(f"{i}. activation={row.activation:.3f} split={row.split} context={row.context}")

    interpro_prompt_text, tally = aggregate_interpro(rows)
    if interpro_prompt_text:
        lines.append("")
        lines.append(interpro_prompt_text)

    return "\n".join(lines), tally


def load_candidate_features(args, examples: pd.DataFrame) -> list[int]:
    if args.features:
        return [int(f.strip()) for f in args.features.split(",")]
    if args.candidates_csv:
        candidates_df = pd.read_csv(args.candidates_csv)
        assert "feature" in candidates_df.columns, f"{args.candidates_csv} has no 'feature' column"
        return sorted(candidates_df["feature"].unique().tolist())
    return sorted(examples["feature"].unique().tolist())


def submit_batch(client, requests: list) -> str:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch_requests = [
        Request(
            custom_id=f"feature-{feature_id}",
            params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            ),
        )
        for feature_id, user_prompt, model, max_tokens in requests
    ]
    batch = client.messages.batches.create(requests=batch_requests)
    print(f"Submitted batch {batch.id} ({len(batch_requests)} requests), status={batch.processing_status}", flush=True)
    return batch.id


def poll_batch(client, batch_id: str, poll_seconds: int) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            print(f"Batch {batch_id} ended. Counts: {batch.request_counts}", flush=True)
            return
        print(
            f"  {batch.processing_status}: processing={batch.request_counts.processing} "
            f"succeeded={batch.request_counts.succeeded} errored={batch.request_counts.errored}",
            flush=True,
        )
        time.sleep(poll_seconds)


def collect_results(client, batch_id: str, tallies: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for result in client.messages.batches.results(batch_id):
        feature_id = int(result.custom_id.removeprefix("feature-"))
        if result.result.type == "succeeded":
            text = next((b.text for b in result.result.message.content if b.type == "text"), "").strip()
            rows.append({"feature": feature_id, "label": text, "interpro_tally": tallies.get(feature_id, {}), "error": None})
        else:
            rows.append({"feature": feature_id, "label": None, "interpro_tally": tallies.get(feature_id, {}), "error": result.result.type})
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--interpro-csv", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    examples = pd.read_csv(args.examples_csv)
    if args.interpro_csv:
        interpro = pd.read_csv(args.interpro_csv)
        examples = examples.merge(
            interpro[["feature", "rank", "row_index", "position", "interpro_annotation"]],
            on=["feature", "rank", "row_index", "position"], how="left",
        )
        n_hits = examples["interpro_annotation"].notna().sum()
        print(f"Merged InterPro annotations: {n_hits}/{len(examples)} example rows have a hit", flush=True)

    feature_ids = load_candidate_features(args, examples)
    print(f"Labeling {len(feature_ids)} feature(s) from {args.examples_csv}", flush=True)

    requests = []
    tallies = {}
    for fid in feature_ids:
        prompt, tally = build_user_prompt(fid, examples, args.top_n)
        tallies[fid] = tally
        requests.append((fid, prompt, args.model, args.max_tokens))

    if args.dry_run:
        print(f"\nWould submit {len(requests)} requests to {args.model}. Example prompt (feature {requests[0][0]}):\n", flush=True)
        print("--- system ---")
        print(SYSTEM_PROMPT)
        print("\n--- user ---")
        print(requests[0][1])
        print("\n--- deterministic interpro_tally (saved to output, not sent for the LLM to recompute) ---")
        print(tallies[requests[0][0]])
        return

    import anthropic

    client = anthropic.Anthropic()
    batch_id = submit_batch(client, requests)
    poll_batch(client, batch_id, args.poll_seconds)

    results = collect_results(client, batch_id, tallies)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "feature_labels.csv"
    results.to_csv(out_path, index=False)

    n_ok = results["error"].isna().sum()
    print(f"\n{n_ok}/{len(results)} labeled successfully. Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
