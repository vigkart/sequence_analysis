"""Feature 10685 (interface_restricted_feature_search.py) never made it
into the curated 95-feature InterProScan candidate set from
select_interpro_candidates.py -- that set was built from the *global*
max-activating examples across the full 80k pool and each feature's
density extremes, neither of which happened to include new157/new089 for
this particular feature (its activation on them, ~4.2-4.3, isn't among
its single highest activations pool-wide). Rather than force this one
feature through that whole batch pipeline, submit the two relevant
sequences directly to EBI's InterProScan5 REST API and check whether any
domain/family hit covers the "RVLADGVD" motif position specifically
(new157 residue 90, new089 residue 81 -- 1-indexed, matching InterPro's
convention) -- reusing the exact same submit/poll/parse functions as
fetch_interpro.py, just skipping its batch-candidate-selection layer.

Usage:
    python interpro_check_motif_candidate.py --email you@example.com
"""

import argparse
import time

from fetch_interpro import submit_job, check_status, fetch_matches, annotate_position

SEQUENCES = {
    "new157": "SLAGMLAEAARAVLHAAATGKIAPCSSPIAQKLGEAAAKLCAKLKEDHPDLFDSIISLLSKIDPLDAATLARIAERVLADGVDWGRVLACFVLAYAAGKVLKEKGKEDLIPAVADALGDVLGERAGAWIEENGGIEGLVKFVEEAAS",
    "new089": "SEADFRALLRTAVKTLIDNAPESLAEEAADVRKNLDSIVDTVVDHLTANRDKLEKLTVDSLPRVVSRVLADGVDLGRIIACLGLGAYLCSLLIDQGRPELIAPVVAAVRDALVEHCPEELKKAIEAAGEALEAAARALAA",
}
# 0-indexed position of the activating Cysteine for feature 10685 (see
# interface_restricted_feature_search.py's output).
MOTIF_POSITION_0INDEXED = {"new157": 89, "new089": 80}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str, required=True)
    args = parser.parse_args()

    job_ids = {}
    for name, seq in SEQUENCES.items():
        job_id = submit_job(seq, args.email, title=f"motif_check_{name}")
        job_ids[name] = job_id
        print(f"Submitted {name}: {job_id}")
        time.sleep(3)  # be polite to the shared service, mirrors fetch_interpro.py's spacing

    print("\nPolling for results (InterProScan typically takes 1-5 min)...")
    remaining = set(job_ids)
    matches_by_name = {}
    poll = 0
    while remaining and poll < 60:
        poll += 1
        time.sleep(10)
        for name in list(remaining):
            status = check_status(job_ids[name])
            if status == "FINISHED":
                matches_by_name[name] = fetch_matches(job_ids[name])
                remaining.discard(name)
                print(f"  {name}: FINISHED, {len(matches_by_name[name])} match(es)")
            elif status in ("FAILURE", "NOT_FOUND", "ERROR"):
                print(f"  {name}: {status} -- giving up")
                matches_by_name[name] = []
                remaining.discard(name)
        if remaining:
            print(f"  ({len(remaining)} still running, poll {poll}/60)")

    print(f"\n{'='*90}\nResults\n{'='*90}")
    for name, matches in matches_by_name.items():
        pos = MOTIF_POSITION_0INDEXED[name]
        print(f"\n{name} ({len(SEQUENCES[name])} aa, motif Cys at 0-indexed pos {pos} / "
              f"1-indexed {pos+1}):")
        if not matches:
            print("  No InterPro/domain hits at all for this sequence (common for this "
                  "100% de novo corpus -- see fetch_interpro.py's coverage caveat).")
            continue
        for m in matches:
            sig = m.get("signature", {})
            desc = sig.get("description") or sig.get("name") or "?"
            entry = sig.get("entry") or {}
            ipr = entry.get("accession") if entry else None
            for loc in m.get("locations", []):
                start, end = loc.get("start"), loc.get("end")
                covers_motif = start is not None and end is not None and start <= pos + 1 <= end
                marker = "  <-- COVERS THE MOTIF POSITION" if covers_motif else ""
                print(f"  [{sig.get('signature_library_release', {}).get('library', '?')}] "
                      f"{desc}  (IPR: {ipr})  residues {start}-{end}{marker}")
        annotation = annotate_position(matches, pos)
        print(f"  annotate_position() at motif residue: {annotation}")


if __name__ == "__main__":
    main()
