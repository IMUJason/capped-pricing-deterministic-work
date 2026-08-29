"""Scan all raw study runs for false-termination / empty-pool-at-cap events (Finding F2 incidence).

Event definitions:
- empty_at_cap: an iteration with pricing_status == TIME_LIMIT and candidate_count == 0
- false_termination: run terminates with 'no_negative_candidate' whose last pricing call was TIME_LIMIT
Output: paper/ejor_submission/generated/f2_incidence.csv + console summary by study/family/variant
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parents[1]
PLAN2 = PKG.parents[1]
RAW = PLAN2 / "results" / "raw"
GENDIR = PLAN2 / "paper" / "ejor_submission" / "generated"
GENDIR.mkdir(parents=True, exist_ok=True)

rows = []
for study_dir in sorted(RAW.iterdir()):
    if not study_dir.is_dir():
        continue
    for f in study_dir.glob("*.json"):
        if f.name == "run_manifest.json":
            continue
        run = json.loads(f.read_text())
        its = run.get("iteration_rows", [])
        n_empty_cap = sum(1 for r in its if r.get("pricing_status") == "TIME_LIMIT" and r.get("candidate_count", 0) == 0)
        n_cap = sum(1 for r in its if r.get("pricing_status") == "TIME_LIMIT")
        term = run.get("termination_reason", "")
        last_status = its[-1]["pricing_status"] if its else ""
        false_term = term == "no_negative_candidate" and last_status == "TIME_LIMIT"
        rows.append(dict(
            study=run.get("study_name", study_dir.name),
            instance=run.get("instance_name", ""), variant=run.get("variant", ""),
            iters=len(its), cap_iters=n_cap, empty_at_cap_iters=n_empty_cap,
            termination=term, last_status=last_status, false_termination=false_term,
        ))

df = pd.DataFrame(rows)
df.to_csv(GENDIR / "f2_incidence.csv", index=False)
print(f"total runs scanned: {len(df)}")
print(f"runs with >=1 empty-at-cap iteration: {(df.empty_at_cap_iters > 0).sum()}")
print(f"false terminations (uncertified): {df.false_termination.sum()}")
print("\nby study x variant:")
print(df.groupby(["study", "variant"]).agg(runs=("instance", "count"),
                                           empty_at_cap=("empty_at_cap_iters", "sum"),
                                           false_term=("false_termination", "sum")).to_string())
