"""Evaluate the signal-threshold rule against fixed admission regimes (Wave-3).

Design: for each held-out family, the rule variant was fitted on the other five families
(fit_thresholds.py) and evaluated only on its held-out family. Baselines are the M1
single/diversified runs on the same instances.

Primary metric: total deterministic ticks (primary), pricing-call counts, TL hits, final LP.
Oracle regime: the fixed regime (single or diversified) with fewer ticks at no worse LP
(epsilon = 0.5% of LP); rule gap-closed fraction reported per family.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

PLAN2 = Path(__file__).resolve().parents[1].parent.parent
RAW = PLAN2 / "results" / "raw"
GENDIR = PLAN2 / "paper" / "ejor_submission" / "generated"
FAMILY_OF = lambda name: ("C1" if name.startswith("C1") else "C2" if name.startswith("C2") else
                          "R1" if name.startswith("R1") else "R2" if name.startswith("R2") else
                          "RC1" if name.startswith("RC1") else "RC2")


def load_run_metrics(study: str) -> dict[tuple[str, str], dict]:
    out = {}
    d = RAW / study
    if not d.exists():
        return out
    for f in d.glob("*.json"):
        if f.name == "run_manifest.json":
            continue
        r = json.loads(f.read_text())
        its = r["iteration_rows"]
        out[(r["instance_name"], r["variant"])] = dict(
            ticks=sum(i.get("pricing_dettime_ticks") or 0 for i in its),
            wall=sum(i["pricing_runtime_seconds"] for i in its),
            tl=sum(1 for i in its if i["pricing_status"] == "TIME_LIMIT"),
            iters=len(its),
            lp=r["final_lp_objective"],
        )
    return out


def main() -> None:
    base = {}
    for s in ("m1_c", "m1_r1", "m1_r2", "m1_rc"):
        base.update(load_run_metrics(s))
    rule = {}
    import re as _re
    for d in sorted(RAW.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("rule_") and d.name != "rule_sanity":
            rule.update(load_run_metrics(d.name))
    if not rule:
        print("no rule runs yet")
        return

    rows = []
    for (inst, var), m in rule.items():
        fam = FAMILY_OF(inst)
        s = base.get((inst, "single_tight_30s"))
        d = base.get((inst, "diversified_tight_30s"))
        if not s or not d:
            continue
        # oracle: fewer ticks among regimes whose LP is not worse by >0.5%
        best_lp = min(s["lp"], d["lp"])
        ok = [x for x in (s, d) if x["lp"] <= best_lp * 1.005]
        oracle_ticks = min(x["ticks"] for x in ok)
        rows.append(dict(instance=inst, family=fam,
                         rule_ticks=m["ticks"], rule_lp=m["lp"], rule_tl=m["tl"], rule_iters=m["iters"],
                         single_ticks=s["ticks"], single_lp=s["lp"], single_tl=s["tl"],
                         div_ticks=d["ticks"], div_lp=d["lp"], div_tl=d["tl"],
                         oracle_ticks=oracle_ticks))
    df = pd.DataFrame(rows)
    df["ticks_vs_div"] = df.rule_ticks / df.div_ticks
    df["ticks_vs_single"] = df.rule_ticks / df.single_ticks
    df["gap_closed"] = 1 - (df.rule_ticks - df.oracle_ticks) / (df.div_ticks - df.oracle_ticks).replace(0, pd.NA)
    df.to_csv(GENDIR / "rule_evaluation.csv", index=False)

    print("=== family-level summary (medians) ===")
    g = df.groupby("family").agg(
        n=("instance", "count"),
        rule_over_div=("ticks_vs_div", "median"),
        rule_over_single=("ticks_vs_single", "median"),
        gap_closed=("gap_closed", "median"),
        lp_vs_best=("rule_lp", "median"),
    )
    print(g.round(3).to_string())
    print("\noverall medians: rule/div =", round(df.ticks_vs_div.median(), 3),
          "| rule/single =", round(df.ticks_vs_single.median(), 3),
          "| gap_closed =", round(df.gap_closed.median(), 3))
    try:
        w = stats.wilcoxon(df.rule_ticks, df.div_ticks)
        print(f"Wilcoxon rule vs diversified ticks: W={w.statistic:.0f}, p={w.pvalue:.2e}, n={len(df)}")
    except ValueError as e:
        print("wilcoxon:", e)
    lp_worse = (df.rule_lp > df[["single_lp", "div_lp"]].min(axis=1) * 1.005).mean()
    print(f"fraction of instances where rule LP worse than best fixed by >0.5%: {lp_worse:.2f}")


if __name__ == "__main__":
    main()
