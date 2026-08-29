"""Fit threshold-rule parameters with leave-family-out nesting and emit Wave-3 configs.

For each held-out family: fit (theta_delta) on the other five families by maximizing
balanced accuracy for predicting next-iteration limit-hits; emit one rule variant per
held-out family, evaluated only on that family. Baselines (single/diversified) are
already available from M1 runs, so Wave 3 runs each rule variant on its held-out family.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PKG = Path(__file__).resolve().parents[1]
PLAN2 = PKG.parents[1]
RAW = PLAN2 / "results" / "raw"
GENDIR = PLAN2 / "paper" / "ejor_submission" / "generated"

M1_STUDIES = ["m1_c", "m1_r1", "m1_r2", "m1_rc"]
FAMILY_OF = lambda name: ("C1" if name.startswith("C1") else "C2" if name.startswith("C2") else
                          "R1" if name.startswith("R1") else "R2" if name.startswith("R2") else
                          "RC1" if name.startswith("RC1") else "RC2")


def load_m1_iterations() -> pd.DataFrame:
    rows = []
    for s in M1_STUDIES:
        d = RAW / s
        if not d.exists():
            continue
        for jf in d.glob("*.json"):
            if jf.name == "run_manifest.json":
                continue
            run = json.loads(jf.read_text())
            rows.extend(run.get("iteration_rows", []))
    df = pd.DataFrame(rows)
    df["family"] = df["instance_name"].map(FAMILY_OF)
    df["tl_hit"] = (df["pricing_status"] == "TIME_LIMIT").astype(int)
    return df


def fit_theta(train: pd.DataFrame) -> dict:
    """theta_delta = value of dual_l1_displacement maximizing balanced accuracy for TL-hit prediction."""
    x = train["dual_l1_displacement"].values
    y = train["tl_hit"].values
    if y.sum() in (0, len(y)):
        return dict(theta_delta=float("inf"), balanced_accuracy=float("nan"), note="degenerate train")
    grid = np.quantile(x, np.linspace(0.1, 0.95, 18))
    best = (0.5, grid[0])
    for th in grid:
        pred = (x > th).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        bal = (tpr + tnr) / 2
        if bal > best[0]:
            best = (bal, th)
    return dict(theta_delta=float(best[1]), balanced_accuracy=float(best[0]))


def main() -> None:
    df = load_m1_iterations()
    families = sorted(df.family.unique())
    fitted = {}
    for hold in families:
        train = df[df.family != hold]
        fitted[hold] = fit_theta(train)
    (GENDIR / "fitted_thresholds.json").write_text(json.dumps(fitted, indent=2))
    print(json.dumps(fitted, indent=2))

    base = dict(pricing_time_limit=30, pricing_threads=4, max_iterations=40, max_route_age=1000,
                enable_pool_pruning=False, tighten_big_m=True, strategy="diversified",
                candidate_pool_size=20, diversity_cost_weight=0.45,
                rule_mode="signal_threshold", rule_tl_window=5, rule_tl_rate_threshold=0.4,
                rule_hard_add_count=3, rule_hard_beta=0.75, rule_pool_first_min_candidates=5,
                use_route_reservoir=True, reservoir_max_size=1200, adaptive_hardening=False,
                use_dual_stabilization=False, use_subset_row_cuts=False)
    for hold in families:
        insts = sorted(df[df.family == hold].instance_name.unique())
        v = dict(base)
        v.update(name=f"rule_lfo_{hold}", add_count=10,
                 rule_dual_disp_threshold=round(fitted[hold]["theta_delta"], 4))
        cfg = {"study_name": f"rule_{hold}", "instances": insts, "variants": [v]}
        out = PKG / "experiments" / "configs" / f"rule_{hold}.yaml"
        out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        print("wrote", out.name, len(insts), "instances, theta_delta=", round(fitted[hold]["theta_delta"], 3))


if __name__ == "__main__":
    main()
