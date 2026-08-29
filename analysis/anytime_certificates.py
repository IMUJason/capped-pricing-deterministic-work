"""Anytime certificate intervals for ALL instances x ALL strategies (Prop 1).

For each (instance, strategy) run: rebuild the final column pool, solve the cut-free
restricted master for optimal duals, run one strict certificate call (plain MIP solve,
no pool), and output the Prop-1 interval [z_RMP - K*eps, z_RMP] when certified, or the
best-found violated column when not.

Usage: python analysis/anytime_certificates.py <family>   # C1 C2 R1 R2 RC1 RC2
Writes: paper/ejor_submission/generated/anytime_cert_<family>.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parents[1]
PLAN2 = PKG.parents[1]
sys.path.insert(0, str(PKG / "src"))

from plan2_route_pool.vrptw_parser import parse_solomon_like_instance
from plan2_route_pool.initializer import build_feasible_seed_routes
from plan2_route_pool.master_problem import solve_master
sys.path.insert(0, str(PKG / "analysis"))
from certify_strict import build_and_solve

RAW = PLAN2 / "results" / "raw"
GENDIR = PLAN2 / "paper" / "ejor_submission" / "generated"
FAMILY_INSTANCES = {
    "C1": ["C101", "C104", "C107"], "C2": ["C204", "C208"],
    "R1": ["R101", "R105", "R108"], "R2": ["R201", "R205"],
    "RC1": ["RC105", "RC108"], "RC2": ["RC204", "RC208"],
}
STRATEGIES = ["single_tight_30s", "diversified_tight_30s"]
CERT_CAP = 300.0


def find_run(instance: str, variant: str) -> dict | None:
    for d in sorted(RAW.iterdir()):
        if not d.is_dir():
            continue
        f = d / f"{instance}__{variant}.json"
        if f.exists():
            return json.loads(f.read_text())
    return None


def rebuild_pool(inst, run):
    cols = {c.signature: c for c in build_feasible_seed_routes(inst)}
    for it in run["iteration_rows"]:
        for sig in (s for s in it.get("selected_signatures", "").split(";") if s):
            if sig not in cols:
                cols[sig] = inst.make_column([int(x) for x in sig.split("-")], None, "rebuilt")
    return list(cols.values())


def main() -> None:
    family = sys.argv[1]
    bks = pd.read_csv(PKG / "data" / "provenance" / "vrptw_dimacs_bks.csv").set_index("instance_name")
    rows = []
    for inst_name in FAMILY_INSTANCES[family]:
        row_bks = bks.loc[inst_name]
        inst = parse_solomon_like_instance(
            PKG / "data" / "raw" / "vrptw" / "controller" / "VRPTWController-master" / row_bks["relative_path"],
            source_family=str(row_bks["family"]),
        )
        K = inst.max_vehicles
        for variant in STRATEGIES + [f"rule_lfo_{family}"]:
            run = find_run(inst_name, variant if variant.startswith("rule") else variant)
            if run is None:
                continue
            pool = rebuild_pool(inst, run)
            master = solve_master(inst, pool, binary=False, subset_row_cuts=[])
            z_rmp = master.objective_value
            out = build_and_solve(inst, master.dual_customers, master.dual_vehicle, CERT_CAP)
            best_rc = out["best_rc"]
            certified = out["status"].lower().startswith("optimal") and best_rc is not None and best_rc >= -1e-6
            eps = max(-best_rc, 0.0) if (best_rc is not None and certified) else None
            lb = z_rmp - K * eps if certified else None
            rows.append(dict(
                instance=inst_name, family=family, variant=variant, K=K,
                z_rmp=round(z_rmp, 2), bks=float(row_bks["best_known_solution"]),
                cert_status=out["status"], best_rc=None if best_rc is None else round(best_rc, 2),
                certified=certified,
                eps=None if eps is None else round(eps, 4),
                lower_bound=None if lb is None else round(lb, 2),
                interval_width=None if lb is None else round(z_rmp - lb, 2),
                cert_seconds=round(out["wall"], 1), cert_nodes=out["nodes"],
            ))
            print(f"{inst_name}/{variant[:18]:<18}: z={z_rmp:.1f} cert={out['status'][:12]:<12} "
                  f"best_rc={best_rc if best_rc is None else round(best_rc,1)} certified={certified} "
                  f"width={'-' if lb is None else round(z_rmp-lb,2)}")
    df = pd.DataFrame(rows)
    GENDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GENDIR / f"anytime_cert_{family}.csv", index=False)
    print(f"\nwrote anytime_cert_{family}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
