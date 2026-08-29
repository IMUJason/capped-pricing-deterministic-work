"""Post-hoc certification: rebuild final column pool from an M1 raw run,
re-solve the master LP for raw duals, and run one final exact pricing call
(long limit) to (in)validate the root bound with a certificate.

Usage: python analysis/certify_runs.py <instance> <variant> [cert_time_limit]
Output: paper/ejor_submission/generated/certification_<instance>_<variant>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
PLAN2 = PKG.parents[1]
sys.path.insert(0, str(PKG / "src"))

from plan2_route_pool.vrptw_parser import parse_solomon_like_instance
from plan2_route_pool.initializer import build_feasible_seed_routes
from plan2_route_pool.master_problem import solve_master
from plan2_route_pool.pricing import generate_candidate_routes

import pandas as pd


def main() -> None:
    instance_name = sys.argv[1]
    variant = sys.argv[2]
    cert_limit = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0

    bks = pd.read_csv(PLAN2 / "legacy/reproducibility_package/data/provenance/vrptw_dimacs_bks.csv")
    row = bks[bks["instance_name"] == instance_name].iloc[0]
    pkg = PLAN2 / "legacy/reproducibility_package"
    instance_path = pkg / "data" / "raw" / "vrptw" / "controller" / "VRPTWController-master" / row["relative_path"]
    instance = parse_solomon_like_instance(path=instance_path, source_family=str(row["family"]),
                                           benchmark_distance=float(row["best_known_solution"]),
                                           benchmark_optimal=bool(row["is_optimal"]))

    run = None
    for study_dir in (PLAN2 / "results" / "raw").glob("m1_*"):
        f = study_dir / f"{instance_name}__{variant}.json"
        if f.exists():
            run = json.loads(f.read_text())
            break
    if run is None:
        raise FileNotFoundError(f"no raw run for {instance_name}/{variant}")

    # rebuild final column pool: seeds + every admitted column (order-free, dedup by signature)
    columns = {}
    for col in build_feasible_seed_routes(instance):
        columns[col.signature] = col
    for it in run["iteration_rows"]:
        for sig in (s for s in it.get("selected_signatures", "").split(";") if s):
            if sig not in columns:
                route = [int(x) for x in sig.split("-")]
                columns[sig] = instance.make_column(route, reduced_cost=None, source="rebuilt")
    pool = list(columns.values())
    print(f"{instance_name}/{variant}: rebuilt pool of {len(pool)} columns")

    final_lp = solve_master(instance, pool, binary=False, subset_row_cuts=[])
    cert = generate_candidate_routes(
        instance=instance,
        dual_customers=final_lp.dual_customers,
        dual_vehicle=final_lp.dual_vehicle,
        pool_solutions=20,
        time_limit=cert_limit,
        threads=4,
        subset_row_cut_duals=None,
    )
    best_rc = cert.candidates[0].reduced_cost if cert.candidates else None
    certified = cert.status == "OPTIMAL" and (best_rc is None or best_rc >= -1e-6)
    out = dict(
        instance_name=instance_name, variant=variant, cert_time_limit=cert_limit,
        final_lp_objective=final_lp.objective_value,
        bks_distance=instance.benchmark_distance,
        certificate_status=cert.status, certificate_best_rc=best_rc,
        certified=certified,
        cert_runtime_seconds=cert.runtime, cert_detticks=cert.dettime_ticks,
        cert_nodes=cert.nodes_processed, cert_mip_gap=cert.mip_gap,
    )
    outdir = PLAN2 / "paper" / "ejor_submission" / "generated"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"certification_{instance_name}_{variant}.json"
    outfile.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
