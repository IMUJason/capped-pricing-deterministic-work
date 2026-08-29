"""GC study loop with instrumentation aligned to the VRPTW pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .gc_models import GCColumn, GCInstance, erdos_renyi, greedy_color_columns
from .gc_solver import gc_pricing, gc_reduced_cost, gc_solve_master


@dataclass
class GCVariant:
    name: str
    strategy: str            # single | naive_multi | diversified
    candidate_pool_size: int
    add_count: int
    pricing_time_limit: float
    pricing_threads: int
    max_iterations: int


def _pairwise_diversity(cols: list[GCColumn]) -> float:
    if len(cols) < 2:
        return 0.0
    tot = 0.0
    cnt = 0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i].vertices, cols[j].vertices
            union = len(a | b)
            tot += 1.0 - (len(a & b) / union if union else 1.0)
            cnt += 1
    return tot / cnt


def _select(cands: list[GCColumn], v: GCVariant) -> list[GCColumn]:
    neg = [c for c in cands if c.reduced_cost is not None and c.reduced_cost < -1e-6]
    if not neg:
        return []
    if v.strategy == "single":
        return neg[:1]
    if v.strategy == "naive_multi":
        return neg[: v.add_count]
    if v.strategy == "diversified":
        # greedy: best RC first, require Jaccard distance >= 0.5 from selected
        sel: list[GCColumn] = []
        for c in neg:
            if len(sel) >= v.add_count:
                break
            if all(1.0 - len(c.vertices & s.vertices) / max(len(c.vertices | s.vertices), 1) >= 0.5 for s in sel):
                sel.append(c)
        return sel or neg[:1]
    raise ValueError(v.strategy)


def run_gc_study(inst: GCInstance, v: GCVariant) -> dict:
    cols: dict[frozenset[int], GCColumn] = {c.vertices: c for c in greedy_color_columns(inst)}
    m0 = gc_solve_master(inst, list(cols.values()))
    iters = []
    prev_duals: dict[int, float] = {}
    statuses = []
    for t in range(1, v.max_iterations + 1):
        pool = list(cols.values())
        master = gc_solve_master(inst, pool)
        duals = master.duals or {}
        delta_dual = sum(abs(duals.get(x, 0.0) - prev_duals.get(x, 0.0)) for x in set(duals) | set(prev_duals)) / max(inst.n, 1)
        pr = gc_pricing(inst, duals, v.candidate_pool_size, v.pricing_time_limit, v.pricing_threads)
        statuses.append(pr.status)
        cands = pr.candidates
        selected = _select(cands, v)
        for c in selected:
            cols.setdefault(c.vertices, c)
        iters.append(dict(
            iteration=t, master_objective=master.objective_value,
            master_delta=master.objective_value - (iters[-1]["master_objective"] if iters else m0.objective_value),
            dual_l1_displacement=delta_dual,
            candidate_count=len(cands),
            selected_count=len(selected),
            candidate_diversity=_pairwise_diversity(cands[: v.candidate_pool_size]),
            selected_diversity=_pairwise_diversity(selected),
            best_reduced_cost=min([c.reduced_cost for c in cands] + [0.0]),
            pricing_runtime_seconds=pr.runtime, pricing_status=pr.status, pricing_gap=pr.mip_gap,
            pricing_dettime_ticks=pr.dettime_ticks, pricing_nodes_processed=pr.nodes,
            pricing_simplex_iterations=pr.simplex_iterations,
            selected_signatures=";".join(c.signature for c in selected),
        ))
        prev_duals = dict(duals)
        if not selected:
            break
    final = gc_solve_master(inst, list(cols.values()))
    final_bin = gc_solve_master(inst, list(cols.values()), binary=True)
    return dict(
        instance_name=inst.name, variant=v.name, n=inst.n, m_edges=len(inst.edges),
        source=inst.source, initial_lp=m0.objective_value, final_lp=final.objective_value,
        final_ip=final_bin.objective_value,
        lp_drop=m0.objective_value - final.objective_value,
        iterations_completed=len(iters),
        termination_reason=("no_negative_candidate" if iters and not iters[-1]["selected_count"] else "max_iterations"),
        pricing_statuses=statuses,
        iteration_rows=iters,
        final_lambda_values={k.replace("lam_", ""): val for k, val in final.lambda_values.items() if val > 1e-6},
    )


def main(cfg_path: str) -> None:
    import yaml
    cfg = yaml.safe_load(open(cfg_path))
    outdir = Path(cfg["out_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    for spec in cfg["instances"]:
        if spec["kind"] == "random":
            inst = erdos_renyi(spec["name"], spec["n"], spec["p"], spec["seed"])
        else:
            from .gc_models import parse_dimacs
            inst = parse_dimacs(Path(spec["path"]), spec["name"])
        for vv in cfg["variants"]:
            v = GCVariant(**vv)
            res = run_gc_study(inst, v)
            (outdir / f"{spec['name']}__{v.name}.json").write_text(json.dumps(res, indent=2))
            print(f"done {spec['name']} / {v.name}: lp {res['initial_lp']:.2f}->{res['final_lp']:.2f} "
                  f"ip={res['final_ip']:.0f} iters={res['iterations_completed']} term={res['termination_reason']}",
                  flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
