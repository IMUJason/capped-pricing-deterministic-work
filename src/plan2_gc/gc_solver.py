"""Covering master and MWIS pricing for graph coloring CG (CPLEX backend)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cplex

from .gc_models import GCColumn, GCInstance


@dataclass
class GCMasterResult:
    objective_value: float
    lambda_values: dict[str, float]
    duals: dict[int, float] | None
    status: str


def gc_solve_master(inst: GCInstance, columns: list[GCColumn], binary: bool = False) -> GCMasterResult:
    m = cplex.Cplex()
    for s in ("log", "error", "warning", "results"):
        getattr(m, f"set_{s}_stream")(None)
    m.objective.set_sense(m.objective.sense.minimize)
    names = [f"lam_{c.signature}" for c in columns]
    obj = [float(c.cost) for c in columns]
    if binary:
        m.variables.add(obj=obj, lb=[0.0] * len(columns), ub=[1.0] * len(columns),
                        types="B" * len(columns), names=names)
    else:
        m.variables.add(obj=obj, lb=[0.0] * len(columns), names=names)
    rows, senses, rhs, rnames = [], [], [], []
    for v in range(inst.n):
        idx = [i for i, c in enumerate(columns) if v in c.vertices]
        if not idx:
            continue
        rows.append(cplex.SparsePair(ind=idx, val=[1.0] * len(idx)))
        senses.append("G")
        rhs.append(1.0)
        rnames.append(f"cover_{v}")
    m.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs, names=rnames)
    m.solve()
    status = "OPTIMAL" if "optimal" in m.solution.get_status_string().lower() else m.solution.get_status_string()
    vals = m.solution.get_values()
    lam = {names[i]: float(vals[i]) for i in range(len(columns))}
    duals = None
    if not binary:
        d = m.solution.get_dual_values(rnames)
        duals = {v: float(x) for v, x in zip([int(r.split("_")[1]) for r in rnames], d)}
    return GCMasterResult(float(m.solution.get_objective_value()), lam, duals, status)


def gc_reduced_cost(col: GCColumn, duals: dict[int, float]) -> float:
    return float(col.cost - sum(duals.get(v, 0.0) for v in col.vertices))


@dataclass
class GCPricingResult:
    status: str
    candidates: list[GCColumn]
    objective_value: float | None
    mip_gap: float | None
    runtime: float
    dettime_ticks: float | None = None
    nodes: int | None = None
    simplex_iterations: int | None = None


def gc_pricing(inst: GCInstance, duals: dict[int, float], pool_solutions: int,
               time_limit: float, threads: int, pool_intensity: int = 2) -> GCPricingResult:
    """MWIS pricing: minimize 1 - sum(pi_v y_v) over stable sets; populate pool."""
    m = cplex.Cplex()
    for s in ("log", "error", "warning", "results"):
        getattr(m, f"set_{s}_stream")(None)
    m.objective.set_sense(m.objective.sense.minimize)
    m.parameters.timelimit.set(time_limit)
    m.parameters.threads.set(max(1, threads))
    m.parameters.mip.pool.intensity.set(pool_intensity)
    m.parameters.mip.pool.capacity.set(max(pool_solutions, 1))
    m.parameters.mip.limits.populate.set(max(pool_solutions, 1))

    names = [f"y_{v}" for v in range(inst.n)]
    obj = [-float(duals.get(v, 0.0)) for v in range(inst.n)]
    m.variables.add(obj=obj, lb=[0.0] * inst.n, ub=[1.0] * inst.n, types="B" * inst.n, names=names)

    rows = [cplex.SparsePair(ind=[u, v], val=[1.0, 1.0]) for u, v in inst.edges]
    senses = ["L"] * len(inst.edges)
    rhs = [1.0] * len(inst.edges)
    m.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs,
                             names=[f"e_{u}_{v}" for u, v in inst.edges])

    t0 = time.perf_counter()
    d0 = m.get_dettime()
    m.populate_solution_pool()
    runtime = time.perf_counter() - t0
    ticks = m.get_dettime() - d0

    stext = m.solution.get_status_string().lower()
    status = ("OPTIMAL" if ("optimal" in stext or "populate" in stext) else
              "TIME_LIMIT" if ("time limit" in stext or "timelimit" in stext) else stext)
    pool = m.solution.pool
    k = pool.get_num()
    if k == 0:
        return GCPricingResult(status, [], None, None, runtime, ticks)

    cands: list[GCColumn] = []
    seen = set()
    for s in range(k):
        vals = pool.get_values(s)
        S = frozenset(v for v in range(inst.n) if vals[v] > 0.5)
        if not S or S in seen:
            continue
        seen.add(S)
        if not inst.is_stable(S):
            continue
        col = GCColumn(S, reduced_cost=gc_reduced_cost(GCColumn(S), duals))
        cands.append(col)
    cands.sort(key=lambda c: c.reduced_cost)
    obj_val = None
    gap = None
    try:
        obj_val = float(m.solution.get_objective_value()) + 1.0  # +1 constant
        gap = float(m.solution.MIP.get_mip_relative_gap())
    except cplex.exceptions.CplexSolverError:
        pass
    return GCPricingResult(status, cands, obj_val, gap, runtime, ticks,
                           m.solution.progress.get_num_nodes_processed(),
                           m.solution.progress.get_num_iterations())
