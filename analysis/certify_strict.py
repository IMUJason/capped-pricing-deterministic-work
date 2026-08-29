"""Standalone certificate solver: strict MIP optimality proof via model.solve(), no pool.

Independent of study.py/pricing.py code paths: builds the ESPPRC MIP from scratch,
solves to optimality (no populate, no pool), and reports the minimum reduced cost
under given duals plus proof time. Used to adjudicate populate-vs-puresolve discrepancy.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cplex

from plan2_route_pool.vrptw_parser import parse_solomon_like_instance
from plan2_route_pool.initializer import build_feasible_seed_routes
from plan2_route_pool.master_problem import solve_master


def build_and_solve(inst, dual_customers, dual_vehicle, time_limit, threads=4):
    m = cplex.Cplex()
    for s in ("log", "error", "warning", "results"):
        getattr(m, f"set_{s}_stream")(None)
    m.objective.set_sense(m.objective.sense.minimize)
    m.parameters.timelimit.set(time_limit)
    m.parameters.threads.set(threads)

    cids = inst.customer_ids
    end = max(cids) + 1
    ew = lambda i, j: inst.edge_weight(i, 0 if j == end else j)
    st = lambda i: 0 if i == 0 else inst.customers[i].service_time
    rt = lambda n: inst.depot.ready_time if n in (0, end) else inst.customers[n].ready_time
    dd = lambda n: float(inst.depot.due_date if n in (0, end) else inst.customers[n].due_date)

    arcs = []
    for i in (0,) + cids:
        for j in cids + (end,):
            if i == j or j == 0 or (i == 0 and j == end):
                continue
            ed = inst.depot.ready_time if i == 0 else inst.customers[i].ready_time
            lim = inst.depot.due_date if j == end else dd(j)
            if ed + st(i) + ew(i, j) > lim:
                continue
            if i != 0 and j != end and inst.customers[i].demand + inst.customers[j].demand > inst.capacity:
                continue
            arcs.append((i, j))

    names, obj, lb, ub, types = [], [], [], [], []
    for i, j in arcs:
        names.append(f"x_{i}_{j}"); obj.append(float(ew(i, j))); lb.append(0.0); ub.append(1.0); types.append("B")
    for c in cids:
        names.append(f"y_{c}"); obj.append(-dual_customers.get(c, 0.0)); lb.append(0.0); ub.append(1.0); types.append("B")
    for n in (0,) + cids + (end,):
        names.append(f"t_{n}"); obj.append(0.0); lb.append(float(rt(n))); ub.append(dd(n)); types.append("C")
    for c in cids:
        names.append(f"l_{c}"); obj.append(0.0); lb.append(0.0); ub.append(float(inst.capacity)); types.append("C")
    m.variables.add(obj=obj, lb=lb, ub=ub, types="".join(types), names=names)
    idx = {n: k for k, n in enumerate(names)}

    rows, senses, rhs, rnames = [], [], [], []

    def add(coefs, sense, r, name):
        rows.append(cplex.SparsePair(ind=[idx[k] for k in coefs], val=list(coefs.values())))
        senses.append(sense); rhs.append(float(r)); rnames.append(name)

    add({f"x_0_{j}": 1 for j in dict.fromkeys(j for i, j in arcs if i == 0)}, "E", 1, "dep")
    add({f"x_{i}_{end}": 1 for i in dict.fromkeys(i for i, j in arcs if j == end)}, "E", 1, "ret")
    add({"t_0": 1}, "E", float(inst.depot.ready_time), "t0")
    for c in cids:
        add({**{f"x_{i}_{c}": 1 for i in dict.fromkeys(i for i, j in arcs if j == c)}, f"y_{c}": -1}, "E", 0, f"in{c}")
        add({**{f"x_{c}_{j}": 1 for j in dict.fromkeys(j for i, j in arcs if i == c)}, f"y_{c}": -1}, "E", 0, f"out{c}")
        add({f"t_{c}": 1, f"y_{c}": -float(inst.customers[c].ready_time)}, "G", 0, f"r{c}")
        add({f"l_{c}": 1, f"y_{c}": -float(inst.customers[c].demand)}, "G", 0, f"dl{c}")
        add({f"l_{c}": 1, f"y_{c}": -float(inst.capacity)}, "L", 0, f"du{c}")
    for i, j in arcs:
        M = max(float(dd(i)) + st(i) + ew(i, j) - float(rt(j)), 1.0) if i != 0 else max(float(ew(i, j) - rt(j)), 1.0)
        if i == 0:
            add({f"t_{j}": 1, f"x_0_{j}": -M}, "G", float(ew(i, j)) - M, f"t{i}_{j}")
            add({f"l_{j}": 1, f"x_0_{j}": -float(inst.capacity)}, "G", float(inst.customers[j].demand - inst.capacity), f"L{i}_{j}")
        else:
            add({f"t_{j}": 1, f"t_{i}": -1, f"x_{i}_{j}": -M}, "G", float(st(i) + ew(i, j)) - M, f"t{i}_{j}")
            if j != end:
                add({f"l_{j}": 1, f"l_{i}": -1, f"x_{i}_{j}": -float(inst.capacity)}, "G",
                    float(inst.customers[j].demand - inst.capacity), f"L{i}_{j}")
    m.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs, names=rnames)

    t0 = time.perf_counter()
    m.solve()
    wall = time.perf_counter() - t0
    status = m.solution.get_status_string()
    objval = None
    best_rc = None
    if m.solution.get_solution_type() != m.solution.type.none:
        objval = m.solution.get_objective_value() - dual_vehicle
        best_rc = objval  # min over routes of c_r - pi a_r (route-level)
    return dict(status=status, wall=wall, best_rc=best_rc,
                nodes=m.solution.progress.get_num_nodes_processed(),
                ticks=m.get_dettime(), gap=m.solution.MIP.get_mip_relative_gap())


if __name__ == "__main__":
    import json
    plan2 = Path(__file__).resolve().parents[3]
    inst_name, variant, limit = sys.argv[1], sys.argv[2], float(sys.argv[3] if len(sys.argv) > 3 else 1800.0)
    r = None
    for s in ("m1_c", "m1_r1", "m1_r2", "m1_rc"):
        f = plan2 / "results" / "raw" / s / f"{inst_name}__{variant}.json"
        if f.exists():
            r = json.loads(f.read_text()); break
    base = Path(__file__).resolve().parents[1]
    row = None
    import csv
    for x in csv.DictReader(open(base / "data/provenance/vrptw_dimacs_bks.csv")):
        if x["instance_name"] == inst_name: row = x; break
    inst = parse_solomon_like_instance(base / "data/raw/vrptw/controller/VRPTWController-master" / row["relative_path"],
                                       source_family=row["family"])
    cols = {c.signature: c for c in build_feasible_seed_routes(inst)}
    for it in r["iteration_rows"]:
        for sig in (s for s in it.get("selected_signatures", "").split(";") if s):
            if sig not in cols:
                cols[sig] = inst.make_column([int(x) for x in sig.split("-")], None, "rebuilt")
    m = solve_master(inst, list(cols.values()), binary=False)
    out = build_and_solve(inst, m.dual_customers, m.dual_vehicle, limit)
    print(json.dumps(dict(lp=m.objective_value, **out), indent=2))
