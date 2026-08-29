from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations

import cplex

from .cuts import subset_row_dual_contribution
from .models import RouteColumn, VRPTWInstance


@dataclass
class PricingResult:
    status: str
    candidates: list[RouteColumn]
    objective_value: float | None
    mip_gap: float | None
    runtime: float
    dettime_ticks: float | None = None
    nodes_processed: int | None = None
    simplex_iterations: int | None = None


def compute_reduced_cost(
    route: RouteColumn,
    dual_customers: dict[int, float],
    dual_vehicle: float,
    subset_row_cut_duals: dict[tuple[int, int, int], float] | None = None,
) -> float:
    dual_contribution = sum(dual_customers.get(customer_id, 0.0) for customer_id in route.customer_incidence)
    cut_dual_contribution = subset_row_dual_contribution(route, subset_row_cut_duals)
    return float(route.distance - dual_contribution - dual_vehicle - cut_dual_contribution)


def _customer_dual(duals: dict[int, float], customer_id: int) -> float:
    return duals.get(customer_id, 0.0)


def _build_candidate_route(instance: VRPTWInstance, end_node: int, arcs: dict[tuple[int, int], float], reduced_cost: float, source: str) -> RouteColumn:
    successor = {i: j for (i, j), value in arcs.items() if value > 0.5}
    route: list[int] = []
    current = 0
    seen = set()
    while current in successor:
        nxt = successor[current]
        if nxt == end_node:
            break
        if nxt in seen:
            raise ValueError(f"Detected a cycle while decoding route in {instance.name}")
        seen.add(nxt)
        route.append(nxt)
        current = nxt
    return instance.make_column(route, reduced_cost=reduced_cost, source=source)


def _map_status(status_string: str) -> str:
    text = status_string.lower()
    if "infeasible" in text:
        return "INFEASIBLE"
    if "time limit" in text or "timelimit" in text:
        return "TIME_LIMIT"
    if "populate" in text or "optimal" in text:
        return "OPTIMAL"
    return status_string


def generate_candidate_routes(
    instance: VRPTWInstance,
    dual_customers: dict[int, float],
    dual_vehicle: float,
    pool_solutions: int,
    time_limit: float,
    threads: int,
    subset_row_cut_duals: dict[tuple[int, int, int], float] | None = None,
    pool_intensity: int = 2,
    tighten_big_m: bool = True,
) -> PricingResult:
    model = cplex.Cplex()
    model.set_log_stream(None)
    model.set_error_stream(None)
    model.set_warning_stream(None)
    model.set_results_stream(None)
    model.objective.set_sense(model.objective.sense.minimize)

    model.parameters.timelimit.set(time_limit)
    model.parameters.threads.set(max(1, threads))
    model.parameters.mip.pool.intensity.set(pool_intensity)
    model.parameters.mip.pool.capacity.set(max(pool_solutions, 1))
    model.parameters.mip.limits.populate.set(max(pool_solutions, 1))

    customer_ids = instance.customer_ids
    end_node = max(customer_ids) + 1

    def edge_weight(i: int, j: int) -> int:
        return instance.edge_weight(i, 0 if j == end_node else j)

    def service_time(i: int) -> int:
        return 0 if i == 0 else instance.customers[i].service_time

    def ready_time(node: int) -> int:
        if node in (0, end_node):
            return instance.depot.ready_time
        return instance.customers[node].ready_time

    def due_date(node: int) -> float:
        if node in (0, end_node):
            return float(instance.depot.due_date)
        return float(instance.customers[node].due_date)

    arcs = []
    outgoing = {}
    incoming = {}
    for i in (0,) + customer_ids:
        for j in customer_ids + (end_node,):
            if i == j or j == 0:
                continue
            if i == 0 and j == end_node:
                continue
            earliest_departure = instance.depot.ready_time if i == 0 else instance.customers[i].ready_time
            limit = instance.depot.due_date if j == end_node else due_date(j)
            if earliest_departure + service_time(i) + edge_weight(i, j) > limit:
                continue
            # 容量冲突对剔除：d_i + d_j > Q 的客户不可能同车，弧永不可行
            if tighten_big_m and i != 0 and j != end_node:
                if instance.customers[i].demand + instance.customers[j].demand > instance.capacity:
                    continue
            arcs.append((i, j))
            outgoing.setdefault(i, []).append(j)
            incoming.setdefault(j, []).append(i)

    var_names: list[str] = []
    var_obj: list[float] = []
    var_lb: list[float] = []
    var_ub: list[float] = []
    var_types: list[str] = []

    for i, j in arcs:
        var_names.append(f"x_{i}_{j}")
        var_obj.append(float(edge_weight(i, j)))
        var_lb.append(0.0)
        var_ub.append(1.0)
        var_types.append("B")

    for customer_id in customer_ids:
        var_names.append(f"y_{customer_id}")
        var_obj.append(-_customer_dual(dual_customers, customer_id))
        var_lb.append(0.0)
        var_ub.append(1.0)
        var_types.append("B")

    time_nodes = (0,) + customer_ids + (end_node,)
    for node in time_nodes:
        var_names.append(f"t_{node}")
        var_obj.append(0.0)
        # 引擎修复：tighten 模式下 t_j 下界提到 ready_j，配合逐弧 big-M 才严格成立
        var_lb.append(float(ready_time(node)) if tighten_big_m else 0.0)
        var_ub.append(due_date(node))
        var_types.append("C")

    for customer_id in customer_ids:
        var_names.append(f"load_{customer_id}")
        var_obj.append(0.0)
        var_lb.append(0.0)
        var_ub.append(float(instance.capacity))
        var_types.append("C")

    subset_row_keys = [customers for customers, dual_value in (subset_row_cut_duals or {}).items() if abs(dual_value) > 1e-9]
    for customers in subset_row_keys:
        var_names.append(f"subset_row_{'_'.join(map(str, customers))}")
        var_obj.append(-(subset_row_cut_duals or {})[customers])
        var_lb.append(0.0)
        var_ub.append(1.0)
        var_types.append("B")

    model.variables.add(obj=var_obj, lb=var_lb, ub=var_ub, types="".join(var_types), names=var_names)
    var_index = {name: idx for idx, name in enumerate(var_names)}
    x_index = {arc: var_index[f"x_{arc[0]}_{arc[1]}"] for arc in arcs}

    rows: list[cplex.SparsePair] = []
    senses: list[str] = []
    rhs_values: list[float] = []
    row_names: list[str] = []

    def add_row(coefficients: dict[str, float], sense: str, rhs: float, name: str) -> None:
        rows.append(cplex.SparsePair(ind=[var_index[var] for var in coefficients], val=list(coefficients.values())))
        senses.append(sense)
        rhs_values.append(float(rhs))
        row_names.append(name)

    add_row({f"x_0_{j}": 1.0 for j in outgoing[0]}, "E", 1.0, "depart_depot")
    add_row({f"x_{i}_{end_node}": 1.0 for i in incoming[end_node]}, "E", 1.0, "return_depot")
    add_row({"t_0": 1.0}, "E", float(instance.depot.ready_time), "depot_time")

    for customer_id in customer_ids:
        add_row(
            {**{f"x_{i}_{customer_id}": 1.0 for i in incoming.get(customer_id, [])}, f"y_{customer_id}": -1.0},
            "E", 0.0, f"in_{customer_id}",
        )
        add_row(
            {**{f"x_{customer_id}_{j}": 1.0 for j in outgoing.get(customer_id, [])}, f"y_{customer_id}": -1.0},
            "E", 0.0, f"out_{customer_id}",
        )
        add_row({f"t_{customer_id}": 1.0, f"y_{customer_id}": -float(instance.customers[customer_id].ready_time)}, "G", 0.0, f"ready_{customer_id}")
        add_row({f"load_{customer_id}": 1.0, f"y_{customer_id}": -float(instance.customers[customer_id].demand)}, "G", 0.0, f"load_lb_{customer_id}")
        add_row({f"load_{customer_id}": 1.0, f"y_{customer_id}": -float(instance.capacity)}, "L", 0.0, f"load_ub_{customer_id}")

    big_m_time = instance.depot.due_date + max(instance.edge_weight(i, j) for i in (0,) + customer_ids for j in customer_ids + (0,) if i != j) + max(
        [instance.customers[customer_id].service_time for customer_id in customer_ids] + [0]
    )

    def big_m(i: int, j: int) -> float:
        # 逐弧最紧 big-M：M_ij = due_i + s_i + t_ij - ready_j（引擎修复核心）
        if not tighten_big_m:
            return float(big_m_time)
        if i == 0:
            return max(float(edge_weight(i, j) - ready_time(j)), 1.0)
        ready_j = float(instance.depot.ready_time) if j == end_node else float(ready_time(j))
        return max(float(due_date(i)) + float(service_time(i)) + float(edge_weight(i, j)) - ready_j, 1.0)

    for i, j in arcs:
        travel = edge_weight(i, j)
        xname = f"x_{i}_{j}"
        arc_m = big_m(i, j)
        if j == end_node:
            add_row({f"t_{end_node}": 1.0, f"t_{i}": -1.0, xname: -arc_m}, "G", float(service_time(i) + travel) - arc_m, f"time_{i}_{j}")
        elif i == 0:
            add_row({f"t_{j}": 1.0, xname: -arc_m}, "G", float(travel) - arc_m, f"time_{i}_{j}")
            add_row({f"load_{j}": 1.0, xname: -float(instance.capacity)}, "G", float(instance.customers[j].demand - instance.capacity), f"load_{i}_{j}")
        else:
            add_row({f"t_{j}": 1.0, f"t_{i}": -1.0, xname: -arc_m}, "G", float(service_time(i) + travel) - arc_m, f"time_{i}_{j}")
            add_row({f"load_{j}": 1.0, f"load_{i}": -1.0, xname: -float(instance.capacity)}, "G", float(instance.customers[j].demand - instance.capacity), f"load_{i}_{j}")

    for customers in subset_row_keys:
        zname = f"subset_row_{'_'.join(map(str, customers))}"
        add_row({**{f"y_{customer_id}": 1.0 for customer_id in customers}, zname: -2.0}, "G", 0.0, f"subset_row_ub_{'_'.join(map(str, customers))}")
        for left, right in combinations(customers, 2):
            add_row({zname: 1.0, f"y_{left}": -1.0, f"y_{right}": -1.0}, "G", -1.0, f"subset_row_lb_{left}_{right}_{customers[-1]}")

    model.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs_values, names=row_names)

    start_wall = time.perf_counter()
    start_dettime = model.get_dettime()
    model.populate_solution_pool()
    runtime = time.perf_counter() - start_wall
    dettime_ticks = model.get_dettime() - start_dettime

    status = _map_status(model.solution.get_status_string())

    pool = model.solution.pool
    num_solutions = pool.get_num()
    if num_solutions == 0:
        return PricingResult(status=status, candidates=[], objective_value=None, mip_gap=None, runtime=runtime, dettime_ticks=dettime_ticks)

    candidates: list[RouteColumn] = []
    seen_routes = set()
    for solution_number in range(num_solutions):
        values = pool.get_values(solution_number)
        arc_values = {arc: values[idx] for arc, idx in x_index.items()}
        route = _build_candidate_route(instance, end_node, arc_values, reduced_cost=0.0, source="pricing_pool")
        route = instance.make_column(route.route, reduced_cost=0.0, source="pricing_pool")
        reduced_cost = compute_reduced_cost(
            route,
            dual_customers=dual_customers,
            dual_vehicle=dual_vehicle,
            subset_row_cut_duals=subset_row_cut_duals,
        )
        route = instance.make_column(route.route, reduced_cost=reduced_cost, source="pricing_pool")
        if route.route and route.route not in seen_routes:
            seen_routes.add(route.route)
            candidates.append(route)

    objective_value = None
    mip_gap = None
    try:
        objective_value = float(model.solution.get_objective_value()) - float(dual_vehicle)
        mip_gap = float(model.solution.MIP.get_mip_relative_gap())
    except cplex.exceptions.CplexSolverError:
        pass

    nodes_processed = None
    simplex_iterations = None
    try:
        nodes_processed = int(model.solution.progress.get_num_nodes_processed())
        simplex_iterations = int(model.solution.progress.get_num_iterations())
    except cplex.exceptions.CplexSolverError:
        pass

    candidates.sort(key=lambda column: column.reduced_cost if column.reduced_cost is not None else float("inf"))
    return PricingResult(
        status=status,
        candidates=candidates,
        objective_value=objective_value,
        mip_gap=mip_gap,
        runtime=runtime,
        dettime_ticks=dettime_ticks,
        nodes_processed=nodes_processed,
        simplex_iterations=simplex_iterations,
    )
