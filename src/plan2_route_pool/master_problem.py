from __future__ import annotations

from dataclasses import dataclass

import cplex

from .cuts import SubsetRowCut, subset_row_coeff
from .models import RouteColumn, VRPTWInstance


@dataclass
class MasterResult:
    objective_value: float
    lambda_values: dict[str, float]
    dual_customers: dict[int, float] | None
    dual_vehicle: float | None
    dual_subset_row_cuts: dict[tuple[int, int, int], float] | None
    status: str


def _map_status(status_string: str) -> str:
    text = status_string.lower()
    if "infeasible" in text:
        return "INFEASIBLE"
    if "time limit" in text or "timelimit" in text:
        return "TIME_LIMIT"
    if "optimal" in text:
        return "OPTIMAL"
    return status_string


def solve_master(
    instance: VRPTWInstance,
    columns: list[RouteColumn],
    binary: bool = False,
    subset_row_cuts: list[SubsetRowCut] | None = None,
) -> MasterResult:
    model = cplex.Cplex()
    model.set_log_stream(None)
    model.set_error_stream(None)
    model.set_warning_stream(None)
    model.set_results_stream(None)
    model.objective.set_sense(model.objective.sense.minimize)

    var_names = [f"lambda_{column.signature}" for column in columns]
    obj = [float(column.distance) for column in columns]
    if binary:
        model.variables.add(obj=obj, lb=[0.0] * len(columns), ub=[1.0] * len(columns), types="B" * len(columns), names=var_names)
    else:
        # 不传 types：CPLEX 一旦带类型向量即按 MIP 处理，对偶不可用
        model.variables.add(obj=obj, lb=[0.0] * len(columns), ub=[cplex.infinity] * len(columns), names=var_names)

    cover_names: list[str] = []
    rows = []
    senses = []
    rhs_values = []
    row_names = []

    for customer_id in instance.customer_ids:
        indices = [idx for idx, column in enumerate(columns) if customer_id in column.customer_incidence]
        rows.append(cplex.SparsePair(ind=indices, val=[1.0] * len(indices)))
        senses.append("E")
        rhs_values.append(1.0)
        name = f"cover_{customer_id}"
        row_names.append(name)
        cover_names.append(name)

    vehicle_name = "vehicle_limit"
    rows.append(cplex.SparsePair(ind=list(range(len(columns))), val=[1.0] * len(columns)))
    senses.append("L")
    rhs_values.append(float(instance.max_vehicles))
    row_names.append(vehicle_name)

    cut_names: dict[tuple[int, int, int], str] = {}
    for cut in subset_row_cuts or []:
        coefficients = [subset_row_coeff(column, cut) for column in columns]
        if not any(coefficients):
            continue
        rows.append(cplex.SparsePair(ind=list(range(len(columns))), val=[float(v) for v in coefficients]))
        senses.append("L")
        rhs_values.append(float(cut.rhs))
        name = f"subset_row_{cut.signature}"
        row_names.append(name)
        cut_names[cut.customers] = name

    model.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs_values, names=row_names)

    model.solve()

    status = _map_status(model.solution.get_status_string())
    values = model.solution.get_values()
    lambda_values = {var_names[idx]: float(values[idx]) for idx in range(len(columns))}

    if binary:
        return MasterResult(float(model.solution.get_objective_value()), lambda_values, None, None, None, status)

    dual_cover = model.solution.get_dual_values(cover_names)
    dual_customers = {customer_id: float(value) for customer_id, value in zip(instance.customer_ids, dual_cover)}
    dual_vehicle = float(model.solution.get_dual_values([vehicle_name])[0])
    if cut_names:
        dual_cut_values = model.solution.get_dual_values(list(cut_names.values()))
        dual_subset_row_cuts = {customers: float(value) for customers, value in zip(cut_names.keys(), dual_cut_values)}
    else:
        dual_subset_row_cuts = {}
    return MasterResult(float(model.solution.get_objective_value()), lambda_values, dual_customers, dual_vehicle, dual_subset_row_cuts, status)
