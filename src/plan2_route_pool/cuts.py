from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .models import RouteColumn


@dataclass(frozen=True)
class SubsetRowCut:
    customers: tuple[int, int, int]
    rhs: int = 1

    @property
    def signature(self) -> str:
        return "__".join(map(str, self.customers))


def subset_row_coeff_from_customers(customer_incidence: frozenset[int], customers: tuple[int, int, int]) -> int:
    return int(len(customer_incidence.intersection(customers)) >= 2)


def subset_row_coeff(route: RouteColumn, cut: SubsetRowCut) -> int:
    return subset_row_coeff_from_customers(route.customer_incidence, cut.customers)


def subset_row_dual_contribution(route: RouteColumn, cut_duals: dict[tuple[int, int, int], float] | None) -> float:
    if not cut_duals:
        return 0.0
    return float(sum(dual * subset_row_coeff_from_customers(route.customer_incidence, customers) for customers, dual in cut_duals.items()))


def customer_ambiguity_scores(columns: list[RouteColumn], lambda_values: dict[str, float], customer_ids: tuple[int, ...]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for customer_id in customer_ids:
        weights = [max(0.0, lambda_values.get(column.signature, 0.0)) for column in columns if customer_id in column.customer_incidence]
        positive_weights = [weight for weight in weights if weight > 1e-8]
        if not positive_weights:
            scores[customer_id] = 0.0
            continue
        total = sum(positive_weights)
        normalized = [weight / total for weight in positive_weights]
        scores[customer_id] = 1.0 - sum(weight * weight for weight in normalized)
    return scores


def route_ambiguity_gain(route: RouteColumn, customer_scores: dict[int, float] | None) -> float:
    if not customer_scores or not route.customer_incidence:
        return 0.0
    values = [customer_scores.get(customer_id, 0.0) for customer_id in route.customer_incidence]
    return float(sum(values) / len(values)) if values else 0.0


def separate_subset_row_cuts(
    columns: list[RouteColumn],
    lambda_values: dict[str, float],
    customer_ids: tuple[int, ...],
    existing_signatures: set[str],
    candidate_customer_pool: int,
    max_new_cuts: int,
    violation_tol: float = 1e-6,
) -> list[SubsetRowCut]:
    if max_new_cuts <= 0 or candidate_customer_pool < 3:
        return []

    ambiguity_scores = customer_ambiguity_scores(columns, lambda_values, customer_ids)
    ranked_customers = sorted(customer_ids, key=lambda customer_id: (ambiguity_scores.get(customer_id, 0.0), customer_id), reverse=True)
    candidate_customers = ranked_customers[:candidate_customer_pool]
    if len(candidate_customers) < 3:
        return []

    violated: list[tuple[float, SubsetRowCut]] = []
    for triplet in combinations(sorted(candidate_customers), 3):
        cut = SubsetRowCut(customers=triplet)
        if cut.signature in existing_signatures:
            continue
        lhs = sum(subset_row_coeff(column, cut) * lambda_values.get(column.signature, 0.0) for column in columns)
        violation = lhs - float(cut.rhs)
        if violation > violation_tol:
            violated.append((violation, cut))

    violated.sort(key=lambda item: (-item[0], item[1].customers))
    return [cut for _, cut in violated[:max_new_cuts]]
