from __future__ import annotations

from itertools import combinations

from .cuts import route_ambiguity_gain
from .models import RouteColumn


def jaccard_distance(left: frozenset, right: frozenset) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def route_diversity(left: RouteColumn, right: RouteColumn, arc_weight: float = 0.6) -> float:
    arc_distance = jaccard_distance(left.arc_incidence, right.arc_incidence)
    customer_distance = jaccard_distance(left.customer_incidence, right.customer_incidence)
    return arc_weight * arc_distance + (1.0 - arc_weight) * customer_distance


def average_pairwise_diversity(columns: list[RouteColumn]) -> float:
    if len(columns) < 2:
        return 0.0
    values = [route_diversity(left, right) for left, right in combinations(columns, 2)]
    return sum(values) / len(values)


def select_diversified_columns(candidates: list[RouteColumn], add_count: int, cost_weight: float = 0.6) -> list[RouteColumn]:
    if add_count <= 0 or not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda column: column.reduced_cost if column.reduced_cost is not None else float("inf"))
    selected = [sorted_candidates[0]]
    remaining = sorted_candidates[1:]

    max_gain = max([max(0.0, -(candidate.reduced_cost or 0.0)) for candidate in sorted_candidates] + [1.0])

    while remaining and len(selected) < add_count:
        scored = []
        for candidate in remaining:
            gain = max(0.0, -(candidate.reduced_cost or 0.0)) / max_gain
            diversity_gain = min(route_diversity(candidate, chosen) for chosen in selected)
            score = cost_weight * gain + (1.0 - cost_weight) * diversity_gain
            scored.append((score, candidate))
        scored.sort(key=lambda item: (-item[0], item[1].reduced_cost if item[1].reduced_cost is not None else float("inf")))
        best = scored[0][1]
        selected.append(best)
        remaining = [candidate for candidate in remaining if candidate.route != best.route]
    return selected


def select_adaptive_columns(
    candidates: list[RouteColumn],
    add_count: int,
    reduced_cost_weight: float,
    diversity_weight: float,
    ambiguity_weight: float,
    customer_scores: dict[int, float] | None = None,
) -> list[RouteColumn]:
    if add_count <= 0 or not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda column: column.reduced_cost if column.reduced_cost is not None else float("inf"))
    selected = [sorted_candidates[0]]
    remaining = sorted_candidates[1:]
    max_gain = max([max(0.0, -(candidate.reduced_cost or 0.0)) for candidate in sorted_candidates] + [1.0])

    while remaining and len(selected) < add_count:
        scored = []
        for candidate in remaining:
            gain = max(0.0, -(candidate.reduced_cost or 0.0)) / max_gain
            diversity_gain = min(route_diversity(candidate, chosen) for chosen in selected)
            ambiguity_gain = route_ambiguity_gain(candidate, customer_scores)
            score = reduced_cost_weight * gain + diversity_weight * diversity_gain + ambiguity_weight * ambiguity_gain
            scored.append((score, candidate))
        scored.sort(key=lambda item: (-item[0], item[1].reduced_cost if item[1].reduced_cost is not None else float("inf")))
        best = scored[0][1]
        selected.append(best)
        remaining = [candidate for candidate in remaining if candidate.route != best.route]
    return selected
