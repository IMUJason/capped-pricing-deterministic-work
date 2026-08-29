from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, sqrt
from typing import Iterable


def round_half_up(value: float) -> int:
    return int(floor(value + 0.5))


@dataclass(frozen=True)
class VRPTWCustomer:
    customer_id: int
    x: float
    y: float
    demand: int
    ready_time: int
    due_date: int
    service_time: int


@dataclass(frozen=True)
class RouteEvaluation:
    feasible: bool
    distance: float
    arrival_times: tuple[float, ...] = ()
    load_profile: tuple[int, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class RouteColumn:
    route: tuple[int, ...]
    distance: float
    reduced_cost: float | None
    customer_incidence: frozenset[int]
    arc_incidence: frozenset[tuple[int, int]]
    source: str = "unknown"

    @property
    def signature(self) -> str:
        return "-".join(map(str, self.route))


@dataclass
class VRPTWInstance:
    name: str
    max_vehicles: int
    capacity: int
    depot: VRPTWCustomer
    customers: dict[int, VRPTWCustomer]
    source_family: str = "unknown"
    source_path: str = ""
    benchmark_distance: float | None = None
    benchmark_optimal: bool | None = None

    _distance_cache: dict[tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)

    @property
    def customer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.customers))

    @property
    def dimension(self) -> int:
        return len(self.customers) + 1

    def node(self, node_id: int) -> VRPTWCustomer:
        if node_id == 0:
            return self.depot
        return self.customers[node_id]

    def edge_weight(self, from_id: int, to_id: int) -> int:
        key = (from_id, to_id)
        if key not in self._distance_cache:
            from_node = self.node(from_id)
            to_node = self.node(to_id)
            distance = sqrt((from_node.x - to_node.x) ** 2 + (from_node.y - to_node.y) ** 2)
            self._distance_cache[key] = round_half_up(distance)
        return self._distance_cache[key]

    def route_arcs(self, route: Iterable[int]) -> tuple[tuple[int, int], ...]:
        sequence = (0,) + tuple(route) + (0,)
        return tuple((sequence[idx], sequence[idx + 1]) for idx in range(len(sequence) - 1))

    def route_distance(self, route: Iterable[int]) -> float:
        sequence = (0,) + tuple(route) + (0,)
        return float(sum(self.edge_weight(sequence[idx], sequence[idx + 1]) for idx in range(len(sequence) - 1)))

    def evaluate_route(self, route: Iterable[int]) -> RouteEvaluation:
        route_tuple = tuple(route)
        if not route_tuple:
            return RouteEvaluation(False, 0.0, reason="empty route")

        seen = set()
        load = 0
        time = 0.0
        total_distance = 0.0
        arrivals: list[float] = []
        loads: list[int] = []
        previous = 0

        for customer_id in route_tuple:
            if customer_id not in self.customers:
                return RouteEvaluation(False, 0.0, reason=f"unknown customer {customer_id}")
            if customer_id in seen:
                return RouteEvaluation(False, 0.0, reason=f"repeated customer {customer_id}")
            seen.add(customer_id)
            customer = self.customers[customer_id]
            travel = self.edge_weight(previous, customer_id)
            total_distance += travel
            time = max(time + travel, float(customer.ready_time))
            if time > customer.due_date:
                return RouteEvaluation(False, total_distance, tuple(arrivals), tuple(loads), f"time window violated at {customer_id}")
            arrivals.append(time)
            load += customer.demand
            loads.append(load)
            if load > self.capacity:
                return RouteEvaluation(False, total_distance, tuple(arrivals), tuple(loads), f"capacity violated at {customer_id}")
            time += customer.service_time
            previous = customer_id

        back_travel = self.edge_weight(previous, 0)
        total_distance += back_travel
        time = max(time + back_travel, float(self.depot.ready_time))
        if time > self.depot.due_date:
            return RouteEvaluation(False, total_distance, tuple(arrivals), tuple(loads), "depot due date violated")
        return RouteEvaluation(True, total_distance, tuple(arrivals), tuple(loads))

    def make_column(self, route: Iterable[int], reduced_cost: float | None, source: str) -> RouteColumn:
        route_tuple = tuple(route)
        return RouteColumn(
            route=route_tuple,
            distance=self.route_distance(route_tuple),
            reduced_cost=reduced_cost,
            customer_incidence=frozenset(route_tuple),
            arc_incidence=frozenset(self.route_arcs(route_tuple)),
            source=source,
        )
