from __future__ import annotations

from .models import RouteColumn, VRPTWInstance


def build_feasible_seed_routes(instance: VRPTWInstance) -> list[RouteColumn]:
    routes: list[list[int]] = []
    ordered_customers = sorted(
        instance.customer_ids,
        key=lambda customer_id: (instance.customers[customer_id].due_date, instance.customers[customer_id].ready_time, customer_id),
    )

    for customer_id in ordered_customers:
        best_index = None
        best_route = None
        best_delta = float("inf")

        for route_index, route in enumerate(routes):
            for position in range(len(route) + 1):
                candidate = route[:position] + [customer_id] + route[position:]
                evaluation = instance.evaluate_route(candidate)
                if evaluation.feasible:
                    delta = evaluation.distance - instance.route_distance(route)
                    if delta < best_delta:
                        best_delta = delta
                        best_index = route_index
                        best_route = candidate

        if best_route is not None and best_index is not None:
            routes[best_index] = best_route
            continue

        if len(routes) < instance.max_vehicles:
            singleton = [customer_id]
            if not instance.evaluate_route(singleton).feasible:
                raise ValueError(f"Cannot create feasible singleton route for customer {customer_id} in {instance.name}")
            routes.append(singleton)
            continue

        raise ValueError(f"Failed to build a feasible initial solution for {instance.name}; vehicle cap exhausted")

    columns = [instance.make_column(route, reduced_cost=None, source="greedy_seed") for route in routes]

    for customer_id in instance.customer_ids:
        singleton = (customer_id,)
        evaluation = instance.evaluate_route(singleton)
        if evaluation.feasible and singleton not in {column.route for column in columns}:
            columns.append(instance.make_column(singleton, reduced_cost=None, source="singleton_seed"))

    return columns
