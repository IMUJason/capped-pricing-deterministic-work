from __future__ import annotations

from pathlib import Path

from .models import VRPTWCustomer, VRPTWInstance


def parse_solomon_like_instance(path: str | Path, source_family: str, benchmark_distance: float | None = None, benchmark_optimal: bool | None = None) -> VRPTWInstance:
    path = Path(path)
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    name = next((line.strip() for line in lines if line.strip()), path.stem)

    numeric_rows: list[list[int]] = []
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        if all(token.lstrip("-").isdigit() for token in tokens):
            numeric_rows.append([int(token) for token in tokens])

    if len(numeric_rows) < 2:
        raise ValueError(f"Cannot parse Solomon-like instance: {path}")

    max_vehicles, capacity = numeric_rows[0][:2]
    customer_rows = numeric_rows[1:]

    depot_row = customer_rows[0]
    depot = VRPTWCustomer(
        customer_id=0,
        x=float(depot_row[1]),
        y=float(depot_row[2]),
        demand=int(depot_row[3]),
        ready_time=int(depot_row[4]),
        due_date=int(depot_row[5]),
        service_time=int(depot_row[6]),
    )

    customers = {
        row[0]: VRPTWCustomer(
            customer_id=int(row[0]),
            x=float(row[1]),
            y=float(row[2]),
            demand=int(row[3]),
            ready_time=int(row[4]),
            due_date=int(row[5]),
            service_time=int(row[6]),
        )
        for row in customer_rows[1:]
    }

    return VRPTWInstance(
        name=name,
        max_vehicles=max_vehicles,
        capacity=capacity,
        depot=depot,
        customers=customers,
        source_family=source_family,
        source_path=str(path),
        benchmark_distance=benchmark_distance,
        benchmark_optimal=benchmark_optimal,
    )
