"""Graph coloring column generation: instance model, generators, seeds, columns."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GCInstance:
    name: str
    n: int
    edges: frozenset[tuple[int, int]]  # normalized (u<v)
    source: str = "random"

    @property
    def neighbors(self) -> dict[int, set[int]]:
        nb: dict[int, set[int]] = {v: set() for v in range(self.n)}
        for u, v in self.edges:
            nb[u].add(v)
            nb[v].add(u)
        return nb

    def is_stable(self, S: frozenset[int]) -> bool:
        for u, v in self.edges:
            if u in S and v in S:
                return False
        return True


@dataclass(frozen=True)
class GCColumn:
    vertices: frozenset[int]  # independent set (one color class)
    cost: float = 1.0
    reduced_cost: float | None = None
    source: str = "unknown"

    @property
    def signature(self) -> str:
        return "-".join(map(str, sorted(self.vertices)))


def erdos_renyi(name: str, n: int, p: float, seed: int) -> GCInstance:
    rng = random.Random(seed)
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return GCInstance(name=name, n=n, edges=frozenset(edges), source=f"G(n={n},p={p},seed={seed})")


def parse_dimacs(path: Path, name: str | None = None) -> GCInstance:
    n, edges = 0, set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            n = int(line.split()[2])
        elif line.startswith("e"):
            _, a, b = line.split()
            u, v = int(a) - 1, int(b) - 1
            edges.add((min(u, v), max(u, v)))
    return GCInstance(name=name or path.stem, n=n, edges=frozenset(edges), source=str(path))


def greedy_color_columns(inst: GCInstance) -> list[GCColumn]:
    """Welsh-Powell style greedy coloring; each color class is a seed column."""
    nb = inst.neighbors
    order = sorted(range(inst.n), key=lambda v: -len(nb[v]))
    color_of: dict[int, int] = {}
    classes: dict[int, set[int]] = {}
    for v in order:
        forbidden = {color_of[u] for u in nb[v] if u in color_of}
        c = 0
        while c in forbidden:
            c += 1
        color_of[v] = c
        classes.setdefault(c, set()).add(v)
    cols = [GCColumn(frozenset(s), source="greedy_seed") for s in classes.values()]
    # singletons as fallback seeds so every vertex is covered even if greedy fails
    covered = set().union(*classes.values())
    cols += [GCColumn(frozenset({v}), source="singleton_seed") for v in range(inst.n) if v not in covered]
    return cols
