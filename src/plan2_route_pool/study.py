from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys

import pandas as pd
import yaml

from .cuts import SubsetRowCut, customer_ambiguity_scores, separate_subset_row_cuts
from .diversity import average_pairwise_diversity, select_adaptive_columns, select_diversified_columns
from .initializer import build_feasible_seed_routes
from .master_problem import solve_master
from .models import RouteColumn
from .pricing import PricingResult, compute_reduced_cost, generate_candidate_routes
from .vrptw_parser import parse_solomon_like_instance


@dataclass
class VariantConfig:
    name: str
    strategy: str
    candidate_pool_size: int
    add_count: int
    pricing_time_limit: float
    pricing_threads: int
    max_iterations: int
    max_route_age: int
    diversity_cost_weight: float
    enable_pool_pruning: bool
    reduced_cost_weight: float = 0.6
    diversity_weight: float = 0.4
    ambiguity_weight: float = 0.0
    use_dual_stabilization: bool = False
    stabilization_beta: float = 1.0
    use_subset_row_cuts: bool = False
    max_subset_row_cuts: int = 0
    subset_row_customer_pool: int = 12
    subset_row_max_new_cuts: int = 3
    use_route_reservoir: bool = False
    reservoir_max_size: int = 0
    exact_pricing_every: int = 1
    reservoir_skip_min_candidates: int | None = None
    reservoir_min_hard_iteration: int = 1
    adaptive_hardening: bool = False
    hard_runtime_trigger_fraction: float = 0.95
    cuts_min_iteration: int = 1
    cuts_min_hard_iteration: int | None = None
    cuts_min_reservoir_size: int = 0
    tighten_big_m: bool = True
    rule_mode: str = "none"
    rule_dual_disp_threshold: float = 5.0
    rule_tl_window: int = 5
    rule_tl_rate_threshold: float = 0.4
    rule_hard_add_count: int = 3
    rule_hard_beta: float = 0.75
    rule_pool_first_min_candidates: int = 5


def _project_root_from_config(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for candidate in resolved.parents:
        direct_marker = candidate / "data" / "provenance" / "vrptw_dimacs_bks.csv"
        if direct_marker.exists():
            return candidate
        legacy_candidate = candidate / "legacy" / "reproducibility_package"
        legacy_marker = legacy_candidate / "data" / "provenance" / "vrptw_dimacs_bks.csv"
        if legacy_marker.exists():
            return legacy_candidate
    raise FileNotFoundError(f"Could not infer reproducibility package root from config path: {config_path}")


def _load_bks_map(project_root: Path) -> dict[str, dict[str, object]]:
    bks_csv = project_root / "data" / "provenance" / "vrptw_dimacs_bks.csv"
    frame = pd.read_csv(bks_csv)
    return {row["instance_name"]: row.to_dict() for _, row in frame.iterrows()}


def _select_columns(
    candidates: list[RouteColumn],
    variant: VariantConfig,
    customer_scores: dict[int, float] | None = None,
    hard_mode: bool = True,
) -> list[RouteColumn]:
    negative_candidates = [candidate for candidate in candidates if (candidate.reduced_cost or 0.0) < -1e-6]
    if not negative_candidates:
        return []
    if variant.strategy == "single":
        return negative_candidates[:1]
    if variant.strategy == "naive_multi":
        return negative_candidates[: variant.add_count]
    if variant.strategy == "diversified":
        return select_diversified_columns(negative_candidates, add_count=variant.add_count, cost_weight=variant.diversity_cost_weight)
    if variant.strategy == "hybrid_plus":
        return select_adaptive_columns(
            negative_candidates,
            add_count=variant.add_count,
            reduced_cost_weight=variant.reduced_cost_weight,
            diversity_weight=variant.diversity_weight,
            ambiguity_weight=variant.ambiguity_weight,
            customer_scores=customer_scores,
        )
    if variant.strategy == "adaptive_super":
        if hard_mode:
            return select_adaptive_columns(
                negative_candidates,
                add_count=variant.add_count,
                reduced_cost_weight=variant.reduced_cost_weight,
                diversity_weight=variant.diversity_weight,
                ambiguity_weight=variant.ambiguity_weight,
                customer_scores=customer_scores,
            )
        return select_diversified_columns(negative_candidates, add_count=variant.add_count, cost_weight=variant.diversity_cost_weight)
    raise ValueError(f"Unknown strategy: {variant.strategy}")


def _prune_pool(active_columns: dict[tuple[int, ...], dict[str, object]], max_route_age: int) -> None:
    doomed = [
        route
        for route, metadata in active_columns.items()
        if metadata["age"] > max_route_age and metadata["last_value"] < 1e-8 and metadata["source"] not in {"greedy_seed"}
    ]
    for route in doomed:
        del active_columns[route]


def _stabilize_map(current: dict, previous: dict, beta: float) -> dict:
    keys = set(current) | set(previous)
    return {key: beta * current.get(key, 0.0) + (1.0 - beta) * previous.get(key, 0.0) for key in keys}


def _rescore_routes(
    columns: list[RouteColumn],
    dual_customers: dict[int, float],
    dual_vehicle: float,
    subset_row_cut_duals: dict[tuple[int, int, int], float] | None,
    source_suffix: str,
) -> list[RouteColumn]:
    rescored = []
    for column in columns:
        reduced_cost = compute_reduced_cost(column, dual_customers=dual_customers, dual_vehicle=dual_vehicle, subset_row_cut_duals=subset_row_cut_duals)
        rescored.append(
            RouteColumn(
                route=column.route,
                distance=column.distance,
                reduced_cost=reduced_cost,
                customer_incidence=column.customer_incidence,
                arc_incidence=column.arc_incidence,
                source=f"{column.source}_{source_suffix}",
            )
        )
    rescored.sort(key=lambda column: column.reduced_cost if column.reduced_cost is not None else float("inf"))
    return rescored


def _cap_reservoir(reservoir: dict[tuple[int, ...], RouteColumn], max_size: int) -> dict[tuple[int, ...], RouteColumn]:
    if max_size <= 0 or len(reservoir) <= max_size:
        return reservoir
    kept = sorted(reservoir.values(), key=lambda column: column.reduced_cost if column.reduced_cost is not None else float("inf"))[:max_size]
    return {column.route: column for column in kept}


def _activation_gate(iteration: int, phase_iteration: int, min_iteration: int, min_phase_iteration: int | None) -> bool:
    if iteration < min_iteration:
        return False
    if min_phase_iteration is not None and phase_iteration < min_phase_iteration:
        return False
    return True


def _reservoir_active(feature_mode: bool, variant: VariantConfig, phase_iteration: int) -> bool:
    return feature_mode and variant.use_route_reservoir and phase_iteration >= variant.reservoir_min_hard_iteration


def _should_use_exact_pricing(
    feature_mode: bool,
    variant: VariantConfig,
    reservoir_candidate_count: int,
    iteration: int,
    phase_iteration: int,
) -> bool:
    if not _reservoir_active(feature_mode, variant, phase_iteration) or variant.exact_pricing_every <= 1:
        return True
    skip_threshold = variant.reservoir_skip_min_candidates or variant.candidate_pool_size
    skip_threshold = min(skip_threshold, variant.candidate_pool_size)
    if reservoir_candidate_count < skip_threshold:
        return True
    return iteration % variant.exact_pricing_every == 0


def run_single_instance(project_root: Path, instance_name: str, variant: VariantConfig) -> dict[str, object]:
    bks_row = _load_bks_map(project_root)[instance_name]
    instance_path = project_root / "data" / "raw" / "vrptw" / "controller" / "VRPTWController-master" / bks_row["relative_path"]
    instance = parse_solomon_like_instance(
        path=instance_path,
        source_family=str(bks_row["family"]),
        benchmark_distance=float(bks_row["best_known_solution"]),
        benchmark_optimal=bool(bks_row["is_optimal"]),
    )

    active_columns = {}
    for column in build_feasible_seed_routes(instance):
        active_columns[column.route] = {"column": column, "age": 0, "last_value": 0.0, "source": column.source}

    subset_row_cuts: list[SubsetRowCut] = []
    subset_row_signatures: set[str] = set()
    reservoir: dict[tuple[int, ...], RouteColumn] = {}
    stabilized_dual_customers: dict[int, float] = {}
    stabilized_dual_vehicle = 0.0
    stabilized_cut_duals: dict[tuple[int, int, int], float] = {}
    hard_mode = False
    hard_phase_iteration = 0

    initial_pool_columns = [metadata["column"] for metadata in active_columns.values()]
    initial_lp = solve_master(instance, initial_pool_columns, binary=False, subset_row_cuts=subset_row_cuts)

    iteration_rows = []
    termination_reason = "max_iterations"
    pricing_statuses = []
    prev_master_objective: float | None = None
    prev_dual_customers: dict[int, float] = {}
    prev_dual_vehicle = 0.0

    for iteration in range(1, variant.max_iterations + 1):
        pool_columns = [metadata["column"] for metadata in active_columns.values()]
        master_result = solve_master(instance, pool_columns, binary=False, subset_row_cuts=subset_row_cuts)

        feature_mode = hard_mode or not variant.adaptive_hardening
        if feature_mode:
            if variant.adaptive_hardening:
                hard_phase_iteration += 1
                phase_iteration = hard_phase_iteration
            else:
                phase_iteration = iteration
        else:
            phase_iteration = 0

        cut_gate = (
            feature_mode
            and variant.use_subset_row_cuts
            and len(subset_row_cuts) < variant.max_subset_row_cuts
            and _activation_gate(
                iteration=iteration,
                phase_iteration=phase_iteration,
                min_iteration=variant.cuts_min_iteration,
                min_phase_iteration=variant.cuts_min_hard_iteration,
            )
            and len(reservoir) >= variant.cuts_min_reservoir_size
        )
        if cut_gate:
            new_cuts = separate_subset_row_cuts(
                columns=pool_columns,
                lambda_values=master_result.lambda_values,
                customer_ids=instance.customer_ids,
                existing_signatures=subset_row_signatures,
                candidate_customer_pool=variant.subset_row_customer_pool,
                max_new_cuts=min(variant.subset_row_max_new_cuts, variant.max_subset_row_cuts - len(subset_row_cuts)),
            )
            if new_cuts:
                subset_row_cuts.extend(new_cuts)
                subset_row_signatures.update(cut.signature for cut in new_cuts)
                master_result = solve_master(instance, pool_columns, binary=False, subset_row_cuts=subset_row_cuts)

        for metadata in active_columns.values():
            signature = metadata["column"].signature
            metadata["last_value"] = master_result.lambda_values.get(signature, 0.0)
            metadata["age"] = 0 if metadata["last_value"] > 1e-8 else metadata["age"] + 1

        if variant.enable_pool_pruning:
            _prune_pool(active_columns, variant.max_route_age)
            pool_columns = [metadata["column"] for metadata in active_columns.values()]
            master_result = solve_master(instance, pool_columns, binary=False, subset_row_cuts=subset_row_cuts)

        actual_dual_customers = master_result.dual_customers or {}
        actual_dual_vehicle = master_result.dual_vehicle or 0.0
        actual_cut_duals = master_result.dual_subset_row_cuts or {}
        customer_scores = customer_ambiguity_scores(pool_columns, master_result.lambda_values, instance.customer_ids)

        dual_l1_now = sum(abs(actual_dual_customers.get(cid, 0.0) - prev_dual_customers.get(cid, 0.0)) for cid in set(actual_dual_customers) | set(prev_dual_customers))
        dual_l1_now += abs(actual_dual_vehicle - prev_dual_vehicle)
        n_duals_now = max(len(set(actual_dual_customers) | set(prev_dual_customers)) + 1, 1)
        dual_l1_now /= n_duals_now

        hard_now = False
        if variant.rule_mode == "signal_threshold":
            window = pricing_statuses[-variant.rule_tl_window:] if pricing_statuses else []
            tl_rate = (sum(1 for s in window if s == "TIME_LIMIT") / len(window)) if window else 0.0
            hard_now = (tl_rate >= variant.rule_tl_rate_threshold) or (dual_l1_now > variant.rule_dual_disp_threshold)
        variant_eff = variant
        if hard_now:
            variant_eff = dataclasses.replace(
                variant,
                add_count=variant.rule_hard_add_count,
                use_dual_stabilization=True,
                stabilization_beta=variant.rule_hard_beta,
            )

        if feature_mode and variant_eff.use_dual_stabilization:
            beta = variant_eff.stabilization_beta
            stabilized_dual_customers = _stabilize_map(actual_dual_customers, stabilized_dual_customers, beta)
            stabilized_dual_vehicle = beta * actual_dual_vehicle + (1.0 - beta) * stabilized_dual_vehicle
            stabilized_cut_duals = _stabilize_map(actual_cut_duals, stabilized_cut_duals, beta)
        else:
            stabilized_dual_customers = dict(actual_dual_customers)
            stabilized_dual_vehicle = actual_dual_vehicle
            stabilized_cut_duals = dict(actual_cut_duals)

        reservoir_candidates: list[RouteColumn] = []
        reservoir_active = _reservoir_active(feature_mode, variant, phase_iteration)
        if reservoir_active and reservoir:
            reservoir = {route: column for route, column in reservoir.items() if route not in active_columns}
            reservoir_candidates = [
                column
                for column in _rescore_routes(
                    list(reservoir.values()),
                    dual_customers=actual_dual_customers,
                    dual_vehicle=actual_dual_vehicle,
                    subset_row_cut_duals=actual_cut_duals,
                    source_suffix="rescored",
                )
                if (column.reduced_cost or 0.0) < -1e-6
            ]

        use_exact_pricing = _should_use_exact_pricing(
            feature_mode=feature_mode,
            variant=variant_eff,
            reservoir_candidate_count=len(reservoir_candidates),
            iteration=iteration,
            phase_iteration=phase_iteration,
        )
        if variant.rule_mode == "signal_threshold" and hard_now and len(reservoir_candidates) >= variant.rule_pool_first_min_candidates:
            use_exact_pricing = False

        if use_exact_pricing:
            pricing_result = generate_candidate_routes(
                instance=instance,
                dual_customers=stabilized_dual_customers,
                dual_vehicle=stabilized_dual_vehicle,
                pool_solutions=variant_eff.candidate_pool_size,
                time_limit=variant_eff.pricing_time_limit,
                threads=variant_eff.pricing_threads,
                subset_row_cut_duals=stabilized_cut_duals,
                tighten_big_m=variant_eff.tighten_big_m,
            )
            fresh_candidates = _rescore_routes(
                pricing_result.candidates,
                dual_customers=actual_dual_customers,
                dual_vehicle=actual_dual_vehicle,
                subset_row_cut_duals=actual_cut_duals,
                source_suffix="actual",
            )
        else:
            pricing_result = PricingResult(status="RESERVOIR", candidates=reservoir_candidates[: variant.candidate_pool_size], objective_value=None, mip_gap=0.0, runtime=0.0)
            fresh_candidates = []

        pricing_statuses.append(pricing_result.status)

        if feature_mode and variant.use_route_reservoir:
            for column in fresh_candidates:
                if column.route not in active_columns:
                    reservoir[column.route] = column
            reservoir = _cap_reservoir(reservoir, variant.reservoir_max_size)

        merged_candidates: dict[tuple[int, ...], RouteColumn] = {}
        for candidate in reservoir_candidates + fresh_candidates:
            if candidate.route in active_columns:
                continue
            current = merged_candidates.get(candidate.route)
            if current is None or (candidate.reduced_cost or float("inf")) < (current.reduced_cost or float("inf")):
                merged_candidates[candidate.route] = candidate
        candidate_pool = sorted(merged_candidates.values(), key=lambda column: column.reduced_cost if column.reduced_cost is not None else float("inf"))
        selected = _select_columns(candidate_pool[: variant_eff.candidate_pool_size], variant_eff, customer_scores=customer_scores if feature_mode else None, hard_mode=feature_mode)

        for column in selected:
            active_columns[column.route] = {"column": column, "age": 0, "last_value": 0.0, "source": column.source}
            reservoir.pop(column.route, None)

        candidate_rcs = [candidate.reduced_cost for candidate in candidate_pool if candidate.reduced_cost is not None]
        rc_dispersion = float(pd.Series(candidate_rcs).std()) if len(candidate_rcs) > 1 else 0.0
        dual_l1 = sum(abs(actual_dual_customers.get(cid, 0.0) - prev_dual_customers.get(cid, 0.0)) for cid in set(actual_dual_customers) | set(prev_dual_customers))
        dual_l1 += abs(actual_dual_vehicle - prev_dual_vehicle)
        n_duals = max(len(set(actual_dual_customers) | set(prev_dual_customers)) + 1, 1)
        master_delta = master_result.objective_value - prev_master_objective if prev_master_objective is not None else 0.0

        iteration_rows.append(
            {
                "instance_name": instance.name,
                "variant": variant.name,
                "iteration": iteration,
                "master_objective": master_result.objective_value,
                "master_delta": master_delta,
                "dual_l1_displacement": dual_l1 / n_duals,
                "rc_dispersion": rc_dispersion,
                "route_pool_size": len(active_columns),
                "candidate_count": len(candidate_pool),
                "fresh_candidate_count": len(fresh_candidates),
                "selected_count": len(selected),
                "candidate_diversity": average_pairwise_diversity(candidate_pool),
                "selected_diversity": average_pairwise_diversity(selected),
                "best_reduced_cost": min([candidate.reduced_cost for candidate in candidate_pool if candidate.reduced_cost is not None] + [0.0]),
                "pricing_runtime_seconds": pricing_result.runtime,
                "pricing_status": pricing_result.status,
                "pricing_gap": pricing_result.mip_gap,
                "pricing_dettime_ticks": pricing_result.dettime_ticks,
                "pricing_nodes_processed": pricing_result.nodes_processed,
                "pricing_simplex_iterations": pricing_result.simplex_iterations,
                "active_subset_row_cuts": len(subset_row_cuts),
                "reservoir_negative_candidates": len(reservoir_candidates),
                "hard_mode_iteration": phase_iteration,
                "reservoir_active": reservoir_active,
                "customer_ambiguity_mean": sum(customer_scores.values()) / len(customer_scores) if customer_scores else 0.0,
                "selected_signatures": ";".join(column.signature for column in selected),
                "rule_hard": int(hard_now),
            }
        )

        prev_master_objective = master_result.objective_value
        prev_dual_customers = dict(actual_dual_customers)
        prev_dual_vehicle = actual_dual_vehicle

        if variant.adaptive_hardening and not hard_mode:
            runtime_trigger = pricing_result.runtime >= variant.hard_runtime_trigger_fraction * variant.pricing_time_limit
            if pricing_result.status == "TIME_LIMIT" or runtime_trigger:
                hard_mode = True

        if not selected:
            termination_reason = "no_negative_candidate"
            break

    final_columns = [metadata["column"] for metadata in active_columns.values()]
    final_lp = solve_master(instance, final_columns, binary=False, subset_row_cuts=subset_row_cuts)
    final_ip = solve_master(instance, final_columns, binary=True, subset_row_cuts=subset_row_cuts)
    bks_distance = float(instance.benchmark_distance) if instance.benchmark_distance is not None else None
    primal_gap = None
    if bks_distance is not None and bks_distance > 0:
        primal_gap = 100.0 * (final_ip.objective_value - bks_distance) / bks_distance
    lp_improvement_percent = 100.0 * (initial_lp.objective_value - final_lp.objective_value) / initial_lp.objective_value if initial_lp.objective_value else 0.0

    return {
        "instance_name": instance.name,
        "family": instance.source_family,
        "variant": variant.name,
        "instance_relative_path": str(bks_row["relative_path"]),
        "instance_path": str(instance_path),
        "bks_source_script": str(bks_row.get("source_script", bks_row.get("script_name", ""))),
        "bks_distance": bks_distance,
        "bks_is_optimal": instance.benchmark_optimal,
        "max_vehicles": instance.max_vehicles,
        "customer_count": len(instance.customer_ids),
        "initial_lp_objective": initial_lp.objective_value,
        "final_lp_objective": final_lp.objective_value,
        "final_ip_objective": final_ip.objective_value,
        "final_primal_gap_percent": primal_gap,
        "lp_improvement_percent": lp_improvement_percent,
        "total_pricing_time_seconds": sum(row["pricing_runtime_seconds"] for row in iteration_rows),
        "average_selected_diversity": sum(row["selected_diversity"] for row in iteration_rows) / len(iteration_rows) if iteration_rows else 0.0,
        "average_candidate_diversity": sum(row["candidate_diversity"] for row in iteration_rows) / len(iteration_rows) if iteration_rows else 0.0,
        "final_route_pool_size": len(final_columns),
        "final_subset_row_cut_count": len(subset_row_cuts),
        "reservoir_size_final": len(reservoir),
        "termination_reason": termination_reason,
        "iterations_completed": len(iteration_rows),
        "pricing_statuses": pricing_statuses,
        "iteration_rows": iteration_rows,
        "final_lambda_values": {signature: float(value) for signature, value in final_lp.lambda_values.items()},
    }


def run_study_from_config(config_path: str | Path) -> tuple[Path, Path]:
    config_path = Path(config_path)
    project_root = _project_root_from_config(config_path)
    results_root = project_root.parent.parent / "results"
    logs_root = project_root.parent.parent / "logs"
    results_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")
    config_sha256 = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    study_name = config["study_name"]
    summary_dir = results_root / "summary"
    raw_dir = results_root / "raw" / study_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)

    variant_objects = [
        VariantConfig(
            name=variant["name"],
            strategy=variant["strategy"],
            candidate_pool_size=int(variant["candidate_pool_size"]),
            add_count=int(variant["add_count"]),
            pricing_time_limit=float(variant["pricing_time_limit"]),
            pricing_threads=int(variant["pricing_threads"]),
            max_iterations=int(variant["max_iterations"]),
            max_route_age=int(variant["max_route_age"]),
            diversity_cost_weight=float(variant.get("diversity_cost_weight", 0.6)),
            enable_pool_pruning=bool(variant.get("enable_pool_pruning", True)),
            reduced_cost_weight=float(variant.get("reduced_cost_weight", variant.get("diversity_cost_weight", 0.6))),
            diversity_weight=float(variant.get("diversity_weight", 0.4)),
            ambiguity_weight=float(variant.get("ambiguity_weight", 0.0)),
            use_dual_stabilization=bool(variant.get("use_dual_stabilization", False)),
            stabilization_beta=float(variant.get("stabilization_beta", 1.0)),
            use_subset_row_cuts=bool(variant.get("use_subset_row_cuts", False)),
            max_subset_row_cuts=int(variant.get("max_subset_row_cuts", 0)),
            subset_row_customer_pool=int(variant.get("subset_row_customer_pool", 12)),
            subset_row_max_new_cuts=int(variant.get("subset_row_max_new_cuts", 3)),
            use_route_reservoir=bool(variant.get("use_route_reservoir", False)),
            reservoir_max_size=int(variant.get("reservoir_max_size", 0)),
            exact_pricing_every=int(variant.get("exact_pricing_every", 1)),
            reservoir_skip_min_candidates=(
                None if variant.get("reservoir_skip_min_candidates") is None else int(variant.get("reservoir_skip_min_candidates"))
            ),
            reservoir_min_hard_iteration=int(variant.get("reservoir_min_hard_iteration", 1)),
            adaptive_hardening=bool(variant.get("adaptive_hardening", False)),
            hard_runtime_trigger_fraction=float(variant.get("hard_runtime_trigger_fraction", 0.95)),
            cuts_min_iteration=int(variant.get("cuts_min_iteration", 1)),
            cuts_min_hard_iteration=(
                None if variant.get("cuts_min_hard_iteration") is None else int(variant.get("cuts_min_hard_iteration"))
            ),
            cuts_min_reservoir_size=int(variant.get("cuts_min_reservoir_size", 0)),
            tighten_big_m=bool(variant.get("tighten_big_m", True)),
            rule_mode=str(variant.get("rule_mode", "none")),
            rule_dual_disp_threshold=float(variant.get("rule_dual_disp_threshold", 5.0)),
            rule_tl_window=int(variant.get("rule_tl_window", 5)),
            rule_tl_rate_threshold=float(variant.get("rule_tl_rate_threshold", 0.4)),
            rule_hard_add_count=int(variant.get("rule_hard_add_count", 3)),
            rule_hard_beta=float(variant.get("rule_hard_beta", 0.75)),
            rule_pool_first_min_candidates=int(variant.get("rule_pool_first_min_candidates", 5)),
        )
        for variant in config["variants"]
    ]

    summary_rows = []
    iteration_rows = []

    for instance_name in config["instances"]:
        for variant in variant_objects:
            run_result = run_single_instance(project_root, instance_name=instance_name, variant=variant)
            run_result["study_name"] = study_name
            run_result["config_path"] = str(config_path.resolve())
            run_result["config_sha256"] = config_sha256
            raw_path = raw_dir / f"{instance_name}__{variant.name}.json"
            raw_path.write_text(json.dumps(run_result, indent=2, ensure_ascii=False), encoding="utf-8")
            summary_rows.append({key: value for key, value in run_result.items() if key not in {"iteration_rows", "pricing_statuses"}})
            iteration_rows.extend(run_result["iteration_rows"])

    summary_frame = pd.DataFrame(summary_rows).sort_values(["instance_name", "variant"])
    iteration_frame = pd.DataFrame(iteration_rows).sort_values(["instance_name", "variant", "iteration"])

    summary_path = summary_dir / f"{study_name}_summary.csv"
    iteration_path = summary_dir / f"{study_name}_iterations.csv"
    summary_frame.to_csv(summary_path, index=False)
    iteration_frame.to_csv(iteration_path, index=False)

    finished_at = datetime.now(timezone.utc)
    manifest = {
        "study_name": study_name,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "project_root": str(project_root.resolve()),
        "results_root": str(results_root.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "iteration_path": str(iteration_path.resolve()),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "instances": list(config["instances"]),
        "variants": list(config["variants"]),
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    manifest_path = raw_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary_path, iteration_path
