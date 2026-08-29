from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_core_figures(summary_csv: str | Path, iteration_csv: str | Path, output_dir: str | Path) -> list[Path]:
    summary_csv = Path(summary_csv)
    iteration_csv = Path(iteration_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_csv)
    iterations = pd.read_csv(iteration_csv)

    generated = []

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = summary.pivot(index="instance_name", columns="variant", values="final_primal_gap_percent")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Final primal gap to DIMACS BKS (%)")
    ax.set_xlabel("Instance")
    ax.set_title("Gap to benchmark by strategy")
    ax.legend(title="Variant")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    gap_path = output_dir / "gap_by_strategy.png"
    fig.savefig(gap_path, dpi=200)
    plt.close(fig)
    generated.append(gap_path)

    if "lp_improvement_percent" in summary.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot = summary.pivot(index="instance_name", columns="variant", values="lp_improvement_percent")
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("LP improvement from initial pool (%)")
        ax.set_xlabel("Instance")
        ax.set_title("Root-node LP improvement by strategy")
        ax.legend(title="Variant")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        lp_path = output_dir / "lp_improvement_by_strategy.png"
        fig.savefig(lp_path, dpi=200)
        plt.close(fig)
        generated.append(lp_path)

        family_frame = summary.copy()
        family_frame["family_group"] = family_frame["instance_name"].str.extract(r"^(RC|R|C)")
        family_pivot = family_frame.groupby(["family_group", "variant"], as_index=False)["lp_improvement_percent"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant_name, frame in family_pivot.groupby("variant"):
            ax.plot(frame["family_group"], frame["lp_improvement_percent"], marker="o", label=variant_name)
        ax.set_xlabel("Solomon family")
        ax.set_ylabel("Average LP improvement (%)")
        ax.set_title("Family-wise benefit of diversification")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        family_path = output_dir / "family_lp_improvement.png"
        fig.savefig(family_path, dpi=200)
        plt.close(fig)
        generated.append(family_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    for variant_name, frame in iterations.groupby("variant"):
        aggregate = frame.groupby("iteration", as_index=False)["selected_diversity"].mean()
        ax.plot(aggregate["iteration"], aggregate["selected_diversity"], marker="o", label=variant_name)
    ax.set_xlabel("Column generation iteration")
    ax.set_ylabel("Average selected-route diversity")
    ax.set_title("Diversity progression across strategies")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    diversity_path = output_dir / "diversity_progression.png"
    fig.savefig(diversity_path, dpi=200)
    plt.close(fig)
    generated.append(diversity_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    for variant_name, frame in iterations.groupby("variant"):
        aggregate = frame.groupby("iteration", as_index=False)["master_objective"].mean()
        ax.plot(aggregate["iteration"], aggregate["master_objective"], marker="o", label=variant_name)
    ax.set_xlabel("Column generation iteration")
    ax.set_ylabel("Average master LP objective")
    ax.set_title("Bound progression across strategies")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    bound_path = output_dir / "bound_progression.png"
    fig.savefig(bound_path, dpi=200)
    plt.close(fig)
    generated.append(bound_path)

    return generated
