"""
Compare the performance of the maximally proportional algorithms
to other algorithms in terms of bundles proportionallity
"""

import os
import experiments_csv as ex_csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fairpyx import divide, Instance
from fairpyx.algorithms.maximally_proportional import maximally_proportional_allocation
from fairpyx.algorithms import round_robin
from fairpyx.algorithms.minimal_bundles_utils import brute_force, recursive, iterative
from collections.abc import Callable
from random import sample
from functools import partial
from pathlib import Path

SEEDS = sorted([234122, 389833, 12131])
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "max_prop"
CSV_DIR = RESULTS_DIR / "results_csv"
PLOTS_DIR = RESULTS_DIR / "plots"

plt.style.use("seaborn-v0_8-white")


def dividor(
    algorithm: Callable,
    seed: int,
    nagents: int | float,
    nitems: int,
    min_bundles_strategy: Callable = iterative,
    parallel: bool = False,
) -> dict:
    instance = Instance.random_uniform(
        num_of_agents=nagents,
        num_of_items=nitems,
        agent_capacity_bounds=(nitems, nitems),
        item_capacity_bounds=(1, 1),
        item_base_value_bounds=(40, 120),
        item_subjective_ratio_bounds=(0.2, 3.0),
        normalized_sum_of_values=1000,
        random_seed=seed,
    )

    allocation = divide(
        algorithm,
        instance,
        min_bundles_strategy=min_bundles_strategy,
        parallel=parallel,
    )

    util_sum = 0
    min_prop = float("inf")

    for agent, bundle in allocation.items():
        agent_bun_val = instance.agent_bundle_value(agent, bundle)
        prop = agent_bun_val / instance.agent_bundle_value(agent, instance.items)
        util_sum += agent_bun_val
        min_prop = min(min_prop, prop)

    return {"utility_sum": util_sum, "min_proportional_utility": min_prop}


def run_multi_exp():
    agents_range = range(2, 51)
    # uncomment to reproduce my results
    # seeds = [165043187,217107415,970775402,2154093945,2600196075]
    seeds = sorted(sample(range(2**32), 5))
    for nagents in agents_range:
        ex = ex_csv.Experiment(
            results_folder=os.path.join("results", "max_prop", "results"),
            results_filename=f"{nagents}_agents.csv",
        )
        input_ranges = {
            "nagents": [nagents],
            "nitems": range(nagents, 51),
            "seed": seeds,
            "algorithm": [maximally_proportional_allocation, round_robin],
        }
        # ex.clear_previous_results()
        ex.run_with_time_limit(dividor, input_ranges, time_limit=60)


def run_imroved_algo_exp():
    max_items = 70
    agents_range = range(23, 51)
    results_folder = CSV_DIR / "improved_algo"
    for nagents in agents_range:
        ex = ex_csv.Experiment(
            results_folder=results_folder,
            results_filename=f"{nagents}_agents.csv",
        )
        input_ranges = {
            "nagents": [nagents],
            "nitems": range(nagents, max_items + 1),
            "seed": SEEDS,
            "min_bundles_strategy": [brute_force, iterative, recursive],
        }
        # ex.clear_previous_results()
        ex.run_with_time_limit(
            partial(dividor, maximally_proportional_allocation),
            input_ranges,
            time_limit=60,
        )


def comp_to_robin_plots():
    results_path = CSV_DIR / "to_round_robin"
    plots_path = PLOTS_DIR / "to_round_robin"

    for nagents in range(2, 24):
        fig, axes = plt.subplots(1, 2, sharey=False, figsize=(14, 6))
        csv_path = results_path / f"{nagents}_agents.csv"
        df = pd.read_csv(csv_path)
        df = df.pivot_table(
            index="nitems",
            columns="algorithm",
            values=["utility_sum", "min_proportional_utility"],
        )
        df["utility_sum"].plot(ax=axes[0], title="Utility Sum", legend=False)
        df["min_proportional_utility"].plot(
            ax=axes[1], title="Minimal Proportional Utility"
        )
        fig.suptitle(f"{nagents} Agents")
        plt.tight_layout()
        fig.savefig(plots_path / f"{nagents}_agents.png", dpi=300)
        plt.close()


def improve_algo_plots():
    results_path = CSV_DIR / "improved_algo"
    plots_path = PLOTS_DIR / "improved_algo"

    sns.set_theme()
    for nagents in range(2, 24):
        csv_path = results_path / f"{nagents}_agents.csv"
        df = pd.read_csv(csv_path)
        metrics = ["utility_sum", "min_proportional_utility", "runtime"]
        df = (
            df.groupby(["nitems", "min_bundles_strategy"])[metrics]
            .mean()
            .stack()
            .rename_axis(index=["nitems", "strategy", "metric"])
            .reset_index(name="performance")
        )
        g = sns.relplot(
            data=df,
            kind="line",
            x="nitems",
            y="performance",
            hue="strategy",
            col="metric",
            facet_kws=dict(sharey=False),
        )
        for ax, title in zip(
            g.axes.flat,
            ["Utility Sum", "Minimal Proportional Utility", "Runtime (sec)"],
        ):
            ax.set_title(title)
        g.figure.suptitle(f"{nagents} Agents", y=1.05)
        g.set_ylabels("Performance")
        g.savefig(plots_path / f"{nagents}_agents.png", dpi=350)
        plt.close()


def parallel_algo_plots():

    parallel_csv_path = CSV_DIR / "parallel"
    iterative_csv_path = CSV_DIR / "improved_algo"
    plots_path = PLOTS_DIR / "concurrent"
    for nagents in range(2, 26):
        csv_path = iterative_csv_path / f"{nagents}_agents.csv"
        df_iter = pd.read_csv(csv_path)
        mask = df_iter["min_bundles_strategy"] == "iterative"
        df_iter = df_iter.loc[mask, ["nitems", "seed", "runtime"]]
        csv_path = os.path.join(parallel_csv_path, f"{nagents}_agents.csv")
        df_parr = pd.read_csv(csv_path, usecols=["nitems", "seed", "runtime"])
        df = pd.merge(left=df_iter, right=df_parr, how="outer", on=["nitems", "seed"])
        df = df.pivot_table(index="nitems", values=["runtime_x", "runtime_y"])
        df.columns = ["iterative", "parallel iterative"]
        ax = df.plot(title=f"Runtime. {nagents} Agents")
        fig = ax.get_figure()
        plt.tight_layout()
        fig.savefig(plots_path / f"{nagents}_agents.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    # run_multi_exp()
    # run_imroved_algo_exp()
    # comp_to_robin_plots()
    improve_algo_plots()
    # parallel_algo_plots()
