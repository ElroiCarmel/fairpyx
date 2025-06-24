"""
Compare the performance of the maximally proportional algorithms
to other algorithms in terms of bundles proportionallity
"""

import os
import logging
import experiments_csv as ex_csv
import pandas as pd
import matplotlib.pyplot as plt
from fairpyx import divide, Instance
from fairpyx.algorithms.maximally_proportional import maximally_proportional_allocation
from fairpyx.algorithms import round_robin
from random import sample


def dividor(algorithm, seed: int, nagents: int, nitems: int) -> dict:
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

    allocation = divide(algorithm, instance)

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
        ex.run_with_time_limit(dividor, input_ranges, time_limit=60)


def plots():
    for nagents in range(2, 24):
        ex_csv.single_plot_results(
            results_csv_file=os.path.join('experiments','results','max_prop','results', f"{nagents}_agents.csv"),
            filter={"nagents": nagents},
            x_field="nitems",
            y_field="min_proportional_utility",
            z_field="algorithm",
            # title=f"{nagents} agents"
        )
        # plt.title(f"{nagents} agents")
        plt.show()


if __name__ == "__main__":
    plots()
    # run_multi_exp()
