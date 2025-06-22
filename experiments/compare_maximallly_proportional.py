"""
Compare the performance of the maximally proportional algorithms
to other algorithms in terms of bundles proportionallity
"""

import logging
import experiments_csv as ex_csv
from fairpyx import divide, Instance
from fairpyx.algorithms.maximally_proportional import maximally_proportional_allocation
from fairpyx.algorithms import round_robin
from random import sample

def dividor(algorithm, seed: int, nagents: int, nitems: int) -> dict:
    nagents = round(1/nagents)
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

if __name__ == "__main__":
    ex = ex_csv.Experiment(results_filename="maximally_proportional.csv")
    # ex.logger.setLevel(logging.DEBUG)
    # stream_handler = logging.StreamHandler()
    # file_handler = logging.FileHandler("experiment.log", mode="w", encoding="utf-8")
    # formatter = logging.Formatter(fmt="{asctime} - {message}", style="{")
    # file_handler.setFormatter(formatter)
    # ex.logger.addHandler(stream_handler)
        
    input_ranges = {
    "nagents": [1/n for n in range(2, 50)],
    "nitems": range(4, 50),
    "seed": sorted(sample(range(2**32), 5)),
    "algorithm": [maximally_proportional_allocation, round_robin],
    }
    # ex.clear_previous_results()
    ex.run_with_time_limit(dividor, input_ranges, time_limit=60)