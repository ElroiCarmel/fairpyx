"""
An implementation of the algorithm in:
"An algorithm for the proportional division of indivisible items"
by Brams, Steven J. and Kilgour, D. Marc and Klamler, Christian (2014),
http://https://mpra.ub.uni-muenchen.de/56587/

Programmer: Elroi Carmel
Date: 2025-05
"""

import logging
import numpy as np
import cvxpy as cp
from fairpyx import Instance, AllocationBuilder, divide
from fairpyx.algorithms.minimal_bundles_utils import recursive
from typing import Any, Optional
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping
from itertools import chain
from math import isclose


logger = logging.getLogger(__name__)


def maximally_proportional_allocation(
    alloc: AllocationBuilder, min_bundles_bound: Optional[int | float] = np.inf
):
    """
    Finds an allocation maximizing the possible number of players with proportional bundles,
    guaranteeing the best attainable minimum rank among them.

    >>> instance = Instance(valuations={"Alice":[40, 35, 25], "Bob":[35, 40, 25], "Tom":[40, 25, 35]})
    >>> divide(maximally_proportional_allocation, instance)
    {'Alice': [0], 'Bob': [1], 'Tom': [2]}
    >>> valuations = [[7, 4, 33, 21, 24],
    ...              [47, 15, 43, 21, 25],
    ...              [10, 24, 20, 31, 5],
    ...              [32, 33, 3, 2, 23],
    ...              [30, 4, 1, 11, 31]]
    >>> instance = Instance(valuations=valuations)
    >>> divide(maximally_proportional_allocation, instance)
    {0: [2], 1: [0], 2: [3], 3: [1], 4: [4]}
    >>> instance = Instance(valuations=[[8,30,48,10,15],[12,7,29,15,5],[35,42,5,22,44]])
    >>> divide(maximally_proportional_allocation, instance)
    {0: [2], 1: [0, 3], 2: [1, 4]}

    """

    agents = alloc.remaining_agents()
    minimal_bundles_by_agent = {
        agent: compute_minimal_bundles_for_agent(
            recursive, alloc.instance, agent, sort_bundle=False
        )
        for agent in agents
    }
    logger.info("Collected minimal bundles by order of priorities for all agents")

    num_min_bun_by_agent = {
        agent: len(min_bundles)
        for agent, min_bundles in minimal_bundles_by_agent.items()
    }
    logger.info("Amount of minimal bundles: %s", num_min_bun_by_agent)

    # Each minimal bundle will be represented by a boolean cvxp.Variable instance that will be used
    # by cvxpy module to maximize the amount of bundles given at each iteration.

    # maps each agent to it's list of minimal bundles Variables
    # used also for constraint so that each agent won't get more than one bundle

    bundles_vars_by_agent = defaultdict(list)

    # maps each item to the list of bundles that contain it
    # used also for constraint so that each item is given exactly one time

    bundles_vars_by_item = defaultdict(list)

    max_rank = max(map(len, minimal_bundles_by_agent.values())) - 1

    logger.info(
        "Starts allocating. Stop when reach rank %s or found complete allocation",
        max_rank,
    )

    rank = max_agents_received = 0

    # while any agent can compromise and at least one agent didn't receive a bundle
    while rank <= max_rank and max_agents_received < len(agents):

        logger.debug("Max rank of minimal bundles allowed: %d", rank)

        # update the variables data structures according to the current rank

        add_bundle_indicator_vars(
            minimal_bundles_by_agent,
            bundles_vars_by_agent,
            bundles_vars_by_item,
            rank,
        )

        curr_alloc = maximise_agent_coverage(
            bundles_vars_by_agent, bundles_vars_by_item
        )
        num_agents_received = len(curr_alloc)
        logger.debug("Number of agents who received bundles: %d", num_agents_received)
        logger.debug("Found allocation (value := bundle rank): %s", curr_alloc)

        if num_agents_received > max_agents_received:
            maxmin_alloc = curr_alloc
            max_agents_received = num_agents_received
            logger.debug("------ New Maxmin allocation found ------")

        rank += 1

    logger.info(
        "Stopped descending in rank %s. Number of agents who received bundles: %d",
        rank - 1,
        max_agents_received,
    )

    logger.info("Finding the pareto optimal solution...")
    pareto_optimal_alloc = find_pareto_dominating_alloc(
        bundles_vars_by_agent, bundles_vars_by_item, maxmin_alloc
    )
    logger.info("Pareto optimal solution found: %s", pareto_optimal_alloc)

    # check if any agent got better minimal bundle

    agents_who_got_better_rank = [
        agent
        for agent in pareto_optimal_alloc
        if pareto_optimal_alloc[agent] < maxmin_alloc[agent]
    ]

    logger.debug("Agents prefering this allocation: %s", agents_who_got_better_rank)

    # give the bundles for the agents

    for agent, rank in pareto_optimal_alloc.items():
        alloc.give_bundle(agent, minimal_bundles_by_agent[agent][rank])

    logger.info("Final allocation: %s", alloc.sorted())

    # the bundle value for each agent

    bundle_value_by_agent = {}
    for agent, rank in pareto_optimal_alloc.items():
        bundle_value_by_agent[agent] = alloc.agent_bundle_value(
            agent, minimal_bundles_by_agent[agent][rank]
        )
    logger.info("Values of given bundles: %s", bundle_value_by_agent)


def compute_minimal_bundles_for_agent(
    strategy: Callable, alloc: Instance, agent: Any, **kwargs
) -> list:
    return strategy(alloc, agent, **kwargs)


def add_bundle_indicator_vars(
    min_bundles: Mapping[Hashable, list[Iterable]],
    bundle_vars_by_agent: Mapping[Hashable, Iterable[cp.Variable]],
    bundles_vars_by_item: Mapping[Hashable, Iterable[cp.Variable]],
    rank: int,
) -> None:
    """
    checks for each agent if he has a minimal bundle at the current rank \\
    and if so creates a cvxpy Variable as an indicator for the Maximization problem.
    """
    for agent, minimal_bundles in min_bundles.items():
        #  If agent has a bundle in this rank
        if rank < len(minimal_bundles):
            # Inidicator for this specific bundle
            bundle_var = cp.Variable(boolean=True)
            bundle_vars_by_agent[agent].append(bundle_var)
            for item in minimal_bundles[rank]:
                bundles_vars_by_item[item].append(bundle_var)


def maximise_agent_coverage(
    bundles_vars_by_agent: Mapping[Hashable, Iterable[cp.Variable]],
    bundles_vars_by_item: Mapping[Hashable, Iterable[cp.Variable]],
) -> Mapping[Hashable, int]:
    """
    Finds allocation which maximizes the amount of agents who receive minimal bundle

    with the following constraints:

    1. Each player can receive at most one bundle
    2. Two bundles with at least one shared item cannot both be given

    Note: The values of the retured allocation are each agent's rank of his allocated minimal bundle

    Args:
        bundle_indicators_by_agent (Mapping[Hashable, Iterable[cp.Variable]]): map each agent to it's list of minimal bundles indocators
        bundles_constraints (Mapping[Hashable, Iterable[cp.Variable]]): map each item to the list of bundles indocators that contain it

    Returns:
       Mapping[Hashable, int]: Mapper from agent to the rank of the bundle given to him
    """
    constraints = []
    for bundles in chain(bundles_vars_by_agent.values(), bundles_vars_by_item.values()):
        constraints.append(cp.sum(cp.hstack(bundles)) <= 1)

    # Collect of all the bundles Variabls to one list
    all_bundles_vars = list(chain.from_iterable(bundles_vars_by_agent.values()))

    objective = cp.Maximize(cp.sum(cp.hstack(all_bundles_vars)))

    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve()
    # Extract soliution from cvxpy Variables

    alloc_by_rank = {}
    for agent, minimal_bundles_vars in bundles_vars_by_agent.items():
        for rank, bundle_var in enumerate(minimal_bundles_vars):
            if isclose(bundle_var.value, 1):
                alloc_by_rank[agent] = rank
    return alloc_by_rank


def find_pareto_dominating_alloc(
    bundles_vars_by_agent: Mapping[Hashable, Iterable[cp.Variable]],
    bundles_vars_by_item: Mapping[Hashable, Iterable[cp.Variable]],
    maxmin_alloc: Mapping[Hashable, int],
) -> Mapping[Hashable, int]:
    """An allocation pareto dominates other allocation iff:
    1. At least one agent is given a bundle he prefers more.
    2. No agent is given a bundle he prefers less.

    Note: The values of the retured allocation are each agent's rank of his allocated minimal bundle.

    Args:
        bundles_vars_by_agent (Mapping[Hashable, Iterable[cp.Variable]]): map each agent to it's list of minimal bundles indocators
        bundles_vars_by_item (Mapping[Hashable, Iterable[cp.Variable]]): map each item to the list of bundles indocators that contain it
        maxmin_alloc (Mapping[Hashable, int]): baseline allocation

    Returns:
        Mapping[Hashable, int]: an alloction that potentioally pareto dominates the maxmin_alloc
    """

    logger.info("Search allocation that pareto dominates allocation: %s", maxmin_alloc)

    # To keep the solution maxminimal, discard bundle's variables
    # with rank worse than the rank in the maxmin allocation.

    updated_vars_by_agent = {}
    for agent, rank in maxmin_alloc.items():
        updated_vars_by_agent[agent] = bundles_vars_by_agent[agent][: rank + 1]

    constraints = []
    for bundles in chain(updated_vars_by_agent.values(), bundles_vars_by_item.values()):
        constraints.append(cp.sum(cp.hstack(bundles)) <= 1)

    # Collect of all the bundles Variabls to one list

    all_bundles_vars = cp.hstack(
        list(chain.from_iterable(updated_vars_by_agent.values()))
    )

    # Number of agents that must receive a bundle

    constraints.append(cp.sum(all_bundles_vars) == len(maxmin_alloc))
    # Search for an allocation that Pareto-dominates the current max-min
    # allocation.  Give each bundle variable a weight from 0 (most preferred)
    # down to −(num_min_bundles).  For example, if Alice has five
    # minimal bundles, their weights are 0, −1, −2, −3, −4.  With these
    # weights, a CVXPY maximisation will yield the desired allocation.

    # ❷  Build a NumPy weight array that matches the *order*
    weights = -np.concatenate(
        [
            np.arange(len(bundle_vars))  # 0,-1,-2,…
            for bundle_vars in updated_vars_by_agent.values()
        ]
    )

    # ❸  Single dot-product node
    objective = cp.Maximize(weights @ all_bundles_vars)

    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve()

    # Extract solution from cvxpy Variables

    ranking_alloc = {}
    for agent, mb_vars in updated_vars_by_agent.items():
        for rank, bundle_var in enumerate(mb_vars):
            if isclose(bundle_var.value, 1):
                ranking_alloc[agent] = rank
    return ranking_alloc


if __name__ == "__main__":
    import doctest, random, numpy as np

    # print(doctest.testmod())
    np.set_printoptions(legacy="1.25")

    RAND_SEED = random.randint(1, 10000)
    nagents, nitems = 2, 26
    instance = Instance.random_uniform(
        num_of_agents=nagents,
        num_of_items=nitems,
        item_capacity_bounds=(1, 1),
        agent_capacity_bounds=(nitems, nitems),
        item_base_value_bounds=(20, 80),
        item_subjective_ratio_bounds=(0.5, 3.0),
        normalized_sum_of_values=100,
        random_seed=RAND_SEED,
    )
    print(instance._valuations)

    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    alloc = divide(maximally_proportional_allocation, instance, min_bundles_bound=500)
    print(alloc)
