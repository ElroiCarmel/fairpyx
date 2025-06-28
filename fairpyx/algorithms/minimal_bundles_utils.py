"""
Differnet strategies to find all minimal bundles of an agent
Used in the maximally proportional algorithm
Programmer: Elroi Carmel
"""

import logging
from collections.abc import Callable, Iterable
from functools import partial
from itertools import combinations, accumulate
from typing import Any
from fairpyx import Instance

logger = logging.Logger(__name__)
########## Basic Instance for doctest ##########
valuation = [[15, 33, 16, 6, 26], [34, 38, 4, 16, 8]]
demo_instance = Instance(valuation)
###############################################


def get_prop_value(instance: Instance, agent: Any) -> float:
    """
    >>> get_prop_value(demo_instance, 0)
    48.0
    >>> get_prop_value(demo_instance, 1)
    50.0
    """
    return instance.agent_bundle_value(agent, instance.items) / instance.num_of_agents


def is_minimal_bundle(bundle: Iterable, val_func: Callable, prop: float) -> bool:
    """
    >>> val_func = partial(instance.agent_item_value, 0)
    >>> prop = 48
    >>> is_minimal_bundle([2, 3, 4], val_func, prop)
    True
    >>> is_minimal_bundle([0, 2, 3, 4], val_func, prop)
    False
    """
    bundle_value = sum(map(val_func, bundle))
    min_val = min(map(val_func, bundle))
    return bundle_value >= prop and bundle_value - min_val < prop


def brute_force(instance: Instance, agent: Any) -> list:
    """
    >>> min_bundles = {(1, 4), (0, 2, 4), (1, 2), (0, 1), (2, 3, 4)}
    >>> set(brute_force(demo_instance, 0)) == min_bundles
    True
    """
    prop = get_prop_value(instance, agent)
    res = []
    for size in range(1, instance.num_of_items):
        for bundle in combinations(instance.items, size):
            if is_minimal_bundle(
                bundle, partial(instance.agent_item_value, agent), prop
            ):
                res.append(bundle)
    return res


def recursive(instance: Instance, agent: Any, sort_bundle: bool = True) -> list:
    prop = get_prop_value(instance, agent)
    res = []
    items_sorted = sorted(
        instance.items,
        key=partial(instance.agent_item_value, agent),
        reverse=True,
    )
    logger.info("Proportional value of agent %s is %s", agent, prop)
    subgroup = []

    def backtrack(i, bundle_value):
        logger.debug("Assesing subgroup %s", subgroup)
        if bundle_value >= prop:
            res.append(sorted(subgroup) if sort_bundle else subgroup.copy())
            logger.debug(
                "Subgroup is minimal bundle! total value: %s. Added to result",
                bundle_value,
            )
        elif i >= len(items_sorted):
            logger.debug("No left items to grow the group")
            return
        else:
            logger.debug("Subgroup total value is too low. Add some item")
            item = items_sorted[i]
            subgroup.append(item)
            backtrack(i + 1, bundle_value + instance.agent_item_value(agent, item))
            subgroup.pop()
            backtrack(i + 1, bundle_value)

    backtrack(0, 0)
    return res


def iterative(instance: Instance, agent: Any) -> list:
    proportional_share = get_prop_value(instance, agent)
    logger.debug(
        "The proportional share of agent '%s' is %s.", agent, proportional_share
    )
    minimal_bundles = []
    items_sorted = sorted(
        instance.items,
        key=lambda item: instance.agent_item_value(agent, item),
        reverse=True,
    )
    item_valuation = [instance.agent_item_value(agent, item) for item in items_sorted]
    logger.debug("Sorted the items by value in decending order: %s.", items_sorted)
    logger.debug("Item valuation: %s.", item_valuation)
    
    # max_gain[i] = the total value of item[i] till the last item
    max_gain = list(accumulate(reversed(item_valuation)))
    max_gain.reverse()
    logger.debug("Max gain: %s.", max_gain)
    
    # consists of (bundle total value, index of the next item to add to bundle, the bundle)
    stack = []
    i = 0
    # add only items that are potentionally part of minimal bundle
    while max_gain[i] >= proportional_share:
        stack.append((item_valuation[i], i + 1, [items_sorted[i]]))
        i += 1
    logger.debug(
        "The initial candidates for minimal bundles are %s.", [x[2] for x in stack]
    )

    while stack:
        bundle_value, next_item_ind, bundle = stack.pop()
        logger.debug("Assesing bundle: %s. Utility: %s.", bundle, bundle_value)
        if bundle_value >= proportional_share:
            minimal_bundles.append(bundle)
            logger.debug("*** Found minimal bundle!! ***")
        else:
            logger.debug("Utility not big enough.")
            while (
                next_item_ind < len(items_sorted)
                and bundle_value + max_gain[next_item_ind] >= proportional_share
            ):
                bundle_copy = bundle.copy()
                bundle_copy.append(items_sorted[next_item_ind])
                stack.append(
                    (
                        bundle_value + item_valuation[next_item_ind],
                        next_item_ind + 1,
                        bundle_copy,
                    )
                )
                next_item_ind += 1

    logger.debug("Agent '%s' has %d minimal bundles.", agent, len(minimal_bundles))
    return minimal_bundles


if __name__ == "__main__":
    agent = 1
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    instance = Instance.random_uniform(
        num_of_agents=2,
        num_of_items=8,
        item_capacity_bounds=(1, 1),
        agent_capacity_bounds=(6, 6),
        item_base_value_bounds=(60, 100),
        item_subjective_ratio_bounds=(0.6, 1.4),
        normalized_sum_of_values=100,
        random_seed=12324,
    )
    print(iterative(instance, "s1"))
