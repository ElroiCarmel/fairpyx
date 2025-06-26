"""
Differnet strategies to find all minimal bundles of an agent
Used in the maximally proportional algorithm
Programmer: Elroi Carmel
"""

import logging
from collections.abc import Callable, Iterable
from functools import partial
from itertools import combinations, accumulate, takewhile
from typing import Any
from fairpyx import Instance, AllocationBuilder

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


def recursive(instance: Instance, agent: Any, sort_bundle: bool=True) -> list:
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
    prop = get_prop_value(instance, agent)
    res = []
    agent_item_value = partial(instance.agent_item_value, agent)
    agent_bundle_value = partial(instance.agent_bundle_value, agent)
    items_sorted = sorted(
        instance.items,
        key=agent_item_value,
        reverse=True,
    )
    total_value = agent_bundle_value(instance.items)
    max_gain = [
        total_value - x for x in accumulate(map(agent_item_value, items_sorted), initial=0)
    ]
    stack = [[i] for i in range(len(items_sorted)) if max_gain[i] >= prop]  # indices
    while stack:
        bundle_idx = stack.pop()
        bundle_value = agent_bundle_value(map(items_sorted.__getitem__, bundle_idx))
        if bundle_value >= prop:
            res.append([items_sorted[i] for i in bundle_idx])
        else:
            i = bundle_idx[-1] + 1
            while i < len(items_sorted) and bundle_value + max_gain[i] >= prop:
                idx_copy = bundle_idx.copy()
                idx_copy.append(i)
                stack.append(idx_copy)
                i += 1
    return res    
        
if __name__ == "__main__":
    print(iterative(demo_instance, 1))