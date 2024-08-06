"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation,
by Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Programmer: Naama Shiponi and Ben Dabush
Date: 1/6/2024
"""
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder

from fairpyx.algorithms.course_match import A_CEEI
from fairpyx.algorithms.course_match import remove_oversubscription
from fairpyx.algorithms.course_match import reduce_undersubscription

def course_match_algorithm(alloc: AllocationBuilder, budget: dict, priorities_student_list: list = [], time : int = 60):
    """
    Perform the Course Match algorithm to find the best course allocations.
    
    :param alloc: (AllocationBuilder) an allocation builder object

    :return: (dict) course allocations

    """
    price_vector = A_CEEI.A_CEEI(alloc,budget,time)
    price_vector = remove_oversubscription.remove_oversubscription(alloc, price_vector, budget)
    reduce_undersubscription.reduce_undersubscription(alloc, price_vector, budget, priorities_student_list)


    return alloc
   
def check_envy(res, instance : Instance):
    alloc = AllocationBuilder(instance)
    my_valuations = {agent: sum([alloc.instance._valuations[agent][item] for item in res[agent]]) for agent in res.keys()}
    # print(my_valuations)
    envy_agent = {agent : {} for agent in res.keys()}
    for agent in res.keys():
        for agent2 in res.keys():
            if agent != agent2:
                agent_val_sun_for_agent2_baskets = sum([alloc.instance._valuations[agent][item] for item in res[agent2]])
                if agent_val_sun_for_agent2_baskets > my_valuations[agent]:
                    envy_agent[agent][agent2] = agent_val_sun_for_agent2_baskets - my_valuations[agent]

    for agent in envy_agent.keys():
        for agent2 in envy_agent[agent].keys():
            agent_val_for_agent2_baskets= [alloc.instance._valuations[agent][item] for item in res[agent2]]
            sum_val = sum(agent_val_for_agent2_baskets)
            boolian_check_envy = False
            for val in agent_val_for_agent2_baskets:
                if  sum_val - val <= my_valuations[agent]:
                    boolian_check_envy=True
                    break
            if not boolian_check_envy:
                print(f"EF?!?! agent1 {agent} envies agent2 {agent2} by {envy_agent[agent][agent2]}")
    print("check_envy done")


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # from fairpyx import divide
    # instance = Instance(
    #   agent_conflicts = {"Alice": [], "Bob": []},
    #   item_conflicts = {"c1": [], "c2": [], "c3": []},
    #   agent_capacities = {"Alice": 2, "Bob": 1},
    #   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    #   valuations = {"Alice": {"c1": 100, "c2": 60, "c3": 0},
    #                 "Bob": {"c1": 0, "c2": 100, "c3": 0},
    # })
    # budget = {"Alice": 3.0, "Bob": 1.0}    
    

    # res = divide(course_match_algorithm, instance, budget=budget)
    # print(res)

    # check_envy(res,instance)