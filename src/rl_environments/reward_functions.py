"""
This script contains the reward functions for both Wagner and 
Brouwer's conjectures. 
"""
import math 

import networkx as nx

from src.graph_theory_utils.graph_theory import (
    calculate_laplacian_eigenvalues,
    calculate_matching_number,
    calculate_max_abs_val_eigenvalue,
)
from src.rl_environments.environments import N_VERTICES


def wagner_inequality_to_reward(graph: nx.Graph) -> float: 
    """
    Since we want to minimize lambda1 + mu (our loss function), we return the
    negative of this. Moreover, given that we want to disprove that 
    lambda1 + mu >= sqrt(N-1) + 1, if we consider our reward function to be 
    sqrt(N-1) + 1 - (lambda1 + mu) then it is enough to check whether the reward is 
    positive. In such a case, we'll have found a counterexample for the conjecture.
    """
    lambda1 = calculate_max_abs_val_eigenvalue(graph=graph)
    mu = calculate_matching_number(graph=graph)
    return math.sqrt(N_VERTICES-1) + 1 - (lambda1 + mu) 


def calculate_reward_wagner(
    graph: nx.Graph, method: str, 
) -> float:
    """
    This function calculates the reward for Wagner's conjecture problem. 
    """
    if method == 'cross_entropy':
        # The conjecture assumes our graph is connected. We get rid of unconnected graphs
        if not nx.is_connected(graph):
            return -float('inf')
        else:  
            return wagner_inequality_to_reward(graph=graph)
 
    if method == 'q_learning':
        return wagner_inequality_to_reward(graph=graph)


def brouwer_inequality_to_reward(graph: nx.Graph) -> float:
    """
    # TODO: explanation of how the reward function is derived from the
    conjecture --> this need to be decided
    """
    eigenvals_list = calculate_laplacian_eigenvalues(graph=graph)
    


def calculate_reward_brouwer(
    graph: nx.Graph, method: str,
) -> float:
    """
    This function calculates the reward for Brouwer's conjecture problem. 
    """
    if method == 'cross_entropy':
        # The conjecture assumes our graph is connected. We get rid of unconnected graphs
        if not nx.is_connected(graph):
            return -float('inf')
        else: 
            brouwer_inequality_to_reward(graph=graph)

    if method == 'q_learning': 
        brouwer_inequality_to_reward(graph=graph)