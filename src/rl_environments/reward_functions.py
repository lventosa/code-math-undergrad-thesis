"""
This script contains the reward functions for both Wagner and 
Brouwer's conjectures. 
"""

import math 
from typing import Union, Optional, List

import networkx as nx
import numpy as np

from src.graph_theory_utils.graph_theory import (
    calculate_laplacian_eigenvalues,
    calculate_matching_number,
    calculate_max_abs_val_eigenvalue,
    print_counterexample_to_file,
)
from src.rl_environments.environments import N_VERTICES, EnvCrossEntropy, EnvQLearning


def wagner_inequality_to_reward(graph: nx.Graph) -> float: 
    """
    We want to disprove that lambda1 + mu >= sqrt(N-1) + 1.
    If we consider our reward function to be sqrt(N-1) + 1 - (lambda1 + mu),
    then it is enough to check whether the reward is positive. 
    In such a case, we'll have found a counterexample for the conjecture.
    """
    lambda1 = calculate_max_abs_val_eigenvalue(graph=graph)
    mu = calculate_matching_number(graph=graph)
    return math.sqrt(N_VERTICES-1) + 1 - (lambda1 + mu) 


def calculate_reward_wagner(
    graph: nx.Graph, method: str, 
    env: Union[EnvCrossEntropy, EnvQLearning],
    current_edge: int, episode: Optional[int],
) -> float:
    """
    This function calculates the total reward for Wagner's conjecture problem. 
    """
    if method == 'cross_entropy':
        # The conjecture assumes our graph is connected. We get rid of unconnected graphs.
        if not nx.is_connected(graph):
            return -float('inf')

    reward = wagner_inequality_to_reward(graph=graph)

    if reward > 0: 
        if method == 'cross_entropy':
            counterexample = env.states[episode,:,current_edge]
        if method == 'q_learning': 
            counterexample = env.states[current_edge], # TODO: make sure this is correct

        print_counterexample_to_file(
            method=method, 
            counterexample=counterexample,
        )

    return reward


def brouwer_inequality_to_reward(
    method: str, t: int, n_edges: int,
    eigenvals_list: List[float],
    env: Union[EnvCrossEntropy, EnvQLearning],
    current_edge: int, episode: Optional[int],
) -> float:
    """
    We want to disprove that for all t in [1,n], the sum of the first t Laplace
    eigenvalues is smaller or equal than the number of edges of the graph plus
    t+1 choose 2. 
    If we consider our reward function to be the sum of the first t Laplace eigenvalues
    minus e and minus t+1 choose 2, then it is enough to check whether the reward is 
    positive. In such a case, we'll have found a counterexample for the conjecture.
    """
    t_eigenvals = eigenvals_list[:t]
    cum_sum_eigenvals = np.cumsum(t_eigenvals)
    reward_t = cum_sum_eigenvals - n_edges - math.comb(t+1, 2)   

    # If reward_t is positive for any t, we've found a counterexample  
    if reward_t > 0:
        if method == 'cross_entropy': 
            counterexample = env.states[episode,:,current_edge]
        if method == 'q_learning': 
            counterexample = counterexample = env.states[current_edge], # TODO: make sure this is correct

        print_counterexample_to_file(
            method=method,
            counterexample=counterexample,
        )

    return reward_t


def calculate_reward_brouwer(
    graph: nx.Graph, method: str,
    env: Union[EnvCrossEntropy, EnvQLearning],
    current_edge: int, episode: Optional[int],
) -> float:
    """
    This function calculates the total reward for Brouwer's conjecture problem. 
    """
    if method == 'cross_entropy':
        # The conjecture assumes our graph is connected. We get rid of unconnected graphs.
        if not nx.is_connected(graph):
            return -float('inf')

    eigenvals_list = calculate_laplacian_eigenvalues(graph=graph)
    n_eigenvals = len(eigenvals_list)
    n_edges = graph.number_of_edges()

    total_reward = 0

    for t in range(1, n_eigenvals+1):
        reward_t = brouwer_inequality_to_reward(
            method=method, n_edges=n_edges,
            n_eigenvals=n_eigenvals, t=t,
            env=env, episode=episode,
            current_edge=current_edge,
        )
        total_reward += reward_t

    return total_reward
