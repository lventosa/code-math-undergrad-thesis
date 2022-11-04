"""
This script contains the reward functions for both Wagner and 
Brouwer's conjectures. 
"""

import math 
from typing import Union, Optional, List

import networkx as nx
import sagemath as sg
import scipy.special

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
    current_edge: int, episode: Optional[int] = None,
) -> float:
    """
    This function calculates the total reward for Wagner's conjecture problem. 
    """
    if method == 'cross_entropy':
        max_penalty = -float('inf')

    # Since q_learning sums reward values, we cannot assign -inf to max_penalty.
    #   Otherwise, all Q-values would be -inf
    elif method == 'q_learning':
        max_penalty = -10000

    # The conjecture assumes our graph is connected. We get rid of unconnected graphs.
    if not nx.is_connected(graph):
        return max_penalty

    else: 
        reward = wagner_inequality_to_reward(graph=graph)
        if reward > 0: 
            if method == 'cross_entropy':
                counterexample = env.states[episode,:,current_edge]
            elif method == 'q_learning': 
                counterexample = env.graph_current_state
            print_counterexample_to_file(
                method=method, 
                counterexample=counterexample,
            )
        else: 
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
    sum_eigenvals = sum(t_eigenvals)
    reward_t = sum_eigenvals - float(n_edges) - scipy.special.comb(t+1, 2)   

    # If reward_t is positive for any t, we have found a counterexample.
    if reward_t > 0:
        if method == 'cross_entropy': 
            counterexample = env.states[episode,:,current_edge]
        elif method == 'q_learning': 
            counterexample = env.graph_current_state
        print_counterexample_to_file(
            method=method,
            counterexample=counterexample,
        )  
    else: 
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
        max_penalty = -float('inf')

    # Since q_learning sums reward values, we cannot assign -inf to max_penalty.
    #   Otherwise, all Q-values would be -inf
    elif method == 'q_learning':
        max_penalty = -100000

    # Graphs for which the Brouwer's conjecture has been proved to 
    #   be true need to have a high penalty attached 
    elif nx.is_regular(graph):
        return max_penalty

    if nx.is_tree(graph):
        return max_penalty

    elif nx.is_connected(graph): 
        if len(nx.simple_cycles(graph)) in [1, 2]: # unicyclic or bicyclic
            return max_penalty

    elif sg.is_cograph(graph): # This includes threshold graphs anc complete k-partite graphs
        return max_penalty

    elif sg.is_split(graph): 
        return max_penalty

    else:
        eigenvals = calculate_laplacian_eigenvalues(graph=graph)
        n_eigenvals = len(eigenvals)
        n_edges = graph.number_of_edges()

        # Total reward accounted as the sum of all rewards fot t in [1,n]
        total_reward = 0 

        for t in range(1, n_eigenvals+1):
            reward_t = brouwer_inequality_to_reward(
                method=method, n_edges=n_edges, 
                eigenvals_list=list(eigenvals),
                env=env, episode=episode, t=t,
                current_edge=current_edge,
            )
            total_reward += reward_t

        return total_reward
