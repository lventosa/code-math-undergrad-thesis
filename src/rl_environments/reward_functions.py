"""
This script contains the reward functions for both Wagner and 
Brouwer's conjectures. 
"""

import math 
from typing import Union, Optional, List

import networkx as nx
import scipy.special

from src.graph_theory_utils.graph_theory import (
    calculate_laplacian_eigenvalues,
    calculate_matching_number,
    calculate_max_abs_val_eigenvalue,
    print_counterexample_to_file,
)
from src.rl_environments.environments import EnvCrossEntropy, EnvQLearning


def wagner_inequality_to_reward(
    graph: nx.Graph, n_vertices: int,
) -> float: 
    """
    We want to disprove that lambda1 + mu >= sqrt(N-1) + 1.
    If we consider our reward function to be sqrt(N-1) + 1 - (lambda1 + mu),
    then it is enough to check whether the reward is positive. 
    In such a case, we'll have found a counterexample for the conjecture.
    """
    lambda1 = calculate_max_abs_val_eigenvalue(graph=graph)
    mu = calculate_matching_number(graph=graph)
    return math.sqrt(n_vertices-1) + 1 - (lambda1 + mu) 


def calculate_reward_wagner(
    graph: nx.Graph, method: str, n_vertices: int,
    env: Union[EnvCrossEntropy, EnvQLearning],
    current_edge: int, episode: Optional[int] = None,
) -> float:
    """
    This function calculates the total reward for Wagner's conjecture problem. 
    If the graph is not connected (an assumption in Wagner's problem), the 
    reward takes the value of a maximum penalty.

    The maximum penalty for the Deep Cross Entropy method is -infinity. 
    However, for the tabular Q-Learning method we cannot establish the maximum 
    penalty to be -infinity since this would make the recursive sum that is 
    computed to obtain the Q-values to be -infinity. Therefore, the maximum 
    penalty for the Q-Learning method is an arbitrarily low number.
    """
    if method == 'cross_entropy':
        max_penalty = -float('inf')
    elif method == 'q_learning':
        max_penalty = -10000

    # The conjecture assumes our graph is connected. 
    #   We get rid of unconnected graphs.
    if not nx.is_connected(graph):
        return max_penalty

    else: 
        reward = wagner_inequality_to_reward(
            graph=graph, n_vertices=n_vertices,
        )
        if reward > 0: 
            if method == 'cross_entropy':
                counterexample = env.states[episode,:,current_edge]
            elif method == 'q_learning': 
                counterexample = env.graph_current_state
            print_counterexample_to_file(
                method=method, 
                conjecture='wagner',
                counterexample=counterexample,
            )
        else: 
            return reward


def brouwer_inequality_to_reward(
    method: str, t: int, n_edges: int,
    eigenvals_list: List[float],
    env: Union[EnvCrossEntropy, EnvQLearning],
    current_edge: int, conjecture: str,
    episode: Optional[int],
) -> float:
    """
    We want to disprove that for all t in [1,n], the sum of the first t Laplace
    eigenvalues is smaller or equal than the number of edges of the graph plus
    t+1 choose 2. 
    If we consider our reward function to be the sum of the first t Laplace 
    eigenvalues minus e and minus t+1 choose 2, then it is enough to check 
    whether the reward is positive. In such a case, we'll have found a 
    counterexample for the conjecture.
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
            conjecture=conjecture,
            counterexample=counterexample,
        )  
    else: 
        return reward_t


def calculate_reward_brouwer(
    graph: nx.Graph, method: str, signless_laplacian: bool,
    env: Union[EnvCrossEntropy, EnvQLearning],
    current_edge: int, episode: Optional[int] = None,
) -> float:
    """
    This function calculates the total reward for Brouwer's conjecture problem. 

    It defines a maximum penalty depending on the method used and assigns this 
    maximum penalty to graphs for which the Brouwer conjecture has been proved 
    to be true. These graphs are: 
        - Regular graphs
        - Trees
        - Unicyclic graphs
        - Bicyclic graphs
        - Split graphs

    The maximum penalty for the Deep Cross Entropy method is -infinity. However,
    for the tabular Q-Learning method we cannot establish the maximum penalty 
    to be -infinity since this would make the recursive sum that is computed 
    to obtain the Q-values to be -infinity. Therefore, the maximum penalty for 
    the Q-Learning method is an arbitrarily low number.
    """
    if method == 'cross_entropy':
        max_penalty = -float('inf')
    elif method == 'q_learning':
        max_penalty = -100000

    graph_c = nx.complement(graph)

    # We replace each undirected edge with two directed edges so that we 
    #   can check if the graph contains simple cycles with simple_cycles()
    #   as simple_cycles() only works with directed graphs.
    graph_dir = graph.to_directed()

    if not signless_laplacian:
        if nx.is_regular(graph):
            return max_penalty

        elif nx.is_tree(graph):
            return max_penalty

        elif ( 
            nx.is_connected(graph) and 
            len(list(nx.simple_cycles(graph_dir))) == 1
        ): # unicyclic
            return max_penalty

        elif len(list(nx.simple_cycles(graph_dir))) == 2: # bicyclic
            return max_penalty

        elif nx.is_chordal(graph) and nx.is_chordal(graph_c): # split graph
            return max_penalty

    eigenvals = calculate_laplacian_eigenvalues(
        graph=graph, signless_laplacian=signless_laplacian,
    )
    n_eigenvals = len(eigenvals)
    n_edges = graph.number_of_edges()

    if signless_laplacian:
        conjecture = 'brouwer_signless_laplacian'
    else: 
        conjecture = 'brouwer'

    # Total reward will be the maximum reward_t
    max_reward = 0 

    for t in range(1, n_eigenvals+1):
        reward_t = brouwer_inequality_to_reward(
            method=method, n_edges=n_edges, 
            eigenvals_list=list(eigenvals),
            env=env, episode=episode, t=t,
            current_edge=current_edge,
            conjecture=conjecture,
        )
        if reward_t > max_reward:
            max_reward = reward_t           

    return max_reward
