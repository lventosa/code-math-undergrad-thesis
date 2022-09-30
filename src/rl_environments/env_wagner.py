"""
Class representing the environment in which our RL agent will be
living when looking for a counterexample to Wagner's conjecture.
"""

import math
import numpy as np
import networkx as nx

from src.graph_theory_utils.graph_theory import (
    calculate_matching_number,
    calculate_max_abs_val_eigenvalue,
)


N_VERTICES = 19
N_EDGES = int(N_VERTICES*(N_VERTICES-1)/2) # A graph of n vertices has at most n(n-1)/2 edges

# At each state (pair of vertices) we only have two actions: to add an edge joining those two 
#   vertices or to leave them unconnected (no edge)
N_ACTIONS = 2 

# The input vector will have size 2*N_EDGES, where the first N_EDGES letters encode our partial word (with zeros on
#   the positions we haven't considered yet), and the next N_EDGES bits one-hot encode which letter we are considering now.
#   For instance, [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
SPACE = N_ACTIONS*N_EDGES 


class EnvWagner():
    def __init__(self, batch_size: int):
        self.states =  np.zeros([batch_size, SPACE, N_EDGES], dtype=int) 
        self.actions = np.zeros([batch_size, N_EDGES], dtype = int)
        self.next_state = np.zeros([batch_size, SPACE], dtype = int)
        self.total_rewards = np.zeros([batch_size])


def calculate_reward(graph: nx.Graph) -> float:
    """
    This function calculates the reward for our reinforcement learning
    problem. The reward depends on the conjecture we are trying to disprove. 

    Since we want to minimize lambda1 + mu (our loss function), we return the
    negative of this. Moreover, given that we want to disprove that 
    lambda1 + mu >= sqrt(N-1) + 1, if we consider our reward function to be 
    sqrt(N-1) + 1 - (lambda1 + mu) then it is enough to check whether the reward is 
    positive. In such a case, we'll have found a counterexample for the conjecture.
    """
    # The conjecture assumes our graph is connected. We get rid of unconnected graphs
    if not nx.is_connected(graph):
        return -float('inf')   
    else:
        lambda1 = calculate_max_abs_val_eigenvalue(graph=graph)
        mu = calculate_matching_number(graph=graph)
        return math.sqrt(N_VERTICES-1) + 1 - (lambda1 + mu)
