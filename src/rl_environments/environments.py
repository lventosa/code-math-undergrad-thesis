"""
Class representing the environment in which our RL agent will be
living when looking for a counterexample to Wagner's conjecture
as well as Brouwer's.
"""

import numpy as np

# These two constants are only valid for **Wagner's conjecture**
N_VERTICES_W = 19
N_EDGES_W = int(N_VERTICES_W*(N_VERTICES_W-1)/2) # A graph of n vertices has at most n(n-1)/2 edges

# At each state (pair of vertices) we only have two actions: to add an edge joining those two 
#   vertices or to leave them unconnected (no edge)
N_ACTIONS = 2 


class EnvCrossEntropy():
    def __init__(
        self, batch_size: int, 
        space_size: int, n_edges: int,
    ):
        self.states =  np.zeros([batch_size, space_size, n_edges], dtype=int) 
        self.actions = np.zeros([batch_size, n_edges], dtype=int)
        self.next_state = np.zeros([batch_size, space_size], dtype=int)
        self.total_rewards = np.zeros(batch_size, dtype=float)


class EnvQLearning():
    def __init__(self, n_edges: int):
        self.states = np.asarray([num for num in range(n_edges)])
        self.graph_current_state = np.zeros(n_edges, dtype=int) 
        self.actions = [0, 1]
        self.n_edges = n_edges
        
    def initialize_q_table(self):
        """
        This function initializes the Q-table. Its dimension
        is equal to the number of states.
        """
        # The number of simple graphs of n vertices is 2^{n(n-1)/2}, where {n(n-1)/2} is 
        #   what we call N_EDGES. That is, we have 2^N_EDGES possible states and N_ACTIONS = 2.
        self.q_table = np.random.randint(0, 10, size=(self.n_edges, N_ACTIONS))
        self.q_table = self.q_table.astype(np.float)
