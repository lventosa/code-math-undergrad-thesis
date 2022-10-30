"""
Class representing the environment in which our RL agent will be
living when looking for a counterexample to Wagner's conjecture
as well as Brouwer's.
"""

import numpy as np

# These two constants are only valid for Wagner's conjecture
N_VERTICES = 19
N_EDGES = int(N_VERTICES*(N_VERTICES-1)/2) # A graph of n vertices has at most n(n-1)/2 edges

# At each state (pair of vertices) we only have two actions: to add an edge joining those two 
#   vertices or to leave them unconnected (no edge)
N_ACTIONS = 2 

# The input vector will have size 2*N_EDGES, where the first N_EDGES letters encode our partial word 
#   (with zeros on the positions the agent has rejected or hasn't considered yet), and the next N_EDGES 
#   bits one-hot encode which letter we are considering now. For instance, [0,1,0,0,   0,0,1,0] means we 
#   have the partial word 01 and we are considering the third letter now.
#   This constant is ONLY used in Wagner's conjecture.
SPACE = N_ACTIONS*N_EDGES 


class EnvCrossEntropy():
    def __init__(self, batch_size: int):
        self.states =  np.zeros([batch_size, SPACE, N_EDGES], dtype=int) 
        self.actions = np.zeros([batch_size, N_EDGES], dtype=int)
        self.next_state = np.zeros([batch_size, SPACE], dtype=int)
        self.total_rewards = np.zeros(batch_size, dtype=float)


class EnvQLearning():
    def __init__(self):
        self.states = np.asarray([num for num in range(N_EDGES)])
        self.actions = [0, 1]
        self.graph_current_state = np.zeros(N_EDGES, dtype=int) 
        
    def initialize_q_table(self):
        """
        This function initializes the Q-table. Its dimension
        is equal to the number of states.
        """
        # The number of simple graphs of n vertices is 2^{n(n-1)/2}, where {n(n-1)/2} is 
        #   what we call N_EDGES. That is, we have 2^N_EDGES possible states and N_ACTIONS = 2.
        self.q_table = np.random.randint(0, 10, size=(N_EDGES, N_ACTIONS))
        self.q_table = self.q_table.astype(np.float)
