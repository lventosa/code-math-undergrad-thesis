"""
Class representing the environment in which our RL agent will be
living when looking for a counterexample to Wagner's conjecture.
"""

import numpy as np


N_VERTICES = 19
N_EDGES = int(N_VERTICES*(N_VERTICES-1)/2)

# The input vector will have size 2*N_EDGES, where the first N_EDGES letters encode our partial word (with zeros on
#   the positions we haven't considered yet), and the next N_EDGES bits one-hot encode which letter we are considering now.
#   For instance, [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
SPACE = 2*N_EDGES 

BATCH_SIZE = 1000 # Number of episodes in each iteration


class EnvWagner():
    def __init__(self):
        self.states =  np.zeros([BATCH_SIZE, SPACE, N_EDGES], dtype=int) 
        self.actions = np.zeros([BATCH_SIZE, N_EDGES], dtype = int)
        self.next_state = np.zeros([BATCH_SIZE, SPACE], dtype = int)
        # prob = np.zeros(BATCH_SIZE) --> I think this is redundant
        self.total_rewards = np.zeros([BATCH_SIZE])
