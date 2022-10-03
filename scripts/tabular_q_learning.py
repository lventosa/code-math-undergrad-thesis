"""
Tabular Q-Learning to disprove a graph theory conjecture.

I'm trying to disprove conjecture 2.1 from the following paper: https://arxiv.org/abs/2104.14516
"""

import logging 
import sys

import numpy as np

from src.graph_theory_utils.graph_theory import (
    build_graph_at_given_state,
    calculate_matching_number,
    calculate_max_abs_val_eigenvalue,
)
from src.rl_environments.env_wagner import calculate_reward, EnvWagner, N_ACTIONS, N_EDGES, N_VERTICES


GAMMA = 0.9 # Discount rate
ALPHA = 0.2 # Learning rate when updating the Q-value


logging.basicConfig(
    stream=sys.stdout,
    datefmt='%Y-%m-%d %H:%M',
    format='%(asctime)s | %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_q_table() -> np.array:
    """
    This function initializes the Q-table. Its dimension
    is equal to the number of states.
    """
    # The number of simple graphs of n vertices is 2^{n(n-1)/2}, where {n(n-1)/2} is 
    #   what we call N_EDGES. That is, we have 2^N_EDGES possible states and N_ACTIONS = 2.
    q_table = np.zeros([N_EDGES, N_ACTIONS])
    return q_table


def calculate_q_value(
    q_table: np.array, edge: int, 
    action: int, reward: float,
) -> float:
    """
    This function allows us to update the Q-value using 
    the Bellman approximation. We are updating Q(s,a) using
    a "blending" technique so that our training is more stable.

    This "blending" technique consists in averaging old and 
    new values of Q using a learning rate, ALPHA. The learning 
    rate determines how much we change the current Q-value towards
    the discounted maximum of existing values.
    """
    max_q_value = 0 # TODO: calculate this. Highest Q-value of any move in the next state so the Q value of the best action in the next state
    q_table[edge][action] = (1-ALPHA)*q_table[edge][action] + ALPHA*(reward + GAMMA*max_q_value)


def tabular_q_learning():
    """
    This is the main function.

    In tabular Q-learning, we go through ALL states and actions
    of an environment.
    """
    env = EnvWagner(batch_size=1) # We don't iterate through many episodes as with deep cross entropy

    q_table = initialize_q_table()

    env.states[:, N_EDGES, 0] = 1 # Pintem el primer edge
    current_edge = 0

    graph = build_graph_at_given_state(
        state=env.states[0], # [0] since we have BATCH_SIZE=1
        n_vertices=N_VERTICES
    )
    env.total_rewards[0] = calculate_reward(graph=graph)

    

    

if __name__ == '__main__':
    tabular_q_learning()