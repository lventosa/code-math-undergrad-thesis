"""
Tabular Q-Learning to disprove a graph theory conjecture.

I'm trying to disprove conjecture 2.1 from the following paper: https://arxiv.org/abs/2104.14516
"""

import logging 
import sys
from typing import List

import numpy as np

from src.graph_theory_utils.graph_theory import build_graph_from_array
from src.rl_environments.env_wagner import (
    calculate_reward, EnvWagnerQLearning, N_VERTICES,
)


GAMMA = 0.9 # Discount rate
ALPHA = 0.2 # Learning rate when updating the Q-value


logging.basicConfig(
    stream=sys.stdout,
    datefmt='%Y-%m-%d %H:%M',
    format='%(asctime)s | %(message)s'
)
log = logging.getLogger(__name__)



def best_value_and_action(
    actions: List[int], state: int, q_table: np.array
):
    best_value = None
    for action in actions:
        action_value = q_table[state][action]
        if best_value is None or best_value < action_value:
            best_value = action_value
            best_action = action
    return best_value, best_action


def value_update(
    env: EnvWagnerQLearning, state: int, action: int, 
    reward: float, next_state: int,
) -> np.array:
    """
    This function allows us to update the Q-value using 
    the Bellman approximation. We are updating Q(s,a) using
    a "blending" technique so that our training is more stable.

    This "blending" technique consists in averaging old and 
    new values of Q using a learning rate, ALPHA. The learning 
    rate determines how much we change the current Q-value towards
    the discounted maximum of existing values.
    """
    best_value, best_action = best_value_and_action(env=env, state=next_state)
    new_value = reward + GAMMA*best_value
    env.q_table[state][action] = (1-ALPHA)*env.q_table[state][action] + ALPHA*new_value



def tabular_q_learning(): # TODO: finish this function
    """
    This is the main function.

    In tabular Q-learning, we go through ALL states and actions
    of an environment.
    """
    env = EnvWagnerQLearning() 
    env.initialize_q_table()

    for state in env.states:
        for action in env.actions:
            graph = build_graph_from_array( 
                array=env.states, 
                n_vertices=N_VERTICES,
            )
            reward = calculate_reward(graph=graph)
            
            value_update(
                env=env, state=state, action=action,
                reward=reward, next_state=state+1,
            )




    

    

if __name__ == '__main__':
    tabular_q_learning()