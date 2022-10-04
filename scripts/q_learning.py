"""
Tabular and Deep Q-Learning to disprove a graph theory conjecture.

I'm trying to disprove conjecture 2.1 from the following paper: https://arxiv.org/abs/2104.14516
"""

import logging 
import sys

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


def value_update(
    env: EnvWagnerQLearning, state: int, 
    next_state: int, reward: float, 
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
    best_value = None
    for action in env.actions:
        action_value = env.q_table[next_state][action] 
        if best_value is None or best_value < action_value:
            best_value = action_value
            best_action = action

    new_value = reward + GAMMA*best_value
    env.q_table[state][best_action] = (1-ALPHA)*env.q_table[state][best_action] + ALPHA*new_value


def tabular_q_learning(): # TODO: no syntax erros but behavior is not as expected
    """
    This is the main function for Tabular Q-Learning.
    """
    env = EnvWagnerQLearning() 
    env.initialize_q_table()

    for state in env.states:
            graph = build_graph_from_array( 
                array=env.states, 
                n_vertices=N_VERTICES,
            )
            reward = calculate_reward(graph=graph)

            if reward <= 0: 
                if reward == -float('inf'):
                    continue
                else:
                    value_update(
                        env=env, state=state, 
                        reward=reward, 
                        next_state=state+1,
                    )
            else: 
                log.info('A counterexample has been found!')

                # We write the graph into a text file and exit the program
                file = open('counterexample_tabular_q_learning.txt', 'w+')
                content = str(env.q_table) # TODO: determine what it makes more sense to write
                file.write(content)
                file.close()
                exit()


def deep_q_learning(): # TODO: finish this function
    """
    This is the main function for Deep Q-Learning.
    """

    env = EnvWagnerQLearning
    env.initialize_q_table()


if __name__ == '__main__':
    tabular_q_learning()