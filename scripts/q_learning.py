"""
Tabular and Deep Q-Learning to disprove a graph theory conjecture.

I'm trying to disprove conjecture 2.1 from the following paper: 
https://arxiv.org/abs/2104.14516
"""

import logging 
import sys
from typing import Tuple

import networkx as nx

from src.graph_theory_utils.graph_theory import build_graph_from_array
from src.rl_environments.environments import EnvQLearning, N_VERTICES
from src.rl_environments.reward_functions import calculate_reward_wagner


GAMMA = 0.9 # Discount rate
ALPHA = 0.2 # Learning rate when updating the Q-value


logging.basicConfig(
    stream=sys.stdout,
    datefmt='%Y-%m-%d %H:%M',
    format='%(asctime)s | %(message)s'
)
log = logging.getLogger(__name__)


def value_update(
    env: EnvQLearning, state: int, next_state: int,
) -> Tuple[nx.Graph, float]:
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

    env.graph_current_state[state] = best_action

    graph_best_action = build_graph_from_array( 
        array=env.graph_current_state, 
        n_vertices=N_VERTICES,
    )
    reward = calculate_reward_wagner(graph=graph_best_action, method='q_learning')                        

    new_value = reward + GAMMA*best_value
    env.q_table[state][best_action] = (1-ALPHA)*env.q_table[state][best_action] + ALPHA*new_value

    return graph_best_action, reward


def tabular_q_learning(): 
    """
    This is the main function for Tabular Q-Learning.
    """
    env = EnvQLearning() 
    env.initialize_q_table()

    iter = 0

    while True: 
        iter += 1
        for state in env.states:
            if state >= len(env.q_table)-1:
                state = -1 

            graph_best_action, reward = value_update(
                env=env, state=state, 
                next_state=state+1,
            )

            if reward >= 0: 
                if nx.is_connected(graph_best_action):
                    log.info('A counterexample has been found!')

                    # We write the graph into a text file and exit the program
                    file = open('counterexample_tabular_q_learning.txt', 'w+')
                    content = str(env.q_table) # TODO: determine what it makes more sense to write
                    file.write(content)
                    file.close()
                    exit()

                else:
                    log.info(
                        'The graph that has been found is not connected --> invalid counterexample'
                    )

        print(f'Iteration #{iter} done')


if __name__ == '__main__':
    tabular_q_learning()
