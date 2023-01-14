"""
Tabular and Deep Q-Learning to disprove a graph theory conjecture.

I'm trying to disprove conjecture 2.1 from the paper 
https://arxiv.org/abs/2104.14516 as well as Brouwer's conjecture.
"""

import logging 
from os import getenv
from typing import Optional

from logtail import LogtailHandler

from src.graph_theory_utils.graph_theory import build_graph_from_array
from src.rl_environments.environments import (
    EnvQLearning, N_VERTICES_W, N_POSSIBLE_EDGES_W
)
from src.rl_environments.reward_functions import (
    calculate_reward_brouwer, 
    calculate_reward_wagner,
)


GAMMA = 0.9 # Discount rate
ALPHA = 0.2 # Learning rate when updating the Q-value
MAX_ITER = 10000


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Uncomment this to send logs to logtail
logtail_handler = LogtailHandler(source_token=getenv('LOGTAIL_TOKEN'))

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s | %(message)s',
    level=logging.INFO,
    handlers = [
        logging.FileHandler('logs_qlearning.log'),
        logging.StreamHandler(),
        logtail_handler # Uncomment this to send logs to logtail
    ]
)


def value_update(
    env: EnvQLearning, state: int, 
    n_vertices: int, next_state: int, 
    conjecture: str, signless_laplacian: bool, 
) -> None:
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
        n_vertices=n_vertices,
    )

    # We print the array representing the graph
    # print(env.graph_current_state)

    if conjecture == 'wagner':
        reward = calculate_reward_wagner(
            graph=graph_best_action,
            method='q_learning', 
            n_vertices=n_vertices,
            env=env, current_edge=state, 
        )    

    if conjecture == 'brouwer': 
        reward = calculate_reward_brouwer(
            graph=graph_best_action, method='q_learning',
            env=env, current_edge=state,
            signless_laplacian=signless_laplacian,
        )                    

    new_value = reward + GAMMA*best_value
    env.q_table[state][best_action] = (
        (1-ALPHA)*env.q_table[state][best_action] + ALPHA*new_value
    )


def tabular_q_learning(
    conjecture: str, n_possible_edges: int, n_vertices: int,
    signless_laplacian: Optional[bool] = False
) -> None: 
    """
    This is the main function for Tabular Q-Learning.

    The signless laplacian argument is only relevant
    for Brouwer's conjecture.
    """
    env = EnvQLearning(
        n_vertices=n_vertices,
        n_possible_edges=n_possible_edges
    ) 
    env.initialize_q_table()

    for iter in range(MAX_ITER):
        for state in env.states:
            try: 
                if state >= len(env.q_table)-1:
                    state = -1 

                value_update(
                    env=env, state=state, 
                    n_vertices=n_vertices,
                    next_state=state+1,
                    conjecture=conjecture,
                    signless_laplacian=signless_laplacian,
                )

            except Exception as error:
                log.error(error)

        log.info(f'Iteration #{iter+1} done')


if __name__ == '__main__':
    # Wagner's conjecture
    log.info(f"Running tabular Q-Learning for Wagner's conjecture")
    tabular_q_learning(
        conjecture='wagner', 
        n_vertices=N_VERTICES_W,
        n_possible_edges=N_POSSIBLE_EDGES_W,
    )

    # Brouwer's conjecture 
    #   From 11 to 20 (it's been proved true for n_vertices<11)
    for n_vertices in range(11, 21):
        # A graph of n vertices has at most n(n-1)/2 edges
        n_possible_edges = int(n_vertices*(n_vertices-1)/2) 
        log.info(
            f"Running tabular Q-Learning for Brouwer's "
            f"conjecture for {n_vertices}-vertex graphs"
        )
        tabular_q_learning(
            conjecture='brouwer', n_vertices=n_vertices, 
            n_possible_edges=n_possible_edges,
        ) 

    # Variant of Brouwer's conjecture
    #   We only consider graphs of at most 10 vertices
    for n_vertices in range(2, 11):
        n_possible_edges = int(n_vertices*(n_vertices-1)/2)
        log.info(
            f"Running tabular Q-Learning for Brouwer's conjecture "
            f"with signless Laplacian for {n_vertices}-vertex graphs"
        )
        tabular_q_learning(
            conjecture='brouwer', n_vertices=n_vertices,
            n_possible_edges=n_possible_edges, signless_laplacian=True
        ) 
