"""
Deep cross-entropy method to disprove a graph theory conjecture. 

I am replicating A. Z. Wagner's work in conjecture 2.1 from the following paper: 
https://arxiv.org/abs/2104.14516

His code can be found here: 
https://github.com/zawagner22/cross-entropy-for-combinatorics

This script further tries to disprove Brouwer's conjecture.
"""

import logging
from typing import Tuple, Optional

from keras.models import Sequential
import numpy as np
import tensorflow as tf

from src.graph_theory_utils.graph_theory import build_graph_from_array
from src.models.deep_cross_entropy_model import create_neural_network_model
from src.rl_environments.environments import (
    EnvCrossEntropy, N_POSSIBLE_EDGES_W, N_VERTICES_W, N_ACTIONS,
)
from src.rl_environments.reward_functions import (
    calculate_reward_brouwer, 
    calculate_reward_wagner,
)


# The following values are set arbitrarily and can be modified 
#   for experimental purposes
MAX_ITER = 100000
BATCH_SIZE = 1000 # Number of episodes in each iteration
PERCENTILE = 93 # Threshold for elite states and actions classification


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s | %(message)s',
    level=logging.INFO,
    handlers = [
        logging.FileHandler('logs_cross_entropy.log'),
        logging.StreamHandler()
    ]
)


def restart_environment_and_iterate(
    agent: Sequential, conjecture: str,
    n_possible_edges: int, n_vertices: int,
    space_size: int, signless_laplacian: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    Each time this function is called the environment is reset. That
    is, we restart the states and actions for the agent as well as the
    predicted probabilities.
    """
    log.info('Resetting environment')
    env = EnvCrossEntropy(
        batch_size=BATCH_SIZE, 
        space_size=space_size, 
        n_vertices=n_vertices,
        n_possible_edges=n_possible_edges,
    )

    env.states[:, n_possible_edges, 0] = 1 # We add the first edge
    current_edge = 0

    while True:
        states_tensor = tf.convert_to_tensor(env.states[:,:,current_edge])
        prob = agent.predict(states_tensor, BATCH_SIZE) 

        for episode in range(BATCH_SIZE):
            if np.random.rand() < prob[episode]:
                action = 1
            else:
                action = 0

            env.actions[episode][current_edge] = action

            # We equate next state and current state for the current edge 
            #   and will adjust next state depending on the action
            env.next_state[episode] = env.states[episode,:,current_edge] 

            if action == 1: # We add that edge to the graph
                env.next_state[episode][current_edge] = action

            # We update the edge we are looking at	
            env.next_state[episode][n_possible_edges + current_edge] = 0
            if current_edge + 1 < n_possible_edges:
                env.next_state[episode][n_possible_edges + current_edge+1] = 1
                terminal = False
            else: 
                terminal = True

            if terminal:
                graph = build_graph_from_array(
                    array=env.next_state[episode], 
                    n_vertices=n_vertices,
                ) 
                # We print the array representing the graph
                # print(env.next_state[episode])

                if conjecture == 'wagner':
                    env.total_rewards[episode] = calculate_reward_wagner(
                        graph=graph, method='cross_entropy', 
                        env=env, n_vertices=n_vertices,
                        episode=episode, 
                        current_edge=current_edge,
                    )

                if conjecture == 'brouwer':
                    env.total_rewards[episode] = calculate_reward_brouwer(
                        graph=graph, method='cross_entropy', env=env,
                        episode=episode, current_edge=current_edge,
                        signless_laplacian=signless_laplacian,
                    )

            if not terminal:
                env.states[episode,:,current_edge+1] = env.next_state[episode]	

        current_edge += 1

        if terminal:
            break

    return env.states, env.actions, env.total_rewards


def select_elites(
    states_batch: np.ndarray, actions_batch: np.ndarray, 
    rewards_batch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function selects the top states and actions given 
    the threshold set by an arbitrary percentage.
    """
    reward_threshold = np.percentile(rewards_batch, PERCENTILE)

    elite_states = []
    elite_actions = []
	
    for batch_index in range(len(states_batch)):
        if rewards_batch[batch_index] > reward_threshold:
            for item in states_batch[batch_index]:
                elite_states.append(item.tolist())
            for item in actions_batch[batch_index]:
                elite_actions.append(item)

    elite_states = np.array(elite_states)
    elite_actions = np.array(elite_actions)

    return elite_states, elite_actions


def deep_cross_entropy_method(
    conjecture: str, n_vertices: int, 
    n_possible_edges: int, n_actions: int,
    signless_laplacian: Optional[bool] = False,
) -> None: 
    """
    This is the main function.

    The signless laplacian argument is only relevant for
    Brouwer's conjecture.
    
    At each iteration we restart the environment and obtain the
    resulting states, actions and rewards. From those values we
    select the elite states and actions that will be used to train
    our agent.

    The input vector (space_size) will have size 2*n_possible_edges, where the 
    first n_possible_edges letters encode our partial word (with zeros on the 
    positions the agent has rejected or hasn't considered yet), and the 
    next n_possible_edges bits one-hot encode which letter we are considering 
    now. For instance, [0,1,0,0,   0,0,1,0] means we have the partial word 01 
    and we are considering the third letter now.
    This parameter is ONLY used for Wagner's conjecture.
    """
    space_size = n_actions*n_possible_edges
    model = create_neural_network_model(space_size=space_size)

    for iter in range(MAX_ITER):
        try: 
            states, actions, total_rewards = restart_environment_and_iterate(
                agent=model, conjecture=conjecture, 
                n_possible_edges=n_possible_edges,
                n_vertices=n_vertices, space_size=space_size,
                signless_laplacian=signless_laplacian,
            )
            states = np.transpose(states, axes=[0,2,1])
            elite_states, elite_actions = select_elites(
                states_batch=states,
                actions_batch=actions,
                rewards_batch=total_rewards,
            )

            # We train the model with elite states and elite actions
            elite_states_tensor = tf.convert_to_tensor(elite_states)
            elite_actions_tensor = tf.convert_to_tensor(elite_actions)
            model.fit(elite_states_tensor, elite_actions_tensor)

        except Exception as error:
            log.error(error)

        log.info(f'Iteration #{iter+1} done')


if __name__ == '__main__': 
    # Wagner's conjecture
    log.info("Running Deep Cross Entropy for Wagner's conjecture")
    deep_cross_entropy_method(
        conjecture='wagner', n_vertices=N_VERTICES_W,
        n_possible_edges=N_POSSIBLE_EDGES_W, n_actions=N_ACTIONS,
    )

    # Brouwer's conjecture
    #   From 11 to 20 (it's been proved true for n_vertices<11)
    for n_vertices in range(11, 21): 
        # A graph of n vertices has at most n(n-1)/2 edges
        n_possible_edges = int(n_vertices*(n_vertices-1)/2) 
        log.info(
            f"Running Deep Cross Entropy for Brouwer's"
            f"conjecture for {n_vertices}-vertex graphs"
        )
        deep_cross_entropy_method(
            conjecture='brouwer', n_vertices=n_vertices,
            n_possible_edges=n_possible_edges, n_actions=N_ACTIONS,
        )

    # Variant of Brouwer's conjecture
    #   We only consider graphs of at most 10 vertices
    for n_vertices in range(2, 11):
        n_possible_edges = int(n_vertices*(n_vertices-1)/2) 
        log.info(
            f"Running Deep Cross Entropy for Brouwer's conjecture"
            f"with signless Laplacian for {n_vertices}-vertex graphs"
        )
        deep_cross_entropy_method(
            conjecture='brouwer', n_vertices=n_vertices,
            n_possible_edges=n_possible_edges, n_actions=N_ACTIONS,
            signless_laplacian=True,
        )
