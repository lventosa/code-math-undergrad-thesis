"""
Deep cross-entropy method to disprove a graph theory conjecture. 

I am replicating A. Z. Wagner's work in conjecture 2.1 from the following paper: 
https://arxiv.org/abs/2104.14516

His code can be found here: 
https://github.com/zawagner22/cross-entropy-for-combinatorics

This script further tries to disprove Brouwer's conjecture.
"""

import logging
import sys
from typing import Tuple

from keras.models import Sequential
import numpy as np

from src.graph_theory_utils.graph_theory import build_graph_from_array
from src.models.deep_cross_entropy_model import model
from src.rl_environments.environments import EnvCrossEntropy, N_VERTICES, N_EDGES
from src.rl_environments.reward_functions import (
    calculate_reward_brouwer, 
    calculate_reward_wagner,
)


# The following values are set arbitrarily and can be modified for experimental purposes
N_ITERATIONS = 100000
BATCH_SIZE = 1000 # Number of episodes in each iteration
PERCENTILE = 93 # Threshold for elite states and actions classification


logging.basicConfig(
    stream=sys.stdout,
    datefmt='%Y-%m-%d %H:%M',
    format='%(asctime)s | %(message)s'
)
log = logging.getLogger(__name__)


def restart_environment_and_iterate(
    agent: Sequential, conjecture: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    Each time this function is called the environment is reset. That
    is, we restart the states and actions for the agent as well as the
    predicted probabilities.
    """
    log.info('Resetting environment')
    env = EnvCrossEntropy(batch_size=BATCH_SIZE)

    env.states[:, N_EDGES, 0] = 1 # We add the first edge
    current_edge = 0

    while True:
        prob = agent.predict(env.states[:,:,current_edge], batch_size=BATCH_SIZE) 

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
            env.next_state[episode][N_EDGES+current_edge] = 0
            if current_edge + 1 < N_EDGES:
                env.next_state[episode][N_EDGES+current_edge+1] = 1
                terminal = False
            else: 
                terminal = True

            if terminal:
                graph = build_graph_from_array(
                    array=env.next_state[episode], 
                    n_vertices=N_VERTICES,
                ) 

                if conjecture == 'wagner':
                    env.total_rewards[episode] = calculate_reward_wagner(
                        graph=graph, method='cross_entropy', env=env,
                        episode=episode, current_edge=current_edge,
                    )

                if conjecture == 'brouwer':
                    env.total_rewards[episode] = calculate_reward_brouwer(
                        graph=graph, method='cross_entropy', env=env,
                        episode=episode, current_edge=current_edge,
                    )

            if not terminal:
                env.states[episode,:,current_edge+1] = env.next_state[episode]	

        current_edge += 1

        if terminal:
            break

    return env.states, env.actions, env.total_rewards


def select_elites(
    states_batch: np.ndarray, actions_batch: np.ndarray, rewards_batch: np.ndarray,
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

    elite_states = np.array(np.array(elite_states))	
    elite_actions = np.array(np.array(elite_actions))

    return elite_states, elite_actions


def deep_cross_entropy_method(conjecture: str): 
    """
    This is the main function.

    We create, build and compile a three-layer neural network. 
    
    At each iteration we restart the environment and obtain the
    resulting states, actions and rewards. From those values we
    select the elite states and actions that will be used to train
    our agent.
    """
    for iter in range(N_ITERATIONS):
        states, actions, total_rewards = restart_environment_and_iterate(
            agent=model, conjecture=conjecture,
        )
        states = np.transpose(states, axes=[0,2,1])
        elite_states, elite_actions = select_elites(
            states_batch=states,
            actions_batch=actions,
            rewards_batch=total_rewards,
        )

        # We train the model with elite states and elite actions
        model.fit(elite_states, elite_actions)


if __name__ == '__main__':
    # deep_cross_entropy_method(conjecture='wagner')
    deep_cross_entropy_method(conjecture='brouwer')
