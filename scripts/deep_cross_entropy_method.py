"""
Deep cross-entropy method to disprove a graph theory conjecture. 

I am replicating A. Z. Wagner's work in conjecture 2.1 from the following paper: https://arxiv.org/abs/2104.14516

His code can be found here: https://github.com/zawagner22/cross-entropy-for-combinatorics
"""

import logging
import math
import sys
from typing import Tuple

from keras.models import Sequential
import networkx as nx
import numpy as np

from src.graph_theory_utils.graph_theory import (
    build_graph_at_given_state,
    calculate_matching_number,
    calculate_max_abs_val_eigenvalue,
)
from src.models.deep_cross_entropy_model import DeepCrossEntropyModel
from src.rl_environments.env_wagner import EnvWagner


N_VERTICES = 19
N_EDGES = int(N_VERTICES*(N_VERTICES-1)/2)

# The input vector will have size 2*N_EDGES, where the first N_EDGES letters encode our partial word (with zeros on
#   the positions we haven't considered yet), and the next N_EDGES bits one-hot encode which letter we are considering now.
#   For instance, [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
SPACE = 2*N_EDGES 

# At each state (pair of vertices) we only have two actions: to add an edge joining those two 
#   vertices or to leave them unconnected (no edge)
N_ACTIONS = 2 

# The following values are set arbitrarily and can be modified for experimental purposes
N_ITERATIONS = 100000
LEARNING_RATE = 0.0001 
BATCH_SIZE = 1000 # Number of episodes in each iteration
PERCENTILE = 93 # We will keep the first 93% episodes sorted by reward, those will be our elite

# Number of neurons in each hidden layer (arbitrary)
FIRST_LAYER_SIZE = 128 
SECOND_LAYER_SIZE = 64
THIRD_LAYER_SIZE = 4


logging.basicConfig(
    stream=sys.stdout,
    datefmt='%Y-%m-%d %H:%M',
    format='%(asctime)s | %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_reward(graph: nx.Graph) -> float:
    """
    This function calculates the reward for our reinforcement learning
    problem. The reward depends on the conjecture we are trying to disprove. 

    Since we want to minimize lambda1 + mu (our loss function), we return the
    negative of this. Moreover, given that we want to disprove that 
    lambda1 + mu >= sqrt(N-1) + 1, if we consider our reward function to be 
    sqrt(N-1) + 1 - (lambda1 + mu) then it is enough to check whether the reward is 
    positive. In such a case, we'll have found a counterexample for the conjecture.
    """
    # The conjecture assumes our graph is connected. We get rid of unconnected graphs
    if not nx.is_connected(graph):
        return -float('inf')   
    else:
        lambda1 = calculate_max_abs_val_eigenvalue(graph=graph)
        mu = calculate_matching_number(graph=graph)
        return math.sqrt(N_VERTICES-1) + 1 - (lambda1 + mu)


def restart_environment_and_iterate(agent: Sequential) -> Tuple[np.array, np.array, np.array]: 
    """
    Each time this function is called the environment is reset. That
    is, we restart the states and actions for the agent as well as the
    predicted probabilities.
    """
    logger.info('Resetting environment')
    env = EnvWagner()

    env.states[:, 0] = 1 # Pintem el primer edge
    current_edge = 0

    while True:
        prob = agent.model.predict(env.states[:,:,current_edge-1], batch_size = BATCH_SIZE) 

        for episode in range(BATCH_SIZE):
            if np.random.rand() < prob[episode]:
                action = 1
            else:
                action = 0

            env.actions[episode][current_edge] = action

            # We equate next state and current state for the current edge and will adjust next state depending on the action
            env.next_state[episode] = env.states[episode,:,current_edge] 

            if action == 1: # We add that edge to the graph
                env.next_state[episode][current_edge] = action	

            if current_edge + 1 < N_EDGES:
                terminal = False
            else: 
                terminal = True

            if terminal:
                graph = build_graph_at_given_state(
                    state=env.states[episode,:,current_edge], # TODO: make sure this is the correct state we should pass
                    n_vertices=N_VERTICES,
                ) 
                env.total_rewards[episode] = calculate_reward(graph=graph)

                if env.total_rewards[episode] > 0: 
                    logger.info('A counterexample has been found!')

                    # We write the graph into a text file and exit the program
                    file = open('counterexample_crossentropy.txt', 'w+')
                    content = str(env.states[episode,:,current_edge])
                    file.write(content)
                    file.close()
                    exit()

            if not terminal:
                env.states[episode,:,current_edge + 1] = env.next_state[episode]	

        current_edge += 1

        if terminal:
            break

    return env.states, env.actions, env.total_rewards


def select_elites(
    states_batch: np.array, actions_batch: np.array, rewards_batch: np.array,
) -> Tuple[np.array, np.array, np.array]:
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

    elite_states = np.array(elite_states, dtype = int)	
    elite_actions = np.array(elite_actions, dtype = int)

    return elite_states, elite_actions


def deep_cross_entropy_method(): 
    """
    This is the main function.

    We create, build and compile a three-layer neural network. 
    
    At each iteration we restart the environment and obtain the
    resulting states, actions and rewards. From those values we
    select the elite states and actions that will be used to train
    our agent.
    """
    model = DeepCrossEntropyModel()
    model.build_and_compile_model()

    for iter in range(N_ITERATIONS):
        states, actions, total_rewards = restart_environment_and_iterate(agent=model)
        states = np.transpose(states, axes=[0,2,1])
        elite_states, elite_actions = select_elites(
            states_batch=states,
            actions_batch=actions,
            rewards_batch=total_rewards,
        )

        # We train the model with elite states and elite actions
        model.fit(elite_states, elite_actions)

        total_rewards.sort() # NOTE: this is useless unless we print the total rewards somewhere


if __name__ == '__main__':
    deep_cross_entropy_method()