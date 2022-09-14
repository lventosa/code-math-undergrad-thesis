"""
Deep cross-entropy method to disprove a graph theory conjecture. 

I am replicating A. Z. Wagner's work in the following paper: https://arxiv.org/abs/2104.14516

His code can be found here: https://github.com/zawagner22/cross-entropy-for-combinatorics
"""
import logging
import math
import sys
from typing import List

import networkx as nx
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


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
BATCH_SIZE = 100 # Number of episodes in each iteration
PERCENTILE = 70 # We will keep the first 70% episodes sorted by reward, those will be our elite

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


def build_graph_at_given_state(state: np.array) -> nx.Graph: 
    """
    This function builds a graph from an array representing the state. 
    """
    graph = nx.Graph() # TODO: add typing
    vertices_list: List[int] = list(range(N_VERTICES))
    graph.add_nodes_from(vertices_list)

    state_index: int = 0

    for i in range(0, N_VERTICES):
        for j in range(i+1, N_VERTICES):
            if state[state_index] == 1:
                graph.add_edge(i,j)
            state_index += 1

    return graph


def calculate_max_abs_val_eigenvalue(graph) -> float:
    """
    This function computes the eigenvalues of the adjacency matrix
    that corresponds to a specific graph. It returns the largest
    eigenvalue in absolute value.
    """
    adjacency_matrix = nx.adjacency_matrix(graph).todense() # TODO: typing
    eigenvals: np.array = np.linalg.eigvalsh(adjacency_matrix) 
    eigenvals_abs: np.array = abs(eigenvals)
    return max(eigenvals_abs)


def calculate_matching_number(graph) -> int: 
    """
    This function calculates all matchings for a given graph and
    it returns the matching number (i.e. the length of the maximum
    matching).
    """
    max_matching = nx.max_weight_matching(graph)
    return len(max_matching)


def calculate_reward(graph) -> float:
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


def restart_environment_and_iterate(agent): # TODO: typing (agent is the neural network)
    """
    Each time this function is called the environment is reset. That
    is, we restart the states and actions for the agent as well as the
    predicted probabilities.
    """
    logger.info('Resetting environment')
    states =  np.zeros([BATCH_SIZE, SPACE, N_EDGES], dtype=int) # TODO: typing d'aquests multidimensional arrays
    actions = np.zeros([BATCH_SIZE, N_EDGES], dtype = int)
    next_state = np.zeros([BATCH_SIZE, SPACE], dtype = int)
    # prob = np.zeros(BATCH_SIZE) --> I think this is redundant
    total_rewards = np.zeros([BATCH_SIZE])

    states[:, 0]: int = 1 # Pintem el primer edge
    current_edge: int = 0

    while True:
        prob = agent.predict(states[:,:,current_edge-1], batch_size = BATCH_SIZE) # TODO: typing (and for the rest of the function)

        for episode in range(BATCH_SIZE):
            
            if np.random.rand() < prob[episode]:
                action: int = 1
            else:
                action: int = 0

            actions[episode][current_edge]: int = action

            # We equate next state and current state for the current edge and will adjust next state depending on the action
            next_state[episode] = states[episode,:,current_edge] 

            if action == 1: # We add that edge to the graph
                next_state[episode][current_edge]: int = action	

            if current_edge + 1 < N_EDGES:
                terminal: bool = False
            else: 
                terminal: bool = True

            if terminal:
                graph = build_graph_at_given_state(state=states[episode,:,current_edge]) # TODO: make sure this is the correct state we should pass
                total_rewards[episode] = calculate_reward(graph=graph)

                if total_rewards[episode] > 0: 
                    logger.info('A counterexample has been found!')

                    # We write the graph into a text file and exit the program
                    file = open('counterexample_crossentropy.txt', 'w+')
                    content = str(states[episode,:,current_edge])
                    file.write(content)
                    file.close()
                    exit()

            if not terminal:
                states[episode,:,current_edge + 1] = next_state[episode]	

        current_edge += 1

        if terminal:
            break

    return states, actions, total_rewards


def select_elites(states_batch, actions_batch, rewards_batch): # TODO: typing and docstrings
    """
    
    """
    reward_threshold = np.percentile(rewards_batch, PERCENTILE)

    elite_states = []
    elite_actions = []
    elite_rewards = []
	
    for batch_index in range(len(states_batch)):
        if rewards_batch[batch_index] > reward_threshold:
            for item in states_batch[batch_index]:
                elite_states.append(item.tolist())
            for item in actions_batch[batch_index]:
                elite_actions.append(item)			

    elite_states = np.array(elite_states, dtype = int)	
    elite_actions = np.array(elite_actions, dtype = int)

    return elite_states, elite_actions, elite_rewards


def deep_cross_entropy_method(): # TODO: finish docstring
    """
    This is the main function
    """
    # We add three linear layers as well as their activation layers and a final output layer
    #   activated by the sigmoid function (so the final result takes values between 0 and 1)
    model = Sequential()
    model.add(Dense(FIRST_LAYER_SIZE,  activation='relu'))
    model.add(Dense(SECOND_LAYER_SIZE, activation='relu'))
    model.add(Dense(THIRD_LAYER_SIZE, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # We build the model based on input shapes received
    model.build((None, SPACE)) 

    model.compile(
        loss='binary_crossentropy', # Since we predict a binary outcome (whether the graph has a given edge or not)
        optimizer=Adam(learning_rate = LEARNING_RATE) # Wagner uses SGD as an optimizer
    ) 

    for iter in range(N_ITERATIONS):
        states, actions, total_rewards = restart_environment_and_iterate(agent=model)
        elite_states, elite_actions, elite_rewards = select_elites(
            states_batch=states,
            actions_batch=actions,
            rewards_batch=total_rewards,
        )

        # We train the model with elite states and elite actions
        model.fit(elite_states, elite_actions)

        total_rewards.sort()

if __name__ == '__main__':
    deep_cross_entropy_method()