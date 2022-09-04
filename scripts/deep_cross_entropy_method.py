"""
Deep cross-entropy method to disprove a graph theory conjecture. 

I am replicating A. Z. Wagner's work in the following paper: https://arxiv.org/abs/2104.14516

His code can be found here: https://github.com/zawagner22/cross-entropy-for-combinatorics
"""

import math
from typing import List

import networkx as nx
import numpy as np

import torch 
import torch.nn as nn


N_VERTICES: int = 19
N_EDGES: int = int(N_VERTICES*(N_VERTICES-1)/2)

# At each state (pair of vertices) we only have two actions: to add an edge joining those two 
#   vertices or to leave them unconnected (no edge)
N_ACTIONS: int = 2 

# The following values are set arbitrarily and can be modified for experimental purposes
N_ITERATIONS: int = 100000 
BATCH_SIZE: int = 100 # Number of episodes in each iteration
PERCENTILE: int = 70 # We will keep the first 70% episodes sorted by reward, those will be our elite

# Number of neurons in each hidden layer (arbitrary)
FIRST_LAYER_SIZE = 128 
SECOND_LAYER_SIZE = 64
THIRD_LAYER_SIZE = 4


class NeuralNetwork(nn.Module): # TODO: docstring
    """
    
    """
    def __init__(self):
        """
        Here we define the operations for the neural network (NN). Our 
        NN has an input layer, two hidden layers and an output layer. A 
        tensor is passed sequentially through operations. 

        We pass to each layer the input size and the output size, in this
        order. Note that the output size of a layer becomes the input size
        of the next layer.
        """
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            # Input layer
            nn.Linear(N_EDGES, FIRST_LAYER_SIZE),
            nn.ReLU(),

            # First hidden layer
            nn.Linear(FIRST_LAYER_SIZE, SECOND_LAYER_SIZE), 
            nn.ReLU(),

            # Second hidden layer
            nn.Linear(SECOND_LAYER_SIZE, THIRD_LAYER_SIZE),
            nn.ReLU(),

            # Output layer
            nn.Linear(THIRD_LAYER_SIZE, N_ACTIONS), 
        )

    def forward(self, x: torch.Tensor):
        """
        This function must exist since we are creating our neural 
        network using Pytorch's nn.Module. 

        It takes a tensor x and passes it through all the operations 
        defined in the __init__ method.
        """
        return self.net(x)


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


def calculate_reward(lambda1: float, mu: int) -> float:
    """
    This function calculates the reward for our reinforcement learning
    problem. The reward depends on the conjecture we are trying to disprove. 

    Since we want to minimize lambda1 + mu (our loss function), we return the
    negative of this. Moreover, given that we want to disprove that 
    lambda1 + mu >= sqrt(N-1) + 1, if we consider our reward function to be 
    sqrt(N-1) + 1 - (lambda1 + mu) then it is enough to check whether the reward is 
    positive. In such a case, we'll have found a counterexample for the conjecture.
    """
    return math.sqrt(N_VERTICES-1) + 1 - (lambda1 + mu)


def restart_environment_and_iterate(agent): # TODO: typing (agent is the neural network)
    """
    Each time this function is called the environment is reset. That
    is, we restart the states and actions for the agent as well as the
    predicted probabilities.
    """
    states =  np.zeros([BATCH_SIZE, N_EDGES], dtype=int) # TODO: typing d'aquests multidimensional arrays
    actions = np.zeros([BATCH_SIZE, N_EDGES], dtype = int)
    next_state = np.zeros([BATCH_SIZE, N_EDGES], dtype = int)
    # prob = np.zeros(BATCH_SIZE) --> I think this is redundant
    total_rewards = np.zeros([BATCH_SIZE])

    states[:, 0]: int = 1
    current_edge: int = 0
    
    while True:
        prob = agent.predict(states[:,current_edge-1], batch_size = BATCH_SIZE) # TODO: typing (and for the rest of the function)

        for episode in range(BATCH_SIZE):
            
            if np.random.rand() < prob[episode]:
                action: int = 1
            else:
                action: int = 0

            actions[episode][current_edge]: int = action

            # We equate next state and current state for the current edge and will adjust next state depending on the action
            next_state[episode] = states[episode, current_edge] 

            if action == 1: # We add that edge to the graph
                next_state[episode][current_edge]: int = action	

            if current_edge + 1 < N_EDGES:
                terminal: bool = False
            else: 
                terminal: bool = True

            if terminal:
                total_rewards[episode] = calculate_reward(next_state[episode])

            if not terminal:
                states[episode, current_edge + 1] = next_state[episode]	

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


def cross_entropy_method(model): # TODO: finish docstring
    """
    This is the main function
    """
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
    cross_entropy_method(model=NeuralNetwork)