"""
Deep Cross-Entropy method

##########
Pseudocode: 
##########
Initialize neural network;
while the best construction found is not a counterexample do
	for i<-1 to N do
		w <- empty string
			while not terminal do
				Input w into the neural net to get a probability distribution F on the
				next letter;
				Sample next letter x according to F;
				w <- w + x
			end
		end
		Evaluate the score of each construction;
		Sort the constructions according to their score;
		Throw away all but the top y percentage of the constructions
		for all remaining constructions do 
			for all (observation, issued action) pairs in the construction do 
				Adjust the weights of the neural net slightly to minimize the cross-entropy 
				loss between issued action and the corresponding predcited action probability;
			end
		end
	Keep the top x percentage of constructions for the next iteration, 
	throw away the rest;
end
"""
import logging # TODO: set logging
from typing import List

from collections import namedtuple
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim


# This values are arbitrary and aren't tuned, as the cross-entropy method is very robust and converges very quickly.

HIDDEN_SIZE: int = 128 # Number of hidden neurons
BATCH_SIZE: int = 16 # Number of episodes we play in each iteration
PERCENTILE: int = 70 # We will keep the first 70% episodes sorted by reward (those will be our elite)


class NeuralNetwork(nn.Module):
    """
    This neural network (NN) takes a single observation from the environment
    as an input vector and outputs a number for every action we can perform.

    The output of from the NN is a probability distribution over actions.
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        ) # TODO: the neural net should have three layers as in Wagner's code

    def forward(self, x):
        return self.net(x)


EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action']) # We'll use episode steps from elite episodes as training data
Episode = namedtuple('Episode', field_names=['reward', 'steps']) # steps: collection of EpisodeStep


def iterate_batches(environment, neural_network, batch_size: int):
    """
    This function takes as arguments the environment the agent will be in, 
    our neural network and the number of episodes in each iteration.

    This function allows the training of our NN and the generation of our 
    episodes to be performed at the same time. Each time we have reached
    the maximum amount of episodes for one single iteration, the NN will 
    be trained using gradient descent. 
    """
    batch: List[Episode] = []
    episode_reward: float = 0
    episode_steps: List[EpisodeStep] = []

    # We reset our environment to obtain the first observation
    obs = environment.reset() 

    # This Softmax layer will be used to convert the neural network's
    #   output to a probability distribution of actions
    softmax = nn.Softmax(dim=1) 

    while True: 
        # At every iteration we convert our current observation to a Pytorch tensor 
        #   and pass it to the neural network to obtain action probabilities
        obs_ = torch.FloatTensor([obs]) # We want a tensor of size 1xn (this is why we pass [obs] instead of obs)
        raw_action_scores: torch.Tensor = neural_network(obs_) 
        action_probs: torch.Tensor = softmax(raw_action_scores)

        # We need to unpack action probabilities. Moreover, action_probs.data.numpy() is
        #   2-dimensional as the input and the batch dimension is on axis 0 of the array
        action_probs_array: np.array = action_probs.data.numpy()[0] 

        action = np.random.choice(len(action_probs_array), p=action_probs_array)

        # By passing an action to the environment we get the next observation, 
        #   the reward and whether the episode has finished
        next_obs, reward, episode_finished, _ = environment.step(action) # TODO: what is _ ????? Add typing

        episode_reward += reward # The reward is added to the current episode's total reward

        # IMPORTANT NOTE: we save the observation that was used to choose the action, not the observation
        #   returned by the environment as a result of the action (next_obs)
        step = EpisodeStep(observation=obs, action=action) 
        episode_steps.append(step) # We add current observation and action to our list of episode steps

        if episode_finished:
            episode = Episode(reward=episode_reward, steps=episode_steps) # We save this episode's data
            batch.append(episode)
            episode_reward = 0.0 # We reset the episode's reward for the next iteration
            episode_steps = [] # We reset the episode's steps for the next iteration
            next_obs = environment.reset() # We reset our environment to start over

            # Once we've reached the desired batch size, we return (yield) the batch and reset it
            if len(batch) == batch_size:
                yield batch # After this, the NN is retrained and will have a different (hopefully better) behaviour
                batch = []

        # The next observation variable is the observation obtained by the environment
        obs = next_obs 


def filter_batch(batch, percentile):
    """
    This function calculates a boundary reward given a batch of
    episodes and a percentile value. The boundary reward is used
    to filter the "elite" episodes that will be later used for training
    """
    rewards = list(map(lambda s: s.reward, batch)) # TODO: read about lambda functions in python
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards)) # This is just for monitoring purposes

    # We want to keep the observations and the actions belonging to the elite episodes
    elite_obs = []
    elite_actions = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        elite_obs.extend(map(lambda step: step.observation, steps))
        elite_actions.extend(map(lambda step: step.action, steps))

    # We convert our "elite" observations and actions into tensor objects
    elite_obs_: torch.Tensor = torch.FloatTensor(elite_obs)
    elite_actions_: torch.Tensor = torch.LongTensor(elite_actions)

    return elite_obs_, elite_actions_, reward_bound, reward_mean


def cross_entropy_method_training(): # TODO: redefine this
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = NeuralNetwork(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss() # TODO: I guess que aquí hauré de posar la meva funció de loss de la conjectura
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    # writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()