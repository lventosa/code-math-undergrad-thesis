"""
In this script we build the model for the deep cross entropy method.
"""

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

from src.rl_environments.environments import SPACE


LEARNING_RATE = 0.0001 

# Number of neurons in each hidden layer (arbitrary)
FIRST_LAYER_SIZE = 128 
SECOND_LAYER_SIZE = 64
THIRD_LAYER_SIZE = 4


model = Sequential() # We instantiate a model of class Sequential()

# We add three linear layers as well as their activation layers and a final output layer
#   activated by the sigmoid function (so the final result takes values between 0 and 1).
model.add(Dense(FIRST_LAYER_SIZE,  activation="relu"))
model.add(Dense(SECOND_LAYER_SIZE, activation="relu"))
model.add(Dense(THIRD_LAYER_SIZE, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.build((None, SPACE)) # We build the model based on input shapes received

model.compile(
    loss="binary_crossentropy", # Since we predict a binary outcome 
    optimizer=SGD(learning_rate = LEARNING_RATE), 
) 
        