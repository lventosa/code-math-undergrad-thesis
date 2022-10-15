"""
Model class for deep cross entropy
"""

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD, Adam

from src.rl_environments.env_wagner import SPACE


LEARNING_RATE = 0.0001 

# Number of neurons in each hidden layer (arbitrary)
FIRST_LAYER_SIZE = 128 
SECOND_LAYER_SIZE = 64
THIRD_LAYER_SIZE = 4


class DeepCrossEntropyModel():    
    def __init__(self):
        self.model = Sequential()

        # We add three linear layers as well as their activation layers and a final output layer
        #   activated by the sigmoid function (so the final result takes values between 0 and 1).
        self.model.add(Dense(FIRST_LAYER_SIZE,  activation='relu'))
        self.model.add(Dense(SECOND_LAYER_SIZE, activation='relu'))
        self.model.add(Dense(THIRD_LAYER_SIZE, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

    def build_and_compile_model(self, method: str):
        # We build the model based on input shapes received
        self.model.build((None, SPACE)) 
        self.model.compile(
            loss='binary_crossentropy', # Since we predict a binary outcome 
            optimizer=SGD(learning_rate=LEARNING_RATE), 
        ) 
        return self.model
        