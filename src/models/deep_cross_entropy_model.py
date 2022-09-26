"""
Model class for deep cross entropy
"""

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


N_VERTICES = 19
N_EDGES = int(N_VERTICES*(N_VERTICES-1)/2)

# The input vector will have size 2*N_EDGES, where the first N_EDGES letters encode our partial word (with zeros on
#   the positions we haven't considered yet), and the next N_EDGES bits one-hot encode which letter we are considering now.
#   For instance, [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
SPACE = 2*N_EDGES 

LEARNING_RATE = 0.0001 

# Number of neurons in each hidden layer (arbitrary)
FIRST_LAYER_SIZE = 128 
SECOND_LAYER_SIZE = 64
THIRD_LAYER_SIZE = 4


class DeepCrossEntropyModel():    
    def __init__(self):
        self.model = Sequential()

        # We add three linear layers as well as their activation layers and a final output layer
        #   activated by the sigmoid function (so the final result takes values between 0 and 1)
        self.model.add(Dense(FIRST_LAYER_SIZE,  activation='relu'))
        self.model.add(Dense(SECOND_LAYER_SIZE, activation='relu'))
        self.model.add(Dense(THIRD_LAYER_SIZE, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

    def build_and_compile_model(self):
        # We build the model based on input shapes received
        self.model.build((None, SPACE)) 

        self.model.compile(
            loss='binary_crossentropy', # Since we predict a binary outcome (whether the graph has a given edge or not)
            optimizer=Adam(learning_rate=LEARNING_RATE) # Wagner uses SGD as an optimizer
        ) 

        return self.model