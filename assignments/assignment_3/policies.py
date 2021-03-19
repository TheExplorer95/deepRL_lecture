import numpy as np
import tensorflow as tf

class Epsilon_Greedy:
    def __init__(self, epsilon=1, epsilon_min=0.01, decay_factor=.996):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_factor = decay_factor

    def take_action(self, state, model):
        p = np.random.uniform()
        q_values = model(tf.constant(state)).numpy()
        q_values = np.squeeze(q_values)

        if p <= 1-self.epsilon:
            # choose optimal action, if several optimal actions, action is choosen randomly between them
            return np.random.choice(np.flatnonzero(q_values == q_values.max()))

        else:
            # choose random between actions
            return np.random.choice(np.arange(len(q_values)))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.decay_factor)



# implement thompson sampling
#class Thomson_Sampling
