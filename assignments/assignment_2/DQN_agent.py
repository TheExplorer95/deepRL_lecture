import tensorflow as tf
from collections import deque
import random
import itertools
import numpy as np
from models import DQN

class DQN_Agent:
    def __init__(self, env, policy, batch_size, buffer_len=100_000):
        self.env = env
        self.policy = policy()

        self.model = DQN(input_dim=(batch_size, env.observation_space.shape[0]),
                         output_dim=env.action_space.n)
        self.target_model = DQN(input_dim=(batch_size, env.observation_space.shape[0]),
                                output_dim=env.action_space.n)
        self.target_model.set_weights(self.model.get_weights())

        self.avg_steps_episode = []
        self.memory = deque(maxlen=buffer_len)

    def sample_from_env(self, episodes=100):
        steps_list = []
        for episode in range(episodes):
            steps = 0
            state = self.env.reset()
            done = False
            #env.render()

            while not done:  # cap timesteps
                action = self.policy.take_action(state.reshape(1, self.env.observation_space.shape[0]),
                                                 self.model)
                new_state, r, done, _ = self.env.step(action)
                new_state = new_state
                self.remember([state, [action], [r], new_state, [int(done)]])
                state = new_state
                #env.render()
                steps += 1

            steps_list.append(steps)

        avg_steps = np.mean(steps_list)
        self.policy.update_epsilon()
        return avg_steps

    def remember(self, transition):
        self.memory.append(list(itertools.chain(*transition)))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_random_batch(self, batch_size):
        # chooses random indexes of the memory buffer and returns them as
        # tf.Tensor (weird way to reduce computations)

        idx_batch = set(random.sample(range(len(self.memory)), batch_size))
        batch = [val for i, val in enumerate(self.memory) if i in idx_batch]
        return tf.constant(batch)
