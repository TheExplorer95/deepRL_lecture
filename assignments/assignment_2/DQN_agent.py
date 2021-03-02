import tensorflow as tf
from collections import deque
import random
import itertools
import numpy as np

class DQN_Agent:
    def __init__(self, env, policy, batch_size, buffer_len=100_000):
        self.env = env
        self.policy = policy
        self.model = model
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
                action = self.policy.take_action(state.reshape(1, self.env.observation_space.shape[0]))
                new_state, r, done, _ = self.env.step(action)
                new_state = new_state
                self.remember([state, [action], [r], new_state])
                state = new_state
                #env.render()
                steps += 1

            steps_list.append(steps)

        avg_steps = np.mean(steps_list)
        self.policy.update_epsilon()
        return avg_steps

    def remember(self, step):
        self.memory.append(list(itertools.chain(*step)))

    def get_random_batch(self, batch_size):
        # chooses random indexes of the memory buffer and returns them as
        # tf.Tensor

        idx_batch = set(random.sample(range(len(self.memory)), batch_size))
        batch = [val for i, val in enumerate(self.memory) if i in idx_batch]
        return tf.constant(batch)
