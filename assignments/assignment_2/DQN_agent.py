import tensorflow as tf
import gym
from collections import deque
import random
import itertools
import numpy as np
from math import ceil
from models import DQN

class DQN_Agent:
    def __init__(self, env_str, policy, batch_size,  memory_len=100_000, env_seed=0):
        self.init_env(env_str, env_seed)
        self.init_delayed_DQN(batch_size=batch_size)
        self.policy = policy()
        self.memory_len = memory_len
        self.memory = deque(maxlen=memory_len)

    def init_env(self, env_str, env_seed):
        self.env = gym.make(env_str)
        self.env.seed(env_seed)

    def init_delayed_DQN(self, batch_size):
        self.model = DQN(input_dim=(batch_size, self.env.observation_space.shape[0]),
                         output_dim=self.env.action_space.n)
        self.target_model = DQN(input_dim=(batch_size, self.env.observation_space.shape[0]),
                                output_dim=self.env.action_space.n)
        self.update_target_model_weights()

    def sample_from_env(self, avg_steps_last_epoch, min_trajectories=10):
        trajectories = ceil(min((avg_steps_last_epoch*min_trajectories), self.memory_len))

        steps_list = []
        for trajectorie in range(trajectories):
            steps = 0
            state = self.env.reset()
            done = False

            while not done:
                action = self.policy.take_action(state.reshape(1, self.env.observation_space.shape[0]),
                                                 self.model)
                new_state, r, done, _ = self.env.step(action)
                new_state = new_state
                self.remember([state, [action], [r], new_state, [int(done)]])
                state = new_state
                steps += 1

            steps_list.append(steps)

        avg_steps = np.mean(steps_list)

        return avg_steps

    def remember(self, transition):
        self.memory.append(list(itertools.chain(*transition)))

    def set_target_model_weights(self, model):
        self.target_model.set_weights(model.get_weights())

    def update_target_model_weights(self):
        self.target_model.set_weights(self.get_model_weights())

    def get_model_weights(self):
        return self.model.get_weights()

    def set_model_weights(self, model):
        self.model.set_weights(model.get_weights)

    def get_random_batch(self, batch_size):
        # chooses random indexes of the memory buffer and returns them as
        # tf.Tensor (weird way to reduce computations)

        idx_batch = set(random.sample(range(len(self.memory)), batch_size))
        batch = [val for i, val in enumerate(self.memory) if i in idx_batch]
        return tf.constant(batch)
