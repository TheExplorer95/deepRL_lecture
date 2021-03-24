import ray
import gym
import itertools
import numpy as np
from models import DQN

@ray.remote
class DQN_Agent_Actor:
    def __init__(self, ID, env_str, policy, batch_size, model_weights, decay_factor=.996, env_seed=0):
        self.init_env(env_str, env_seed)
        self.clone_main_model(DQN, model_weights, batch_size)
        self.policy = policy(decay_factor=decay_factor)
        self.ID = ID
        self.memory = []
        self.avg_steps = []

    def init_env(self, env_str, env_seed):
        self.env = gym.make(env_str)
        self.env.seed(env_seed)

    def clone_main_model(self, model_class, model_weights, batch_size):
        self.model = model_class(input_dim=(batch_size, self.env.observation_space.shape[0]),
                                 output_dim=(self.env.action_space.n))
        self.model.build((batch_size, self.env.observation_space.shape[0]))
        self.model.set_weights(model_weights)

    def sample_from_env(self, trajectories):
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
        self.avg_steps.append(np.mean(steps_list))

        return self.ID

    def remember(self, transition):
        self.memory.append(list(itertools.chain(*transition)))

    def get_memory(self):
        memory = self.memory.copy()
        self.memory.clear()
        return memory

    def get_avg_steps(self):
        avg_steps = self.avg_steps.copy()
        self.avg_steps.clear()
        return avg_steps

    def set_model_weights(self, model_weights):
        self.model.set_weights(model_weights)

    def update_policy(self):
        self.policy.update_epsilon()
