import ray
import gym
import itertools
import numpy as np
import sklearn
from models import DQN, TDadvActor, TDadvCritic


@ray.remote
class A2C_Agent:
    def __init__(self, ID, env_str, batch_size, actor_class, actor_weights,
                 env_seed=0, expl_factor=1., expl_decay=.998, epsilon=0.9):
        self.init_env(env_str, env_seed)
        self.clone_main_model(actor_class, actor_weights, batch_size)
        self.ID = ID
        self.expl_factor = expl_factor
        self.expl_decay = expl_decay
        self.memory = []
        self.avg_steps = []
        self.cum_reward = []
        self.epsilon = epsilon
        self.counter = 0

    def init_env(self, env_str, env_seed):
        self.env = gym.make(env_str)
        self.env.seed(env_seed)
        self.state_min = self.env.observation_space.low
        self.state_max = self.env.observation_space.high

    def scale_state(self, state):
        min = -1.
        max = 1.
        state[0] = min + (((state[0] - self.state_min[0]) * (max - min)) / (self.state_max[0] - self.state_min[0]))
        state[1] = min + (((state[1] - self.state_min[1]) * (max - min)) / (self.state_max[1] - self.state_min[1]))

        return state

    def clone_main_model(self, actor_class, actor_weights, batch_size):
        self.actor = TDadvActor(input_dim=(batch_size, self.env.observation_space.shape[0]),
                                output_dim=self.env.action_space.shape[0],
                                act_low=self.env.action_space.low[0],
                                act_high=self.env.action_space.high[0])

        self.actor(np.ones(batch_size*self.env.observation_space.shape[0]).reshape((batch_size, self.env.observation_space.shape[0])))
        self.actor.set_weights(actor_weights)

    def sample_from_env(self, trajectories, max_steps):
        steps_list = []
        cum_reward_list = []
        for trajectorie in range(trajectories):
            steps = 0
            cum_reward = 0
            state = self.env.reset()
            done = False

            while not done and steps <= max_steps-1:
                self.env.render()
                action = self.actor(state.reshape(1, self.env.observation_space.shape[0])).numpy()[0]
                action = (1-self.epsilon) * action + self.epsilon * np.random.uniform(low=-self.expl_factor, high=self.expl_factor)
                new_state, r, done, _ = self.env.step(action)
                self.remember([state, action, [r], new_state, [int(done)]])
                state = new_state
                steps += 1
                cum_reward += r

            steps_list.append(steps)
            cum_reward_list.append(cum_reward)
        self.avg_steps.append(np.mean(steps_list))
        self.cum_reward.append(np.mean(cum_reward_list))

        self.update_epsilon()

        return self.ID

    def remember(self, transition):
        self.memory.append(list(itertools.chain(*transition)))

    def get_memory(self):
        memory = self.memory.copy()
        self.memory.clear()
        return memory

    def update_epsilon(self):
        #self.epsilon *= self.expl_decay
        self.counter +=1

        if self.counter > 200:
            self.epsilon =  0.8
        elif self.counter > 300:
            self.epsilon =  0.5
        elif self.counter > 350:
            self.epsilon =  0.3
        elif self.counter > 450:
            self.epsilon =  0.1
        elif self.counter > 500:
            self.epsilon =  0.

    def get_cum_reward(self):
        cum_reward = self.cum_reward.copy()
        self.cum_reward.clear()
        return cum_reward

    def get_avg_steps(self):
        avg_steps = self.avg_steps.copy()
        self.avg_steps.clear()
        return avg_steps

    def set_actor_weights(self, model_weights):
        self.actor.set_weights(model_weights)

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

    def get_avg_reward(self):
        avg_reward = self.avg_reward.copy()
        self.avg_reward.clear()
        return avg_reward

    def set_model_weights(self, model_weights):
        self.model.set_weights(model_weights)

    def update_policy(self):
        self.policy.update_epsilon()
