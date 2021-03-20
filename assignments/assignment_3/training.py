import gym
import random
import ray
import time
import tensorflow as tf

from collections import deque
from math import ceil
from tensorflow.keras.optimizers import SGD
from tensorflow import constant
import numpy as np
from numpy import mean
from psutil import cpu_count

from models import DQN, TDadvActor, TDadvCritic

class TrainingManagerTDa2c:
    def __init__(self, agent, sample_factor=10, max_steps=200, cpu_cores=1, batch_size=128,
                 env_str='MountainCarContinuous-v0', optimizer=SGD, lr=0.00001, gamma=0.95):

        if cpu_cores is None:
            self.cpu_cores = cpu_count()
        else:
            self.cpu_cores = cpu_cores

        self.max_steps = max_steps

        self.ER_memory = ER_Memory(batch_size=batch_size)
        self.sample_factor = sample_factor

        self.timer = Timer()
        self.batch_size = batch_size
        self.optimizer = optimizer(learning_rate=lr)
        self.gamma = gamma

        self.get_env_dimensions(env_str, batch_size)
        self.init_Networks(self.model_input_dim, self.model_output_dim)

        self.agents = [agent.remote(ID, env_str, batch_size, TDadvActor, self.actor.get_weights()) for ID in range(self.cpu_cores)]

        # statistics
        self.avg_steps_per_epoch = []
        self.cum_reward_per_epoch = []
        self.critic_loss_per_epoch = []
        self.actor_loss_per_epoch = []

        self.actor_loss_metric = tf.keras.metrics.Mean('actor_loss')
        self.critic_loss_metric = tf.keras.metrics.Mean('critic_loss')


    def get_env_dimensions(self, env_str, batch_size):
        env = gym.make(env_str)
        self.model_input_dim = (batch_size, env.observation_space.shape[0])
        self.model_output_dim = (env.action_space.shape[0])
        self.env_a_low = env.action_space.low[0]
        self.env_a_high = env.action_space.high[0]

    def init_Networks(self, input_dim, output_dim):
        tf.keras.backend.clear_session()

        self.actor = TDadvActor(input_dim, output_dim,
                                act_low=self.env_a_low,
                                act_high=self.env_a_high)
        self.actor(np.zeros(self.batch_size*self.model_input_dim[-1]).reshape(self.model_input_dim))
        self.actor.summary()

        self.critic = TDadvCritic(input_dim, output_dim)
        self.critic.build(input_dim)
        self.critic.summary()

        self.target_critic = TDadvCritic(input_dim, output_dim)
        self.target_critic.build(input_dim)
        self.update_target_critic_weights()

    def update_target_critic_weights(self):
        self.target_critic.set_weights(self.get_critic_weights())

    def get_critic_weights(self):
        return self.critic.get_weights()

    def save_model(self, path, epoch, model_name="model"):
        time_stamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        full_path = f"{path}/{model_name}_{epoch}_{time_stamp}"
        agent = self.get_agent()
        print("saving model...")
        agent.critic.save(full_path)

    def update_agents(self):
        futures = [agent.set_actor_weights.remote(self.actor.get_weights()) for agent in self.agents]
        while futures:
            _, futures = ray.wait(futures)

    def get_prev_avg_steps(self):
        try:
            return self.avg_steps_per_epoch[-1]
        except IndexError:
            return self.max_steps

    def calc_amount_trajectories_per_agent(self):
        trajectories = ceil(self.batch_size * self.sample_factor / self.get_prev_avg_steps() / self.cpu_cores)
        trajectories = max(trajectories, self.cpu_cores * 5)
        return trajectories

    def start_sampling(self):
        amount_trajectories = self.calc_amount_trajectories_per_agent()

        futures = [agent.sample_from_env.remote(amount_trajectories, self.max_steps) for agent in self.agents]
        while futures:
            agent_IDs, futures = ray.wait(futures)
            self.ER_memory.remember(ray.get(self.agents[ray.get(agent_IDs[0])].get_memory.remote()))

    def update_stats(self):
        self.avg_steps_per_epoch.append(mean([ray.get(agent.get_avg_steps.remote()) for agent in self.agents]))
        self.cum_reward_per_epoch.append(mean([ray.get(agent.get_cum_reward.remote()) for agent in self.agents]))
        self.critic_loss_per_epoch.append(self.critic_loss_metric.result())
        self.actor_loss_per_epoch.append(self.actor_loss_metric.result())

    # @tf.function
    def update_models(self, batch):
        # for a state space with dim 4
        state = tf.reshape(batch[:, 0:2], [self.batch_size, -1])
        action = tf.cast(tf.reshape(batch[:, 2], [self.batch_size, -1]), tf.float32)
        reward = tf.cast(tf.reshape(batch[:, 3], [self.batch_size, -1]), tf.float32)
        done = tf.subtract(tf.constant(1.0), tf.cast(tf.reshape(batch[:, 6], [self.batch_size, -1]), tf.float32))
        suc_state = tf.reshape(batch[:, 4:6], [self.batch_size, -1])

        # TD-target
        # target = r + gamma * V(suc_state)
        td_target_critic = tf.add(reward, tf.multiply(done, tf.cast(tf.multiply(self.gamma, self.target_critic(suc_state)), tf.float32)))
        td_target_actor = tf.add(reward, tf.multiply(done, tf.cast(tf.multiply(self.gamma, self.critic(suc_state)), tf.float32)))

        # advantage
        # A = r + gamma* V(suc_state) - V(state)
        advantage = td_target_actor - self.critic(state)

        # breakpoint()
        with tf.GradientTape() as tape:
            loss_actor = tf.reduce_sum(tf.math.subtract(tf.constant(0.0), self.actor(state, prob=True).log_prob(action)) * tf.stop_gradient(advantage))
            loss_act_reg = tf.add(loss_actor, tf.reduce_sum(self.actor.losses))
        gradients = tape.gradient(loss_act_reg, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            loss_critic = tf.reduce_sum(tf.square(tf.math.subtract(self.critic(state), tf.stop_gradient(td_target_critic))))
            loss_crit_reg = tf.add(loss_critic, tf.reduce_sum(self.critic.losses))
        gradients = tape.gradient(loss_crit_reg, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        self.critic_loss_metric.update_state(loss_critic)
        self.actor_loss_metric.update_state(abs(loss_actor))


class ER_Memory:
    def __init__(self, batch_size, memory_len=100_000):
        self.batch_size = batch_size
        self.memory_len = memory_len
        self.memory = deque(maxlen=memory_len)

    def remember(self, samples):
        self.memory.extend(samples)

    def get_random_batch(self):
        # chooses random indexes of the memory buffer and returns them as
        # tf.Tensor (weird way to reduce computations)
        idx_list = random.sample(range(len(self.memory)), self.batch_size)

        idx_batch = set(idx_list)
        batch = [val for i, val in enumerate(self.memory) if i in idx_batch]
        batch = constant(batch)

        idx_list.sort(reverse=True)
        for i in idx_list:
            del self.memory[i]

        return batch

    def erase_memory(self):
        self.memory.clear()


class Timer():
    # small class to measure time during training.
    def __init__(self):
        self._start_time = None

    def start(self):
        # Start a new timer
        if self._start_time is not None:
            print(f"Timer is running. Use .stop() to stop it")
            return None
        self._start_time = time.perf_counter()

    def stop(self):
        # Stop the timer, and report the elapsed time
        if self._start_time is None:
            print(f"Timer is not running. Use .start() to start it")
            return 0
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed_time
