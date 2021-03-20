import gym
import random
import ray
import time
import tensorflow as tf

from collections import deque
from math import ceil
from tensorflow.keras.optimizers import SGD
from tensorflow import constant
from numpy import mean
from psutil import cpu_count

from models import DQN

class TrainingManager:
    def __init__(self, agent, policy, samples_per_epoch=1_000, cpu_cores=None, batch_size=128,
                 env_str='CartPole-v0', optimizer=SGD, lr=0.00001, gamma=0.99, decay_factor=.996):

        if cpu_cores is None:
            self.cpu_cores = cpu_count()

        self.timer = Timer()

        self.batch_size = batch_size

        self.optimizer = optimizer(learning_rate=lr)
        self.gamma = gamma

        self.init_model_dimensions(env_str, batch_size)
        self.init_delayed_DQN(self.model_input_dim, self.model_output_dim)

        self.ER_memory = ER_Memory(batch_size, memory_len=10_000)
        self.agents = [agent.remote(ID, env_str, policy, batch_size, self.model.get_weights()) for ID in range(self.cpu_cores)]
        self.samples_per_epoch = samples_per_epoch

        # statistics
        self.avg_steps_per_epoch = []
        self.loss_per_epoch = []

        self.loss_metric = tf.keras.metrics.Mean('train_loss')

    def init_model_dimensions(self, env_str, batch_size):
        env = gym.make(env_str)
        self.model_input_dim = (batch_size, env.observation_space.shape[0])
        self.model_output_dim = (env.action_space.n)

    def init_delayed_DQN(self, input_dim, output_dim):
        tf.keras.backend.clear_session()
        self.model = DQN(input_dim, output_dim)
        self.model.build(input_dim)
        self.model.summary()

        self.target_model = DQN(input_dim, output_dim)
        self.target_model.build(input_dim)
        self.target_model.summary()
        self.update_target_model_weights()

    def update_target_model_weights(self):
        self.target_model.set_weights(self.get_model_weights())

    def get_model_weights(self):
        return self.model.get_weights()

    def save_model(self, path, epoch, model_name="model"):
        time_stamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        full_path = f"{path}/{model_name}_{epoch}_{time_stamp}"
        agent = self.get_agent()
        print("saving model...")
        agent.model.save(full_path)

    def update_agents(self):
        futures = [agent.set_model_weights.remote(self.model.get_weights()) for agent in self.agents]
        futures.extend([agent.update_policy.remote() for agent in self.agents])
        while futures:
            _, futures = ray.wait(futures)

    def get_prev_avg_steps(self):
        try:
            return self.avg_steps_per_epoch[-1]
        except IndexError:
            return 20

    def calc_amount_trajectories_per_agent(self):
        amount_trajectories = min(ceil(self.samples_per_epoch/self.get_prev_avg_steps()), self.ER_memory.memory_len)
        return ceil(amount_trajectories/len(self.agents))

    def start_sampling(self):
        amount_trajectories = self.calc_amount_trajectories_per_agent()

        futures = [agent.sample_from_env.remote(amount_trajectories) for agent in self.agents]
        while futures:
            agent_IDs, futures = ray.wait(futures)
            self.ER_memory.remember(ray.get(self.agents[ray.get(agent_IDs[0])].get_memory.remote()))

    def update_stats(self):
        self.avg_steps_per_epoch.append(mean([ray.get(agent.get_avg_steps.remote()) for agent in self.agents]))
        self.loss_per_epoch.append(self.loss_metric.result())

    @tf.function
    def update_model(self, batch):
        # for a state space with dim 4
        suc_state = tf.reshape(batch[:, 6:10], [self.batch_size, -1])
        q_suc_state = self.target_model(suc_state)
        q_max_suc_state = tf.reduce_max(q_suc_state, axis=-1, keepdims=True)

        reward = tf.cast(tf.reshape(batch[:, 5], [self.batch_size, -1]), tf.float32)
        done = tf.subtract(tf.constant(1.0), tf.cast(tf.reshape(batch[:, 10], [self.batch_size, -1]), tf.float32))

        q_target = tf.add(reward, tf.multiply(done, tf.cast(tf.multiply(self.gamma, q_max_suc_state), tf.float32)))

        state = tf.reshape(batch[:, 0:4], [self.batch_size, -1])
        action = tf.cast(tf.reshape(batch[:, 4], [self.batch_size, -1]), tf.int32)

        with tf.GradientTape() as tape:
            prediction = tf.gather(self.model(state), action, batch_dims=1)
            loss = tf.reduce_mean(tf.square(prediction - tf.stop_gradient(q_target)))
            loss_reg = tf.add(loss, tf.reduce_sum(self.model.losses))

        gradients = tape.gradient(loss_reg, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss_metric.update_state(loss)

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

        idx_batch = set(random.sample(range(len(self.memory)), self.batch_size))
        batch = [val for i, val in enumerate(self.memory) if i in idx_batch]
        return constant(batch)


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
