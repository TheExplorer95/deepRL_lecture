import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym
import ray
from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets


class ModelContinuous(tf.keras.Model):
    def __init__(self, output_units=2):
        super(ModelContinuous, self).__init__()

        self.layer_mu1 = tf.keras.layers.Dense(256,activation="relu")
        self.layer_mu2 = tf.keras.layers.Dense(256, activation="relu")
        self.output_mu = tf.keras.layers.Dense(output_units)

        #self.layer_sigma1 = tf.keras.layers.Dense(256,activation="relu")
        #self.layer_sigma2 = tf.keras.layers.Dense(256,activation="relu")

        self.output_sigma = tf.keras.layers.Dense(output_units, activation=None)
        self.gamma = 0.99
        #self.layer_v = tf.keras.layers.Dense(1)

    def call(self, x_in):

        output = {}
        mus = self.layer_mu1(x_in)
        mus = self.layer_mu2(mus)
        mus = self.output_mu(mus)

        #sigmas = self.layer_sigma1(x_in)
        #sigmas = self.layer_sigma2(sigmas)
        sigmas = self.output_sigma(x_in)
        sigmas = tf.exp(sigmas)
        sigmas = tf.clip_by_value(sigmas, 0.1, 1)
        output["mu"] = mus
        output["sigma"] = sigmas

        return output

if __name__ == "__main__":


    kwargs = {
        "model": ModelContinuous,
        "environment": "LunarLanderContinuous-v2",
        "num_parallel": 1,
        "total_steps": 30,
        "num_episodes": 1,
        #"gamma": 0.99,
        "action_sampling_type": "continuous_normal_diagonal",
        "returns": ["monte_carlo"] # reward to go
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test"

    buffer_size = 1
    test_steps = 5000
    epochs = 20
    sample_size = 1000
    optim_batch_size = 1
    saving_after = 5

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    manager.test(test_steps, do_print=True)

    # get initial agent
    agent = manager.get_agent()

    optimizer = tf.keras.optimizers.Adam(1e-3)

    for e in range(epochs):
        print("epoch:", e)
        # training core
        # experience replay
        print("collecting experience..")
        data = manager.get_data(total_steps = 10)
        print(data.keys())
        #manager.store_in_buffer(data)
        # sample data to optimize on from buffer
        #sample_dict = manager.sample(sample_size)
        print(f"collected data for: {data.keys()}")
        # create and batch tf datasets
        data = dict_to_dict_of_datasets(data, batch_size=1)
        print("optimizing...")
        t = 0
        for action, state, reward, next_state, not_done, mc in zip(*[ data.get(key) for key in data.keys() ]):

            action = tf.cast(action, tf.float32)
            state = tf.cast(state, tf.float32)
            mc = tf.cast(mc, tf.float32)

            # add batch dimension if there is none in the dictionary, make tf.constant if it's numpy, adjust dtype (tf.cast) etc.
            with tf.GradientTape() as tape:
            # for state s and action a calculate the gradient of "log_pi_a_s"

                policy_output = agent.model(state)
                dist = tfp.distributions.Normal(policy_output["mu"], policy_output["sigma"])
                log_pi_a_s = - tf.reduce_sum( dist.log_prob(action) )

            gradients = tape.gradient(log_pi_a_s, agent.model.trainable_variables)


            #gradients = np.array(gradients)
            gradients = [g*mc[0] * agent.model.gamma**t for g in gradients] # scale gradients by mc values
            #policy_gradients = gradients * tf.squeeze(mc) * agent.model.gamma**t
            optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))
            t+=1

        new_weights = agent.get_weights()
        # # set new weights
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)


    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=15, render=True)
