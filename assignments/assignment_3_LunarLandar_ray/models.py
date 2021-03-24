import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.activations import tanh, relu, softmax, elu
from tensorflow.keras.regularizers import l1, l2

class TDadvActor(tf.keras.Model):
    def __init__(self, input_dim, output_dim, act_low, act_high, act_fct=tanh, out_fct=tanh, reg=l2):
        super(TDadvActor, self).__init__()
        self.act_low = act_low
        self.act_high = act_high

        self.layer_shared = [
                Dense(64, input_dim=input_dim, kernel_regularizer=reg()),
                BatchNormalization(),
                Activation(act_fct),
                Dense(32, kernel_regularizer=reg()),
                BatchNormalization(),
                Activation(act_fct),
                Dense(16, kernel_regularizer=reg()),
                BatchNormalization(),
                Activation(act_fct),
                Dense(8, kernel_regularizer=reg()),
                BatchNormalization(),
                Activation(act_fct),
        ]

        self.mu = Dense(output_dim, kernel_regularizer=reg(), use_bias=False)

        self.sigma = Dense(output_dim, kernel_regularizer=reg(), use_bias=False)


    # @tf.function
    def call(self, x, training=False, prob=False):
        for l in self.layer_shared:
            try:
                x = l(x, training)
            except:
                x = l(x)

        mu = self.mu(x)

        sigma = self.sigma(x)
        sigma = tf.exp(sigma)
        sigma = tf.clip_by_value(sigma, 0.1, 1)

        if prob:
            norm_dist = tfp.distributions.Normal(mu, sigma)

            return norm_dist

        else:
            norm_dist = tfp.distributions.Normal(mu, sigma)
            action = tf.squeeze(norm_dist.sample(1), axis=0)
            action = tf.clip_by_value(action, self.act_low, self.act_high)

            return action

class TDadvCritic(tf.keras.Model):
    def __init__(self, input_dim, output_dim, act_fct=tanh, reg=l2):
        super(TDadvCritic, self).__init__()

        self.layer = [
                Dense(64, input_dim=input_dim, kernel_regularizer=reg()),
                BatchNormalization(),
                Activation(act_fct),
                Dense(32, kernel_regularizer=reg()),
                BatchNormalization(),
                Activation(act_fct),
                Dense(16, kernel_regularizer=reg()),
                BatchNormalization(),
                Activation(act_fct),
                Dense(8, kernel_regularizer=reg()),
                BatchNormalization(),
                Activation(act_fct),
                Dense(output_dim, use_bias=False)
        ]

    @tf.function
    def call(self, x, training=False):
        for l in self.layer:
            # training argument for BN and DropOut
            try:
                x = l(x, training)
            except:
                x = l(x)
        return x


class DQN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, act_fct=tanh, reg=l2):
        super(DQN, self).__init__()

        self.layer = [
                Dense(128, input_dim=input_dim, kernel_regularizer=reg()),
                BatchNormalization(),
                Activation(act_fct),
                Dense(64, kernel_regularizer=reg()),
                BatchNormalization(),
                Activation(act_fct),
                Dense(output_dim, use_bias=False)
        ]

    @tf.function
    def call(self, x, training=False):
        for l in self.layer:
            # training argument for BN and DropOut
            try:
                x = l(x, training)
            except:
                x = l(x)
        return x
