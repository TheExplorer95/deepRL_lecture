import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.activations import tanh, relu
from tensorflow.keras.regularizers import l1

class DQN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, act_fct=tanh):
        super(DQN, self).__init__()

        self.layer = [
                Dense(64, input_dim=input_dim, kernel_regularizer=l1()),
                Activation(act_fct),
                Dense(32, kernel_regularizer=l1()),
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
