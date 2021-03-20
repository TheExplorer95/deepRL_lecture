import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.activations import tanh, relu
from tensorflow.keras.regularizers import l1, l2

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
