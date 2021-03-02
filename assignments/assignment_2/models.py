import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.regularizers import l1


class DQN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, activation=ReLU):
        super(DQN, self).__init__()

        self.layer = [
                Dense(256, input_dim=input_dim, kernel_regularizer=l1()),
                activation(),
                Dense(128, kernel_regularizer=l1()),
                activation(),
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
