import tensorflow as tf
from tensorflow.keras.optimizers import SGD

class DQN_TrainingManager:
    def __init__(self, model, batch_size, optimizer=SGD, lr=0.0001, gamma=0.99):
        self.model = model
        self.optimizer = optimizer(learning_rate=lr)
        self.batch_size = batch_size
        self.gamma = 0.99

    def update_model(self, agent, batch):
        suc_state = tf.reshape(batch[:, 6:], [self.batch_size, -1])
        q_suc_state = self.model(suc_state)
        q_max_suc_state = tf.reduce_max(q_suc_state, axis=-1, keepdims=True)
        reward = tf.cast(tf.reshape(batch[:, 5], [self.batch_size, -1]), tf.float32)

        q_target = tf.add(reward, tf.cast(tf.multiply(self.gamma, q_max_suc_state), tf.float32))

        state = tf.reshape(batch[:, 0:4], [self.batch_size, -1])
        actions = tf.cast(tf.reshape(batch[:, 4], [self.batch_size, -1]), tf.int32)

        with tf.GradientTape() as tape:
            prediction = tf.gather(self.model(state), actions, batch_dims=1)
            loss = tf.reduce_mean(tf.square(prediction - tf.stop_gradient(q_target)))
            loss_reg = loss + tf.reduce_sum(self.model.losses)

        gradients = tape.gradient(loss_reg, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss_reg
