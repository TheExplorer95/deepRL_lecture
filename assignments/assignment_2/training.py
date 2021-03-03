import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam

class DQN_TrainingManager:
    def __init__(self, agent, batch_size, optimizer=SGD, lr=0.00001, gamma=0.99):
        self.agent = agent
        self.optimizer = optimizer(learning_rate=lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.avg_steps_epoch = []
        self.loss_epoch = []

    def update_stats(self, avg_steps, loss):
        self.avg_steps_epoch.append(avg_steps)
        self.loss_epoch.append(loss)

    def get_last_avg_steps(self):
        try:
            return self.avg_steps_epoch[-1]
        except IndexError:
            return 20

    @tf.function
    def update_model(self, batch):
        suc_state = tf.reshape(batch[:, 6:10], [self.batch_size, -1])
        q_suc_state = self.agent.target_model(suc_state)
        q_max_suc_state = tf.reduce_max(q_suc_state, axis=-1, keepdims=True)
        reward = tf.cast(tf.reshape(batch[:, 5], [self.batch_size, -1]), tf.float32)

        done = tf.subtract(tf.constant(1.0), tf.cast(tf.reshape(batch[:, 10], [self.batch_size, -1]), tf.float32))
        q_target = tf.add(reward, tf.multiply(done, tf.cast(tf.multiply(self.gamma, q_max_suc_state), tf.float32)))

        state = tf.reshape(batch[:, 0:4], [self.batch_size, -1])
        actions = tf.cast(tf.reshape(batch[:, 4], [self.batch_size, -1]), tf.int32)

        with tf.GradientTape() as tape:
            prediction = tf.gather(self.agent.model(state), actions, batch_dims=1)
            loss = tf.reduce_mean(tf.square(prediction - tf.stop_gradient(q_target)))
            loss_reg = tf.add(loss, tf.reduce_sum(self.agent.model.losses))

        gradients = tape.gradient(loss_reg, self.agent.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.agent.model.trainable_variables))

        return loss_reg
