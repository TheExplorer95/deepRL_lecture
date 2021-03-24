from ray_multiprocessing import A2C_Agent
from training import TrainingManagerTDa2c
from analysis import Experiment_Manager
import numpy as np
from datetime import datetime
import ray
import tensorflow as tf

def configure_gpu_options():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

NUM_REPETETIONS = 5
EPOCHS = 600
DELAY = 6
BATCH_SIZE = 128
LR = 0.000_0001
MAX_STEPS = 350
EXPL_FACTOR = 0.1

if __name__ == '__main__':
    for repetition in range(NUM_REPETETIONS):
        tf.keras.backend.clear_session()
        configure_gpu_options()
        now = datetime.now()
        now = now.strftime('%Y%m%d-%H%M%S')
        result_folder = f'{now}-polAdvantage-MaxStep{MAX_STEPS}-ExplRange{EXPL_FACTOR}-BatchSize{BATCH_SIZE}_Delay{DELAY}_lr{LR}_L2reg'
        exp_manager = Experiment_Manager(result_folder,
                                         str(repetition))
        flag = False
        counter = 0
        while not flag:
            counter +=1
            ray.init()

            training_manager = TrainingManagerTDa2c(agent=A2C_Agent,
                                               batch_size=BATCH_SIZE,
                                               lr=LR,
                                               max_steps=MAX_STEPS,
                                               expl_factor=EXPL_FACTOR,
                                               update_delay=DELAY,
                                               result_folder=result_folder)

            for epoch in range(EPOCHS):
                training_manager.timer.start()
                training_manager.start_sampling()

                update = 0
                for i in range(DELAY):
                    try:
                        batch = training_manager.ER_memory.get_random_batch()
                        training_manager.update_models(batch)
                    except:
                        print(f'[INFO] - skipped update {update}')
                        training_manager.ER_memory.print_len()
                    update += 1

                training_manager.ER_memory.erase_memory()
                training_manager.update_stats()
                training_manager.update_agents()
                training_manager.update_target_critic_weights()

                elapsed_time_epoch = training_manager.timer.stop()
                print(f'[Ep - {epoch} - rep {repetition} - start {counter}] delta_t: {elapsed_time_epoch:0.2f} sec. - avg_steps: {training_manager.avg_steps_per_epoch[-1]:0.4f} - actor_loss: {training_manager.actor_loss_per_epoch[-1]:0.4f} - critic_loss: {training_manager.critic_loss_per_epoch[-1]:0.4f} - cum_reward: {training_manager.cum_reward_per_epoch[-1]:0.4f}')

                if (epoch >= 50 and tf.math.is_nan(training_manager.actor_loss_per_epoch[-1])) or epoch == EPOCHS-1:
                    ray.shutdown()
                    exp_manager.plot_steps(training_manager.avg_steps_per_epoch)
                    exp_manager.plot_actor_loss(training_manager.actor_loss_per_epoch)
                    exp_manager.plot_critic_loss(training_manager.critic_loss_per_epoch)
                    exp_manager.plot_cum_sum(training_manager.cum_reward_per_epoch)
                    flag = True
                    break
                elif tf.math.is_nan(training_manager.actor_loss_per_epoch[-1]) or training_manager.actor_loss_per_epoch[-1] > 2_500:
                    ray.shutdown()
                    flag = False
                    break
                # checkpoints
                elif epoch%50 == 0 and not epoch == 0:
                    training_manager.render()
                    exp_manager.plot_steps(training_manager.avg_steps_per_epoch)
                    exp_manager.plot_actor_loss(training_manager.actor_loss_per_epoch)
                    exp_manager.plot_critic_loss(training_manager.critic_loss_per_epoch)
                    exp_manager.plot_cum_sum(training_manager.cum_reward_per_epoch)
                    training_manager.save_models(str(repetition), result_folder)
