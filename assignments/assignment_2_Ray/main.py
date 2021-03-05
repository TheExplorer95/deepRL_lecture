from ray_multiprocessing import DQN_Agent_Actor
from policies import Epsilon_Greedy
from training import TrainingManager

import ray; ray.init()

EPOCHS = 1000
DELAY = 5

if __name__ == '__main__':

    done = False

    while not done:
        training_manager = TrainingManager(agent=DQN_Agent_Actor,
                                           policy=Epsilon_Greedy)

        for epoch in range(EPOCHS):
            training_manager.timer.start()
            training_manager.start_sampling()

            batch = training_manager.ER_memory.get_random_batch()
            for i in range(DELAY):
                training_manager.update_model(batch)

            training_manager.update_stats()

            training_manager.update_agents()

            # if epoch%DELAY == 0: training_manager.update_target_model_weights()
            training_manager.update_target_model_weights()

            elapsed_time_epoch = training_manager.timer.stop()
            print(f'[Epoch - {epoch}] delta_t: {elapsed_time_epoch:0.4f} sec. - avg_steps: {training_manager.avg_steps_per_epoch[-1]:0.4f} - loss: {training_manager.loss_per_epoch[-1]:0.4f}')

            if epoch == EPOCHS-1:
                ray.shutdown()
                done = True

            if training_manager.avg_steps_per_epoch[-1] < 18:
                print(f'[INFO] - Starting over, avg_steps too low {training_manager.avg_steps_per_epoch[-1]:0.4f}')
                break
