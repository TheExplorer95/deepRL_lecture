from ray_multiprocessing import DQN_Agent_Actor
from policies import Epsilon_Greedy
from training import TrainingManager
from analysis import Experiment_Manager

import ray

NUM_REPETETIONS = 5
EPOCHS = 500
DELAY = 8
BATCH_SIZE = 128
SAMPLES_PER_EPOCH = 3_000
LR = 0.000_01
DECAY_FACTOR = .99

if __name__ == '__main__':
    for repetition in range(NUM_REPETETIONS):
        ray.init()
        done = False
        exp_manager = Experiment_Manager(f'BatchSize{BATCH_SIZE}_Delay{DELAY}_Samples{SAMPLES_PER_EPOCH}_lr{LR}_EpsDecay{DECAY_FACTOR}_L2reg',
                                         str(repetition))

        while not done:
            training_manager = TrainingManager(agent=DQN_Agent_Actor,
                                               policy=Epsilon_Greedy,
                                               samples_per_epoch=SAMPLES_PER_EPOCH,
                                               batch_size=BATCH_SIZE,
                                               lr=LR,
                                               decay_factor=DECAY_FACTOR)

            for epoch in range(EPOCHS):
                training_manager.timer.start()
                training_manager.start_sampling()

                for i in range(DELAY):
                    batch = training_manager.ER_memory.get_random_batch()
                    training_manager.update_model(batch)

                training_manager.update_stats()
                training_manager.update_agents()
                training_manager.update_target_model_weights()

                elapsed_time_epoch = training_manager.timer.stop()
                print(f'[Ep - {epoch} - rep {repetition}] delta_t: {elapsed_time_epoch:0.4f} sec. - avg_steps: {training_manager.avg_steps_per_epoch[-1]:0.4f} - loss: {training_manager.loss_per_epoch[-1]:0.4f}')

                if epoch == EPOCHS-1:
                    ray.shutdown()
                    exp_manager.plot_steps(training_manager.avg_steps_per_epoch)
                    exp_manager.plot_loss(training_manager.loss_per_epoch)
                    break

                if training_manager.avg_steps_per_epoch[-1] < 18:
                    print(f'[INFO] - Starting over, avg_steps too low {training_manager.avg_steps_per_epoch[-1]:0.4f}')
                    break
