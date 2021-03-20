from ray_multiprocessing import A2C_Agent
from policies import Epsilon_Greedy
from training import TrainingManagerTDa2c
from analysis import Experiment_Manager

from datetime import datetime
import ray

NUM_REPETETIONS = 1
EPOCHS = 2_500
DELAY = 7
BATCH_SIZE = 128
LR = 0.000_01
MAX_STEPS = 400

def main():
    for repetition in range(NUM_REPETETIONS):
        ray.init()
        now = datetime.now()
        now = now.strftime('%Y%m%d')
        exp_manager = Experiment_Manager(f'{now}-polAdvantage-MaxSteps{MAX_STEPS}-BatchSize{BATCH_SIZE}_Delay{DELAY}_lr{LR}_L2Reg',
                                         str(repetition))

        training_manager = TrainingManagerTDa2c(agent=A2C_Agent,
                                           batch_size=BATCH_SIZE,
                                           lr=LR,
                                           max_steps=MAX_STEPS)

        for epoch in range(EPOCHS):
            training_manager.timer.start()
            training_manager.start_sampling()

            for i in range(DELAY):
                batch = training_manager.ER_memory.get_random_batch()
                training_manager.update_models(batch)

            training_manager.ER_memory.erase_memory()
            training_manager.update_stats()
            training_manager.update_agents()
            training_manager.update_target_critic_weights()

            elapsed_time_epoch = training_manager.timer.stop()
            print(f'[Ep - {epoch} - rep {repetition}] delta_t: {elapsed_time_epoch:0.2f} sec. - avg_steps: {training_manager.avg_steps_per_epoch[-1]:0.4f} - actor_loss: {training_manager.actor_loss_per_epoch[-1]:0.4f} - critic_loss: {training_manager.critic_loss_per_epoch[-1]:0.4f} - cum_reward: {training_manager.cum_reward_per_epoch[-1]:0.4f}')

            if epoch == EPOCHS-1:
                ray.shutdown()

                exp_manager.plot_steps(training_manager.avg_steps_per_epoch)
                exp_manager.plot_actor_loss(training_manager.actor_loss_per_epoch)
                exp_manager.plot_critic_loss(training_manager.critic_loss_per_epoch)
                exp_manager.plot_cum_sum(training_manager.cum_reward_per_epoch)
                break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exp_manager.plot_steps(training_manager.avg_steps_per_epoch)
        exp_manager.plot_actor_loss(training_manager.actor_loss_per_epoch)
        exp_manager.plot_critic_loss(training_manager.critic_loss_per_epoch)
        exp_manager.plot_cum_sum(training_manager.cum_reward_per_epoch)