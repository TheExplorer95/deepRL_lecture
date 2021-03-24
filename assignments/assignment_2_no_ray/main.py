from DQN_agent import DQN_Agent
from policies import Epsilon_Greedy
from training import DQN_TrainingManager

import numpy as np

BATCH_SIZE = 64
EPOCHS = 1000
UPDATES = 5
DELAY = 5

if __name__ == '__main__':
    agent = DQN_Agent(env_str='CartPole-v0',
                      policy=Epsilon_Greedy,
                      batch_size=BATCH_SIZE)

    training_manager = DQN_TrainingManager(agent=agent,
                                           batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        avg_steps = agent.sample_from_env(avg_steps_last_epoch=training_manager.get_last_avg_steps())
        batch = agent.get_random_batch(BATCH_SIZE)

        # losses = []
        # for update in range(UPDATES):
        loss = training_manager.update_model(batch)
            # losses.append(loss)

        training_manager.update_stats(avg_steps, loss)
        agent.policy.update_epsilon()
        if epoch%DELAY == 0: agent.update_target_model_weights()

        print(f'[Epoch - {epoch}] avg_steps: {avg_steps:0.4f}, loss: {loss:0.4f}')
