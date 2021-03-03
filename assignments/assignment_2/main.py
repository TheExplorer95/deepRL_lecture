import gym
import numpy as np

from models import DQN
from DQN_agent import DQN_Agent
from policies import Epsilon_Greedy
from training import DQN_TrainingManager

RANDOM_SEED = 0
BATCH_SIZE = 64
EPOCHS = 1000
UPDATES = 5
DELAY = 5

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    agent = DQN_Agent(env=env,
                      policy=Epsilon_Greedy,
                      batch_size=BATCH_SIZE)

    training_manager = DQN_TrainingManager(agent=agent,
                                           batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        avg_steps = agent.sample_from_env()
        losses = []
        for update in range(UPDATES):
            batch = agent.get_random_batch(BATCH_SIZE)
            loss = training_manager.update_model(batch)
            losses.append(loss)

        if epoch%DELAY == 0: agent.update_target_model()

        avg_loss = np.mean(losses)
        print(f'[INFO] - Epoch: {epoch}, avg. steps: {avg_steps}, avg. loss: {avg_loss}')
