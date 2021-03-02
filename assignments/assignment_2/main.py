import gym
import numpy as np

from models import DQN
from DQN_agent import DQN_Agent
from policies import Epsilon_Greedy
from training import DQN_TrainingManager

RANDOM_SEED = 0
BATCH_SIZE = 64
EPOCHS = 100_000
UPDATES = 20

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    DQN_model = DQN(input_dim=(BATCH_SIZE, env.observation_space.shape[0]),
                    output_dim=env.action_space.n)



    agent = DQN_Agent(env=env,
                      policy=Epsilon_Greedy(DQN_model),
                      batch_size=BATCH_SIZE)

    training_manager = DQN_TrainingManager(model=DQN_model,
                                          batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        avg_steps = agent.sample_from_env()
        losses = []
        for update in range(UPDATES):
            batch = agent.get_random_batch(BATCH_SIZE)
            loss = training_manager.update_model(agent, batch)
            losses.append(loss)

        avg_loss = np.mean(losses)
        print(f'[INFO] - Epoch: {epoch}, avg. steps: {avg_steps}, avg. loss: {avg_loss}')
