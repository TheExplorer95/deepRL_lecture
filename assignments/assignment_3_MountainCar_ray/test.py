import gym
from time import sleep

env = gym.make('MountainCarContinuous-v0')
print(env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(action)
        sleep(0.2)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
