#from really.sample_manager import SampleManager

import gym
import gridworlds

if __name__ == '__main__':
    env = gym.make('gridworld-v0')

    state = env.reset()

    max_timesteps = 1000

    for t in range(max_timesteps):
        env.render()
        print(state)
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)

        if done:
            print(f'Finished episode after {t}-timesteps.')
            break

    env.close()
