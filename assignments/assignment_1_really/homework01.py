import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld
import os

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):
    def __init__(self, h, w, action_space):
        self.action_space = action_space

        self.q_values = np.zeros((h,w,action_space))


    def __call__(self, state):
        x,y = np.squeeze(state)
        x = int(x)
        y = int(y)

        output = {}
        qvals_at_xy = self.q_values[x,y,:]
        output["q_values"] = np.expand_dims(qvals_at_xy, axis=0)


        # set state
        return output

    # # TODO:
    def get_weights(self):
    #
        return self.q_values

    def set_weights(self, q_vals):

        self.q_values = q_vals


    # what else do you need?


if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 3,
        "width": 4,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 4,
        "total_steps": 100,
        "model_kwargs": model_kwargs
        # and more
    }

    # initilize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)


    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test"

    buffer_size = 5000
    test_steps = 1000
    epochs = 2
    sample_size = 100
    optim_batch_size = 8
    saving_after = 5

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    manager.test(test_steps, do_print=True)

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        # training core

        # experience replay
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        #print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        #data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        print("optimizing...")


        # get all


        # define gamma and alpha
        gamma = 0.9
        alpha=1

        for state, action, reward, state_new, not_done in zip(sample_dict["state"], sample_dict["action"], sample_dict["reward"], sample_dict["state_new"], sample_dict["not_done"]):

            #get current q value
            x,y = np.squeeze(state)

            qtable = agent.get_weights()
            qval = qtable[int(x),int(y), int(action)]

            xnew, ynew = np.squeeze(state_new)

            best_nextstate_qval = np.max(qtable[int(xnew),int(ynew),:])

            td_error = reward + gamma*best_nextstate_qval - qval

            new_qval = qval + alpha * td_error

            # apply new_qval to agents q-table for taken action at state
            qtable_new = np.copy(qtable)
            qtable_new[int(x),int(y),int(action)] = new_qval
            agent.model.set_weights(qtable_new)


        dummy_losses = [
            np.mean(np.random.normal(size=(64, 100)), axis=0) for _ in range(1000)
        ]

        new_weights = agent.model.get_weights()

        # set new weights
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=dummy_losses, time_steps=time_steps)
        # print progress


        # yeu can also alter your managers parameters
        #manager.set_epsilon(epsilon=0.99)

        #if e % saving_after == 0:
            # you can save models
            #manager.save_model(saving_path, e)

    # and load mmodels
    #manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    print(agent.model.get_weights())
    manager.test(test_steps, test_episodes=10, render=True)
