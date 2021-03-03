from psutil import cpu_count
import ray
from collections import deque

cpu_cores = cpu_count(logical=False)

class ERP_Memory_Actor:
    def __init__(self, memory_len=100_000):
        self.memory = deque(maxlen=memory_len)
        self.agent = DQN_Agent()

    @ray.remote(num_cpus=cpu_cores)
    def get_sample(self, idx, agent_=DQN_Agent, env_str='CartPole'):
        env = gym.make(env_str)
        agent = agent_()


        # create list to append samples
        # sample from env

        # return sample_list, idx

        # idx for reinstantiation


    def remember(self, trajectorie):
        list = [get_sample.remote(idx=cpu_core) for cpu_core in range(cpu_cores)]

        self.memory.append(trajectorie)

    def get_memory(self):
        pass
