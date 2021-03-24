import matplotlib.pyplot as plt
import os

class Experiment_Manager:
    def __init__(self, exp_name, repetition):
        self.repetition = repetition
        self.experiment_name = exp_name
        self.path = os.path.join('results', exp_name)

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def plot_steps(self, steps, file_extension='.png'):
        fig, ax = plt.subplots()
        ax.plot(steps)
        ax.set(xlabel='Episode', ylabel='avg. steps per trajectorie', title='Amount of steps taken by the Agent during training')
        path = os.path.join(self.path, 'steps_' + self.repetition + file_extension)
        plt.savefig(path)

    def plot_loss(self, losses, file_extension='.png'):
        fig, ax = plt.subplots()
        ax.plot(losses)
        ax.set(xlabel='Episode', ylabel='Loss', title='Loss after each Epoch')
        path = os.path.join(self.path, 'loss_' + self.repetition + file_extension)
        plt.savefig(path)
