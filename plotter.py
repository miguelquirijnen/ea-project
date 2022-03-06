import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self):
        self.best_res = []
        self.mean_res = []
        self.y_values = []

    def add_value(self, best, mean):
        self.best_res.append(best)
        self.mean_res.append(mean)

    def plot(self):
        plt.plot(self.best_res, label="Best fitness")
        plt.plot(self.mean_res, label="Mean Fitness")
        plt.ylabel('Best Fitness')
        plt.xlabel('Generation')
        plt.legend(loc='upper right')
        ax = plt.gca()
        ax.set_ylim([221000, 233000])
        ax.set_xlim([0, 194])

        plt.savefig("convergence.png")
