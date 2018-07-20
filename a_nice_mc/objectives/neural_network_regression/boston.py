import numpy as np
from a_nice_mc.objectives.neural_network_regression.NN import NN


class Boston(NN):
    def __init__(self, name='boston'):
        data = np.load('data/boston/data.npy')
        labels = np.load('data/boston/labels.npy')

        D = data.shape[1]
        arch = [D + 1, 50, 1]
        self.theta_dim = np.sum([arch[i] * arch[i + 1] for i in range(len(arch) - 1)])
        super(Boston, self).__init__(data, labels, arch=arch)
        self.name = name

    @staticmethod
    def mean():
        return np.array(0.0)

    @staticmethod
    def std():
        return  np.array