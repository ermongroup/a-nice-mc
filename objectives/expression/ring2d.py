import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from objectives import Energy
from utils.evaluation import effective_sample_size
from utils.logger import get_logger

logger = get_logger(__name__)


class Ring2d(Energy):
    def __init__(self, name='u', display=True):
        super(Ring2d, self).__init__()
        self.name = name
        self.z = tf.placeholder(tf.float32, [None, 2], name='z')
        self.display = display
        if display:
            plt.ion()
        else:
            matplotlib.use('Agg')
        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=2, ncols=1)

    def __call__(self, z):
        with tf.variable_scope(self.name):
            z1 = tf.reshape(tf.slice(z, [0, 0], [-1, 1]), [-1])
            z2 = tf.reshape(tf.slice(z, [0, 1], [-1, 1]), [-1])
            v = (tf.sqrt(z1 * z1 + z2 * z2) - 2) / 0.4
            return 0.5 * v * v

    @staticmethod
    def mean():
        return np.array([0., 0.])

    @staticmethod
    def std():
        return np.array([1.5, 1.5])

    @staticmethod
    def statistics(z):
        return z

    def evaluate(self, zv, path):
        z, v = zv
        ess = effective_sample_size(z, self.mean(), self.std() * self.std(), logger=logger)
        self.visualize(zv)

    @staticmethod
    def xlim():
        return [-4, 4]

    @staticmethod
    def ylim():
        return [-4, 4]

    def visualize(self, zv, path):
        self.ax1.clear()
        self.ax2.clear()
        z, v = zv
        z = np.reshape(z, [-1, 2])
        self.ax1.hist2d(z[:, 0], z[:, 1], bins=400)
        self.ax1.set(xlim=self.xlim(), ylim=self.ylim())

        v = np.reshape(v, [-1, 2])
        self.ax2.hist2d(v[:, 0], v[:, 1], bins=400)
        self.ax2.set(xlim=self.xlim(), ylim=self.ylim())

        if self.display:
            plt.show()
            plt.pause(0.1)
        else:
            self.fig.savefig(path + '/visualize.png')
