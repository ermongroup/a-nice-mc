import tensorflow as tf
import numpy as np
from objectives import Energy
from utils.evaluation import effective_sample_size
from utils.logger import get_logger

logger = get_logger(__name__)


class Ring2d(Energy):
    def __init__(self, name='u'):
        super(Ring2d, self).__init__()
        self.name = name
        self.z = tf.placeholder(tf.float32, [None, 2], name='z')

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

    def evaluate(self, z):
        effective_sample_size(z, self.mean(), self.std() * self.std(), logger=logger)