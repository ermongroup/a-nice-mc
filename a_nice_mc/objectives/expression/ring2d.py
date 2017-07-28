import numpy as np
import tensorflow as tf
from a_nice_mc.objectives.expression import Expression
from a_nice_mc.utils.logger import create_logger

logger = create_logger(__name__)


class Ring2d(Expression):
    def __init__(self, name='ring2d', display=True):
        super(Ring2d, self).__init__(name=name, display=display)
        self.z = tf.placeholder(tf.float32, [None, 2], name='z')

    def __call__(self, z):
        with tf.variable_scope(self.name):
            z1 = tf.reshape(tf.slice(z, [0, 0], [-1, 1]), [-1])
            z2 = tf.reshape(tf.slice(z, [0, 1], [-1, 1]), [-1])
            v = (tf.sqrt(z1 * z1 + z2 * z2) - 2) / 0.4
            return v * v

    @staticmethod
    def mean():
        return np.array([0., 0.])

    @staticmethod
    def std():
        return np.array([1.456, 1.456])

    @staticmethod
    def xlim():
        return [-4, 4]

    @staticmethod
    def ylim():
        return [-4, 4]

