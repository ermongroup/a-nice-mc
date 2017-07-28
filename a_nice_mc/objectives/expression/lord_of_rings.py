import numpy as np
import tensorflow as tf
from a_nice_mc.objectives.expression import Expression
from a_nice_mc.utils.logger import create_logger


logger = create_logger(__name__)


class LordOfRings(Expression):
    def __init__(self, name='lord_of_rings', display=True):
        super(LordOfRings, self).__init__(name=name, display=display)
        self.z = tf.placeholder(tf.float32, [None, 2], name='z')

    def __call__(self, z):
        with tf.variable_scope(self.name):
            z1 = tf.reshape(tf.slice(z, [0, 0], [-1, 1]), [-1])
            z2 = tf.reshape(tf.slice(z, [0, 1], [-1, 1]), [-1])
            v1 = (tf.sqrt(z1 * z1 + z2 * z2) - 1) / 0.2
            v2 = (tf.sqrt(z1 * z1 + z2 * z2) - 2) / 0.2
            v3 = (tf.sqrt(z1 * z1 + z2 * z2) - 3) / 0.2
            v4 = (tf.sqrt(z1 * z1 + z2 * z2) - 4) / 0.2
            v5 = (tf.sqrt(z1 * z1 + z2 * z2) - 5) / 0.2
            p1, p2, p3, p4, p5 = v1 * v1, v2 * v2, v3 * v3, v4 * v4, v5 * v5
            return tf.minimum(tf.minimum(tf.minimum(tf.minimum(p1, p2), p3), p4), p5)

    @staticmethod
    def mean():
        return np.array([3.6])

    @staticmethod
    def std():
        return np.array([1.24])

    @staticmethod
    def xlim():
        return [-6, 6]

    @staticmethod
    def ylim():
        return [-6, 6]

    @staticmethod
    def statistics(z):
        z_ = np.sqrt(np.sum(np.square(z), axis=-1, keepdims=True))
        return z_
