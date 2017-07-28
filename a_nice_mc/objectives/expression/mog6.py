import numpy as np
import tensorflow as tf
from a_nice_mc.objectives.expression import Expression
from a_nice_mc.utils.logger import create_logger

logger = create_logger(__name__)


class MixtureOfGaussians(Expression):
    def __init__(self, name='mog6', display=True):
        super(MixtureOfGaussians, self).__init__(name=name, display=display)
        self.z = tf.placeholder(tf.float32, [None, 2], name='z')

    def __call__(self, z):
        z1 = tf.reshape(tf.slice(z, [0, 0], [-1, 1]), [-1])
        z2 = tf.reshape(tf.slice(z, [0, 1], [-1, 1]), [-1])
        v1 = tf.sqrt((z1 - 5) * (z1 - 5) + z2 * z2) * 2
        v2 = tf.sqrt((z1 + 5) * (z1 + 5) + z2 * z2) * 2
        v3 = tf.sqrt((z1 - 2.5) * (z1 - 2.5) + (z2 - 2.5 * np.sqrt(3)) * (z2 - 2.5 * np.sqrt(3))) * 2
        v4 = tf.sqrt((z1 + 2.5) * (z1 + 2.5) + (z2 + 2.5 * np.sqrt(3)) * (z2 + 2.5 * np.sqrt(3))) * 2
        v5 = tf.sqrt((z1 - 2.5) * (z1 - 2.5) + (z2 + 2.5 * np.sqrt(3)) * (z2 + 2.5 * np.sqrt(3))) * 2
        v6 = tf.sqrt((z1 + 2.5) * (z1 + 2.5) + (z2 - 2.5 * np.sqrt(3)) * (z2 - 2.5 * np.sqrt(3))) * 2
        pdf1 = tf.exp(-0.5 * v1 * v1) / tf.sqrt(2 * np.pi * 0.25)
        pdf2 = tf.exp(-0.5 * v2 * v2) / tf.sqrt(2 * np.pi * 0.25)
        pdf3 = tf.exp(-0.5 * v3 * v3) / tf.sqrt(2 * np.pi * 0.25)
        pdf4 = tf.exp(-0.5 * v4 * v4) / tf.sqrt(2 * np.pi * 0.25)
        pdf5 = tf.exp(-0.5 * v5 * v5) / tf.sqrt(2 * np.pi * 0.25)
        pdf6 = tf.exp(-0.5 * v6 * v6) / tf.sqrt(2 * np.pi * 0.25)
        return -tf.log((pdf1 + pdf2 + pdf3 + pdf4 + pdf5 + pdf6) / 6)

    @staticmethod
    def mean():
        return np.array([0.0, 0.0])

    @staticmethod
    def std():
        return np.array([3.57, 3.57])

    @staticmethod
    def statistics(z):
        return z

    @staticmethod
    def xlim():
        return [-8, 8]

    @staticmethod
    def ylim():
        return [-8, 8]
