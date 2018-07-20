import numpy as np
import tensorflow as tf
from a_nice_mc.objectives import Energy
from a_nice_mc.utils.evaluation import batch_effective_sample_size as effective_sample_size
from a_nice_mc.utils.evaluation import  acceptance_rate
from a_nice_mc.utils.logger import save_ess, create_logger

logger = create_logger(__name__)


class NN(Energy):
    def __init__(self, data, labels, arch, act=tf.nn.tanh, prec=1.0):
        """
        Bayesian Neural Network Model (assumes factored Normal prior)
        :param data: data for Regression task
        :param labels: label for Regression task
        :param scale: std of the Normal prior
        :param arch: list of layer widths for feed-forward network
        """
        super(NN, self).__init__()
        self.arch = arch
        self.theta_dim = np.sum([arch[i] * arch[i + 1] for i in range(len(arch) - 1)])
        self.act = act
        self.x_dim = data.shape[1]
        self.y_dim = labels.shape[1]
        self.prec_prior = prec
        self.z = tf.placeholder(tf.float32, [None, self.theta_dim])

        self.data = tf.constant(data, tf.float32)
        self.labels = tf.constant(labels, tf.float32)

    def _unflatten(self, theta):
        """theta is assumed to have shape (num_chains, target_dim)"""
        m = tf.shape(theta)[0]  # num chains
        weights = []
        start = 0
        for i in range(len(self.arch) - 1):
            size = self.arch[i] * self.arch[i + 1]
            w = tf.reshape(theta[:, start:start + size],
                           (m, self.arch[i], self.arch[i + 1]))
            weights.append(w)
            start += size
        return weights

    def energy_fn(self, theta, x, y):
        """ theta has shape  (num_chains, target_dim)
            We subsume the biases into the weight matrices by appending ones to
            the hidden state."""
        h = tf.expand_dims(x, 0)
        h = tf.concat([h, tf.ones((1, h.shape[1], 1))], axis=2)
        h = tf.tile(h, [tf.shape(theta)[0], 1, 1])
        weights = self._unflatten(theta)
        for W in weights[:-1]:
            h = self.act(h @ W)
        mean = h @ weights[-1]
        mahalob = 0.5 * tf.reduce_sum((y - mean) ** 2, axis=2)
        prior = 0.5 * tf.reduce_sum(theta ** 2, axis=1, keepdims=True)

        return tf.reduce_sum(mahalob + self.prec_prior * prior, axis=1)

    def __call__(self, v):
        return self.energy_fn(v, self.data, self.labels)

    def evaluate(self, zv, path=None):
        z, v = zv
        z_ = np.reshape(z, [-1, z.shape[-1]])
        logger.info('Acceptance rate %.4f' % (acceptance_rate(z)))
        ess = effective_sample_size(
            z,
            None, None,
            logger=logger
        )
        if path:
            save_ess(ess, path)
            np.save(path +'/trajectory.npy', z)

    @staticmethod
    def mean():
        return None

    @staticmethod
    def std():
        return None
