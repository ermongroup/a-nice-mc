import numpy as np
import tensorflow as tf
from a_nice_mc.objectives import Energy
from a_nice_mc.utils.evaluation import batch_effective_sample_size as effective_sample_size
from a_nice_mc.utils.evaluation import acceptance_rate
from a_nice_mc.utils.logger import save_ess, create_logger

logger = create_logger(__name__)


class BayesianLogisticRegression(Energy):
    def __init__(self, data, labels, batch_size=None,
                 loc=0.0, scale=1.0):
        """
        Bayesian Logistic Regression model (assume Normal prior)
        :param data: data for Logistic Regression task
        :param labels: label for Logistic Regression task
        :param batch_size: batch size for Logistic Regression; setting it to None
        adds flexibility at the cost of speed.
        :param loc: mean of the Normal prior
        :param scale: std of the Normal prior
        """
        super(BayesianLogisticRegression, self).__init__()
        self.x_dim = data.shape[1]
        self.y_dim = labels.shape[1]
        self.dim = self.x_dim * self.y_dim + self.y_dim
        self.mu_prior = tf.ones([self.dim]) * loc
        self.sig_prior = tf.ones([self.dim]) * scale

        self.data = tf.constant(data, tf.float32)
        self.labels = tf.constant(labels, tf.float32)
        self.z = tf.placeholder(tf.float32, [batch_size, self.dim])

        if batch_size:
            self.data = tf.tile(
                tf.reshape(self.data, [1, -1, self.x_dim]),
                tf.stack([batch_size, 1, 1])
            )
            self.labels = tf.tile(
                tf.reshape(self.labels, [1, -1, self.y_dim]),
                tf.stack([batch_size, 1, 1])
            )
        else:
            self.data = tf.tile(
                tf.reshape(self.data, [1, -1, self.x_dim]),
                tf.stack([tf.shape(self.z)[0], 1, 1])
            )
            self.labels = tf.tile(
                tf.reshape(self.labels, [1, -1, self.y_dim]),
                tf.stack([tf.shape(self.z)[0], 1, 1])
            )

    def _vector_to_model(self, v):
        w = v[:, :-self.y_dim]
        b = v[:, -self.y_dim:]
        w = tf.reshape(w, [-1, self.x_dim, self.y_dim])
        b = tf.reshape(b, [-1, 1, self.y_dim])
        return w, b

    def energy_fn(self, v, x, y):
        w, b = self._vector_to_model(v)
        logits = tf.matmul(x, w) + b
        ll = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        ll = tf.reduce_sum(ll, axis=[1, 2])
        pr = tf.square((v - self.mu_prior) / self.sig_prior)
        pr = 0.5 * tf.reduce_sum(pr, axis=1)
        return pr + ll

    def __call__(self, v):
        return self.energy_fn(v, self.data, self.labels)

    def evaluate(self, zv, path=None):
        z, v = zv
        z_ = np.reshape(z, [-1, z.shape[-1]])
        m = np.mean(z_, axis=0, dtype=np.float64)
        v = np.std(z_, axis=0, dtype=np.float64)
        print('mean: {}'.format(m))
        print('std: {}'.format(v))
        logger.info('Acceptance rate %.4f' % (acceptance_rate(z)))
        ess = effective_sample_size(
            z,
            self.mean(), self.std() * self.std(),
            logger=logger
        )
        if path:
            save_ess(ess, path)

    @staticmethod
    def mean():
        return None

    @staticmethod
    def std():
        return None
