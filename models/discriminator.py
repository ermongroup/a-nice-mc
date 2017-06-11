import tensorflow as tf
from utils.layers import dense, lrelu


class Discriminator(object):
    def __init__(self):
        self.name = 'discriminator'

    def __call__(self, x, reuse=True):
        raise NotImplementedError(str(type(self)))


class MLPDiscriminator(Discriminator):
    def __init__(self, dims):
        super(MLPDiscriminator).__init__()
        self.dims = dims

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            for dim in self.dims:
                x = dense(x, dim, activation_fn=lrelu)
            y = dense(x, 1, activation_fn=tf.identity)
        return y