import tensorflow as tf
import numpy as np
import time
import os
from nice import TrainingOperator, InferenceOperator
from utils.bootstrap import Buffer
from utils.logger import create_logger


class Trainer(object):
    def __init__(self,
                 network, energy_fn, discriminator,
                 noise_sampler,
                 b, m, eta=1.0, scale=10.0):
        self.energy_fn = energy_fn
        self.logger = create_logger(__name__)
        self.train_op = TrainingOperator(network)
        self.infer_op = InferenceOperator(network, energy_fn)
        self.b = tf.to_int32(tf.reshape(tf.multinomial(tf.ones([1, b]), 1), [])) + 1
        self.m = tf.to_int32(tf.reshape(tf.multinomial(tf.ones([1, m]), 1), [])) + 1
        self.network = network
        self.x_dim, self.v_dim = network.x_dim, network.v_dim

        self.z = tf.placeholder(tf.float32, [None, self.x_dim])
        self.x = tf.placeholder(tf.float32, [None, self.x_dim])
        self.xl = tf.placeholder(tf.float32, [None, self.x_dim])
        self.steps = tf.placeholder(tf.int32, [])
        bx, bz = tf.shape(self.x)[0], tf.shape(self.z)[0]

        # Obtain values from inference ops
        # `infer_op` contains Metropolis step
        v = tf.random_normal(tf.stack([bz, self.v_dim]))
        self.z_, self.v_ = self.infer_op((self.z, v), self.steps)

        # Reshape for pairwise discriminator
        x = tf.reshape(self.x, [-1, 2 * self.x_dim])
        xl = tf.reshape(self.xl, [-1, 2 * self.x_dim])

        # Obtain values from train ops
        v1 = tf.random_normal(tf.stack([bz, self.v_dim]))
        x1_, v1_ = self.train_op((self.z, v1), self.b)
        x1_ = x1_[-1]
        x1_sg = tf.stop_gradient(x1_)
        v2 = tf.random_normal(tf.stack([bx, self.v_dim]))
        x2_, v2_ = self.train_op((self.x, v2), self.m)
        x2_ = x2_[-1]
        v3 = tf.random_normal(tf.stack([bx, self.v_dim]))
        x3_, v3_ = self.train_op((x1_sg, v3), self.m)
        x3_ = x3_[-1]

        # The pairwise discriminator has two components:
        # (x, x2) from x -> x2
        # (x1, x3) from z -> x1 -> x3
        #
        # The optimal case is achieved when x1, x2, x3
        # are all from the data distribution
        x_ = tf.concat([
                tf.concat([x2_, self.x], 1),
                tf.concat([x3_, x1_], 1)
        ], 0)

        # Concat all v values for log-likelihood training
        v_ = tf.concat([v1_, v2_, v3_], 0)
        v_ = tf.reshape(v_, [-1, self.v_dim])

        d = discriminator(x, reuse=False)
        d_ = discriminator(x_)

        # generator loss
        self.v_loss = tf.reduce_mean(0.5 * tf.multiply(v_, v_))
        self.g_loss = tf.reduce_mean(d_) + self.v_loss * eta

        # discriminator loss
        self.d_loss = tf.reduce_mean(d) - tf.reduce_mean(d_)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = xl * epsilon + x_ * (1 - epsilon)
        d_hat = discriminator(x_hat)
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.norm(ddx, axis=1)
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        self.d_loss = self.d_loss + ddx

        # I don't have a good solution to the tf variable scope mess.
        # So I basically force the NiceLayer to contain the 'generator' scope.
        # See `nice/__init__.py`.
        g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        d_vars = [var for var in tf.global_variables() if discriminator.name in var.name]

        self.d_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
            .minimize(self.d_loss, var_list=d_vars)
        self.g_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
            .minimize(self.g_loss, var_list=g_vars)

        self.init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
            gpu_options=gpu_options,
        ))
        self.sess.run(self.init_op)
        self.ns = noise_sampler
        self.ds = None
        self.path = 'logs/' + energy_fn.name
        try:
            os.makedirs(self.path)
        except OSError:
            pass

    def sample(self, steps=2000, batch_size=100):
        start = time.time()
        z, v = self.sess.run([self.z_, self.v_], feed_dict={self.z: self.ns(batch_size), self.steps: steps})
        end = time.time()
        self.logger.info('A-NICE-MC: batches [%d] steps [%d] time [%5.4f] samples/s [%5.4f]' %
                         (batch_size, steps, end - start, (batch_size * steps) / (end - start)))
        z = np.transpose(z, axes=[1, 0, 2])
        v = np.transpose(v, axes=[1, 0, 2])
        return z, v

    def bootstrap(self, steps=5000, burn_in=1000, batch_size=100, discard_ratio=0.5):
        z, _ = self.sample(steps + burn_in, batch_size)
        z = np.reshape(z[:, burn_in:], [-1, z.shape[-1]])
        if self.ds:
            self.ds.discard(ratio=discard_ratio)
            self.ds.insert(z)
        else:
            self.ds = Buffer(z)

    def train(self, d_iters=5, epoch_size=500, log_freq=100, max_iters=100000):
        def _feed_dict(bs):
            return {
                self.z: self.ns(bs),
                self.x: self.ds(bs),
                self.xl: self.ds(4 * bs)
            }
        batch_size = 32
        train_time = 0
        for t in range(0, max_iters):
            if t % epoch_size == 0:
                self.bootstrap()
                z, v = self.sample(steps=2000)
                z, v = z[:, 1000:], v[:, 1000:]
                self.energy_fn.evaluate([z, v], path=self.path)
                # TODO: save model
            if t % log_freq == 0:
                d_loss = self.sess.run(self.d_loss, feed_dict=_feed_dict(batch_size))
                g_loss, v_loss = self.sess.run([self.g_loss, self.v_loss], feed_dict=_feed_dict(batch_size))
                self.logger.info('Iter [%d] time [%5.4f] d_loss [%.4f] g_loss [%.4f] v_loss [%.4f]' %
                                 (t, train_time, d_loss, g_loss, v_loss))
            start = time.time()
            for _ in range(0, d_iters):
                self.sess.run(self.d_train, feed_dict=_feed_dict(batch_size))
            self.sess.run(self.g_train, feed_dict=_feed_dict(batch_size))
            end = time.time()
            train_time += end - start

    def load(self):
        # TODO: load model
        raise NotImplementedError(str(type(self)))

    def save(self):
        # TODO: save model
        raise NotImplementedError(str(type(self)))
