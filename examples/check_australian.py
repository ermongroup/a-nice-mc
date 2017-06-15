import numpy as np
import tensorflow as tf
import os
import sys

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0.0, 1.0, [bs, 15])


if __name__ == '__main__':
    from objectives.bayes_logistic_regression.australian import Australian
    energy_fn = Australian(batch_size=None)
    a = np.zeros([1, 15])
    b = energy_fn(energy_fn.z)
    c = tf.gradients(tf.reduce_sum(energy_fn(energy_fn.z)), energy_fn.z)[0]
    sess = tf.Session()
    print(sess.run(b, feed_dict={energy_fn.z: a}))
    print(sess.run(c, feed_dict={energy_fn.z: a}))