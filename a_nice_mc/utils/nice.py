import tensorflow as tf

from a_nice_mc.utils.hmc import hamiltonian, metropolis_hastings_accept
from a_nice_mc.utils.layers import dense


class Layer(object):
    """
    Base method for implementing flow based models.
    `forward` and `backward` methods return two values:
     - the output of the layer
     - the resulting change of log-determinant of the Jacobian.
    """
    def __init__(self):
        pass

    def forward(self, inputs):
        raise NotImplementedError(str(type(self)))

    def backward(self, inputs):
        raise NotImplementedError(str(type(self)))


class NiceLayer(Layer):
    def __init__(self, dims, name='nice', swap=False):
        """
        NICE Layer that takes in [x, v] as input and updates one of them.
        Note that for NICE, the Jacobian is always 1; but we keep it for
        possible extensions to real NVP based flow models.
        :param dims: structure of the nice network
        :param name: TensorFlow variable name scope for variable reuse.
        :param swap: Update x if True, or update v if False.
        """
        super(NiceLayer, self).__init__()
        self.dims, self.reuse, self.swap = dims, False, swap
        self.name = 'generator/' + name

    def forward(self, inputs):
        x, v = inputs
        x_dim, v_dim = x.get_shape().as_list()[-1], v.get_shape().as_list()[-1]
        if self.swap:
            t = self.add(v, x_dim, reuse=self.reuse)
            x = x + t
        else:
            t = self.add(x, v_dim, reuse=self.reuse)
            v = v + t
        return [x, v], 0.0

    def backward(self, inputs):
        x, v, = inputs
        x_dim, v_dim = x.get_shape().as_list()[-1], v.get_shape().as_list()[-1]
        if self.swap:
            t = self.add(v, x_dim, reuse=True)
            x = x - t
        else:
            t = self.add(x, v_dim, reuse=True)
            v = v - t
        return [x, v], 0.0

    def add(self, x, dx, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            for dim in self.dims:
                x = dense(x, dim, activation_fn=tf.nn.relu)
            x = dense(x, dx)
            return x

    def create_variables(self, x_dim, v_dim):
        assert not self.reuse
        x = tf.zeros([1, x_dim])
        v = tf.zeros([1, v_dim])
        _ = self.forward([x, v])
        self.reuse = True


class NiceNetwork(object):
    def __init__(self, x_dim, v_dim):
        self.layers = []
        self.x_dim, self.v_dim = x_dim, v_dim

    def append(self, layer):
        layer.create_variables(self.x_dim, self.v_dim)
        self.layers.append(layer)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x, _ = layer.forward(x)
        return x

    def backward(self, inputs):
        x = inputs
        for layer in reversed(self.layers):
            x, _ = layer.backward(x)
        return x

    def __call__(self, x, is_backward):
        return tf.cond(
            is_backward,
            lambda: self.backward(x),
            lambda: self.forward(x)
        )


class TrainingOperator(object):
    def __init__(self, network):
        self.network = network

    def __call__(self, inputs, steps):
        def fn(zv, x):
            """
            Transition for training, without Metropolis-Hastings.
            `z` is the input state.
            `v` is created as a dummy variable to allow output of v_, for training p(v).
            :param x: variable only for specifying the number of steps
            :return: next state `z_`, and the corresponding auxiliary variable `v_`.
            """
            z, v = zv
            v = tf.random_normal(shape=tf.stack([tf.shape(z)[0], self.network.v_dim]))
            z_, v_ = self.network.forward([z, v])
            return z_, v_

        elems = tf.zeros([steps])
        return tf.scan(fn, elems, inputs, back_prop=True)


class InferenceOperator(object):
    def __init__(self, network, energy_fn):
        self.network = network
        self.energy_fn = energy_fn

    def __call__(self, inputs, steps, nice_steps=1):
        def nice_proposal(zv, x):
            """
            Nice Proposal (without Metropolis-Hastings).
            `z` is the input state.
            `v` is created as a dummy variable to allow output of v_, for debugging purposes.
            :param zv:
            :param x:
            :return: next state `z_`, and the corresponding auxiliary variable `v_' (without MH).
            """
            z, v = zv
            z_, v_ = self.network([z, v], is_backward=(x < 0.5)) #(tf.random_uniform([]) < 0.5))
            return z_, v_

        def fn(zv, x):
            """
            Transition with Metropolis-Hastings.
            `z` is the input state.
            `v` is created as a dummy variable to allow output of v_, for debugging purposes.
            :param zv: [z, v]. It is written in this form merely to appeal to Python 3.
            :param x: variable only for specifying the number of steps
            :return: next state `z_`, and the corresponding auxiliary variable `v_`.
            """
            z, v = zv
            v = tf.random_normal(shape=tf.stack([tf.shape(z)[0], self.network.v_dim]))
            # z_, v_ = self.network([z, v], is_backward=(tf.random_uniform([]) < 0.5))
            z_, v_ = tf.scan(nice_proposal, x * tf.random_uniform([]), (z, v), back_prop=False)
            z_, v_ = z_[-1], v_[-1]
            ep = hamiltonian(z, v, self.energy_fn)
            en = hamiltonian(z_, v_, self.energy_fn)
            accept = metropolis_hastings_accept(energy_prev=ep, energy_next=en)
            z_ = tf.where(accept, z_, z)
            return z_, v_

        elems = tf.ones([steps, nice_steps])
        return tf.scan(fn, elems, inputs, back_prop=False)
