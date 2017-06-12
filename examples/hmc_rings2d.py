import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0, 1, [bs, 2])


if __name__ == '__main__':
    from hmc import HamiltonianMonteCarloSampler
    from objectives.expression.ring2d import Ring2d
    energy_fn = Ring2d()
    hmc = HamiltonianMonteCarloSampler(energy_fn, prior)
    z = hmc.sample(2000, 2000)
    z = np.reshape(z, [-1, 2])
    x, y = z[:, 0], z[:, 1]
    plt.hist2d(x, y, bins=400)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.show()
