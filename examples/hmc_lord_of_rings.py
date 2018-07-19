import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0.0, 1.0, [bs, 2])


if __name__ == '__main__':
    from a_nice_mc.utils.hmc import HamiltonianMonteCarloSampler
    from a_nice_mc.objectives.expression.lord_of_rings import LordOfRings

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    energy_fn = LordOfRings(display=True)
    hmc = HamiltonianMonteCarloSampler(energy_fn, prior)
    z = hmc.sample(100, 8000)
    z = np.reshape(z, [-1, 2])
    x, y = z[:, 0], z[:, 1]
    plt.hist2d(x, y, bins=400)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.show()
    np.save('hmc_lord_of_rings.npy', z)
