import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0, 1, [bs, 2])


if __name__ == '__main__':
    from a_nice_mc.utils.statistics import NormalMonteCarloSampler
    from a_nice_mc.objectives.expression.ring2d import Ring2d

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    energy_fn = Ring2d(display=True)
    hmc = NormalMonteCarloSampler(energy_fn, prior)
    z = hmc.sample(8000, 8000)
    z = z[:, 3000:]
    z = np.reshape(z, [-1, 2])
    x, y = z[:, 0], z[:, 1]
    z = np.reshape(z, [-1, 2])
    plt.hist2d(x, y, bins=400)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    #plt.savefig('ring2d.png')
    print(np.mean(x))
    print(np.std(x))
