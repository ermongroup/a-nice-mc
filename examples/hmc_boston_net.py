import os
import sys


import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0.0, 1.0, [bs, 750])


if __name__ == '__main__':
    from a_nice_mc.utils.hmc import HamiltonianMonteCarloSampler
    from a_nice_mc.objectives.neural_network_regression.boston import Boston
    from a_nice_mc.utils.evaluation import batch_effective_sample_size
    from a_nice_mc.utils.logger import ensure_directory

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    energy_fn = Boston()
    hmc = HamiltonianMonteCarloSampler(energy_fn, prior)
    z = hmc.sample(1000, 1)
    print(z.shape)
    ensure_directory('logs/hmc_boston_net')
    np.save('logs/hmc_boston_net/hmc_lord_of_rings.npy', z)
    print(batch_effective_sample_size(z, None, None, None))
