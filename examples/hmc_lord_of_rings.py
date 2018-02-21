import os
import sys

import numpy as np

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0.0, 1.0, [bs, 2])


if __name__ == '__main__':
    from a_nice_mc.utils.statistics import NormalMonteCarloSampler
    from a_nice_mc.objectives.expression.lord_of_rings import LordOfRings

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    energy_fn = LordOfRings(display=False)
    hmc = NormalMonteCarloSampler(energy_fn, prior)
    z = hmc.sample(10, 8000)
    np.save('hmc_lord_of_rings.npy', z)
