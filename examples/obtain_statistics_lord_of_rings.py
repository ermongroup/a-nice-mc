import os
import sys

import numpy as np

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0.0, 1.0, [bs, 2])

if __name__ == '__main__':
    from a_nice_mc.objectives.expression.lord_of_rings import LordOfRings
    from a_nice_mc.utils.hmc import HamiltonianMonteCarloSampler
    from a_nice_mc.utils.statistics import obtain_statistics

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    energy_fn = LordOfRings(display=False)
    sampler = HamiltonianMonteCarloSampler(energy_fn, prior)
    obtain_statistics(sampler, steps=5000, burn_in=1000, batch_size=32)
