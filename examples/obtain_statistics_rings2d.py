import numpy as np
import os
import sys

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0.0, 1.0, [bs, 2])

if __name__ == '__main__':
    from objectives.expression.ring2d import Ring2d
    from hmc import HamiltonianMonteCarloSampler
    from utils.statistics import obtain_statistics, NormalMonteCarloSampler
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    energy_fn = Ring2d(display=False)
    sampler = HamiltonianMonteCarloSampler(energy_fn, prior)
    obtain_statistics(sampler, steps=1000, burn_in=300, batch_size=800)
