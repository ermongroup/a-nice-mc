import numpy as np
import os
import sys

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0.0, 1.0, [bs, 2])

if __name__ == '__main__':
    from objectives.expression.mog import MixtureOfGaussians
    from utils.statistics import obtain_statistics
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    energy_fn = MixtureOfGaussians(display=False)
    obtain_statistics(energy_fn, prior, steps=10000, burn_in=3000, batch_size=8000)
