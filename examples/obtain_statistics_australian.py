import sys

import numpy as np
import os

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0.0, 1.0, [bs, 15])

if __name__ == '__main__':
    from objectives.bayes_logistic_regression.australian import Australian
    from utils.statistics import obtain_statistics
    from utils.hmc import HamiltonianMonteCarloSampler
    from utils.logger import create_logger

    logger = create_logger(__name__)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    energy_fn = Australian(batch_size=32)
    sampler = HamiltonianMonteCarloSampler(
        energy_fn, prior, stepsize=0.0115, n_steps=40
    )
    obtain_statistics(sampler, steps=5000, burn_in=1000, batch_size=32)
