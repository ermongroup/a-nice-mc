import os
import sys
import argparse

import numpy as np

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0.0, 1.0, [bs, 14])

if __name__ == '__main__':
    from a_nice_mc.objectives.bayes_logistic_regression.heart import Heart
    from a_nice_mc.utils.statistics import obtain_statistics
    from a_nice_mc.utils.hmc import HamiltonianMonteCarloSampler
    from a_nice_mc.utils.logger import create_logger

    parser = argparse.ArgumentParser()
    parser.add_argument('--stepsize', type=float, default=0.01)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    logger = create_logger(__name__)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    energy_fn = Heart(batch_size=32)
    sampler = HamiltonianMonteCarloSampler(
        energy_fn, prior, stepsize=args.stepsize, n_steps=40
    )
    obtain_statistics(sampler, steps=5000, burn_in=1000, batch_size=32)
