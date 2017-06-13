import numpy as np
import os
import sys

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0.0, 1.0, [bs, 25])

if __name__ == '__main__':
    from objectives.bayes_logistic_regression.german import German
    from utils.statistics import obtain_statistics
    from utils.statistics import HamiltonianMonteCarloSampler
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    energy_fn = German(batch_size=100)
    sampler = HamiltonianMonteCarloSampler(
        energy_fn, prior, stepsize=0.005, n_steps=10,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8
    )
    obtain_statistics(sampler, steps=5000, burn_in=1000, batch_size=100)
