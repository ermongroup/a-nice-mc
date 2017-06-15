import numpy as np
import os
import sys

sys.path.append(os.getcwd())


def prior(bs):
    return np.random.normal(0.0, 1.0, [bs, 3])

if __name__ == '__main__':
    from objectives.bayes_logistic_regression.synthetic import Synthetic
    from utils.statistics import obtain_statistics, NormalMonteCarloSampler
    from utils.statistics import HamiltonianMonteCarloSampler
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    energy_fn = Synthetic(batch_size=100)
    # sampler = HamiltonianMonteCarloSampler(
    #     energy_fn, prior, stepsize=0.005, n_steps=40,
    #     inter_op_parallelism_threads=8,
    #     intra_op_parallelism_threads=8
    # )
    sampler = NormalMonteCarloSampler(
        energy_fn, prior,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8
    )
    obtain_statistics(sampler, steps=30000, burn_in=1000, batch_size=100)
