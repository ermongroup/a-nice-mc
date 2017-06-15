import numpy as np
import os
import sys

sys.path.append(os.getcwd())


def noise_sampler(bs):
    return np.random.normal(0.0, 1.0, [bs, 3])

if __name__ == '__main__':
    from objectives.bayes_logistic_regression.synthetic import Synthetic
    from models.discriminator import MLPDiscriminator
    from models.generator import create_nice_network
    from train.wgan_nll import Trainer

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    energy_fn = Synthetic(batch_size=32)
    discriminator = MLPDiscriminator([800, 800, 800])
    generator = create_nice_network(
        3, 10,
        [
            ([400], 'v1', False),
            ([400, 400], 'x1', True),
            ([400], 'v2', False),
        ]
    )

    trainer = Trainer(generator, energy_fn, discriminator, noise_sampler, b=16, m=4, eta=5)
    trainer.train(bootstrap_steps=3000, bootstrap_burn_in=1000)
