import os
import sys

import numpy as np

sys.path.append(os.getcwd())


def noise_sampler(bs):
    return np.random.normal(0.0, 1.0, [bs, 2])

if __name__ == '__main__':
    from a_nice_mc.objectives.expression.mog6 import MixtureOfGaussians
    from a_nice_mc.models.discriminator import MLPDiscriminator
    from a_nice_mc.models.generator import create_nice_network
    from a_nice_mc.train.wgan_nll import Trainer

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    energy_fn = MixtureOfGaussians(display=False)
    discriminator = MLPDiscriminator([400, 400, 400])
    generator = create_nice_network(
        2, 2,
        [
            ([400], 'v1', False),
            ([400], 'x1', True),
            ([400], 'v2', False),
        ]
    )

    trainer = Trainer(generator, energy_fn, discriminator, noise_sampler, b=8, m=2)
    trainer.train()
