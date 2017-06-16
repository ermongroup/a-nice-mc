import numpy as np
import os
import sys

sys.path.append(os.getcwd())


def noise_sampler(bs):
    return np.random.normal(0.0, 1.0, [bs, 2])

if __name__ == '__main__':
    from objectives.expression.ring2d import Ring2d
    from models.discriminator import MLPDiscriminator
    from models.generator import create_nice_network
    from train.wgan_nll import Trainer

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    energy_fn = Ring2d(display=False)
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
