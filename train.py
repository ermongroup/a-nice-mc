import numpy as np
import os
from objectives.expression.ring2d import Ring2d
from models.discriminator import MLPDiscriminator
from models.generator import create_nice_network
from train.wgan_gradient_penalty import Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = ''

energy_fn = Ring2d()
discriminator = MLPDiscriminator([400, 400, 400])
generator = create_nice_network(
    2, 2,
    [
        ([400], 'v1', False),
        ([400], 'x1', True),
        ([400], 'v2', False),
    ]
)


def noise_sampler(x):
    return np.random.normal(0.0, 1.0, [x, 2])


trainer = Trainer(generator, energy_fn, discriminator, noise_sampler, b=8, m=2)
trainer.train()
