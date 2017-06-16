import numpy as np
from objectives import Energy
from utils.evaluation import effective_sample_size, acceptance_rate
from utils.logger import save_ess, create_logger

logger = create_logger(__name__)


class Expression(Energy):
    def __init__(self, name='expression', display=True):
        super(Expression, self).__init__()
        self.name = name
        self.display = display
        if display:
            import matplotlib.pyplot as plt
            plt.ion()
        else:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=2, ncols=1)

    def __call__(self, z):
        raise NotImplementedError(str(type(self)))

    @staticmethod
    def xlim():
        return None

    @staticmethod
    def ylim():
        return None

    def evaluate(self, zv, path=None):
        z, v = zv
        logger.info('Acceptance rate %.4f' % (acceptance_rate(z)))
        z = self.statistics(z)
        ess = effective_sample_size(z, self.mean(), self.std() * self.std(), logger=logger)
        if path:
            save_ess(ess, path)
        self.visualize(zv, path)

    def visualize(self, zv, path):
        self.ax1.clear()
        self.ax2.clear()
        z, v = zv
        if path:
            np.save(path + '/trajectory.npy', z)

        z = np.reshape(z, [-1, 2])
        self.ax1.hist2d(z[:, 0], z[:, 1], bins=400)
        self.ax1.set(xlim=self.xlim(), ylim=self.ylim())

        v = np.reshape(v, [-1, 2])
        self.ax2.hist2d(v[:, 0], v[:, 1], bins=400)
        self.ax2.set(xlim=self.xlim(), ylim=self.ylim())

        if self.display:
            import matplotlib.pyplot as plt
            plt.show()
            plt.pause(0.1)
        elif path:
            self.fig.savefig(path + '/visualize.png')
