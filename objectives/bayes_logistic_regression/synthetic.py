import numpy as np
from objectives.bayes_logistic_regression import BayesianLogisticRegression


class Synthetic(BayesianLogisticRegression):
    def __init__(self, name='synthetic', batch_size=32):
        n = 2000
        data = np.random.normal(0.0, 1.0, [n, 2])
        labels = np.zeros([n, 1])
        for i in range(0, n):
            if data[i, 0] > data[i, 1] - 1:
                labels[i] = 1

        super(Synthetic, self).__init__(data, labels, batch_size=batch_size)
        self.name = name
