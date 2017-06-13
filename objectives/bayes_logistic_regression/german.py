import numpy as np
from objectives.bayes_logistic_regression import BayesianLogisticRegression


class German(BayesianLogisticRegression):
    def __init__(self, name='german', batch_size=32):
        data = np.load('data/german/data.npy')
        labels = np.load('data/german/labels.npy')
        super(German, self).__init__(data, labels, batch_size=batch_size)
        self.name = name
