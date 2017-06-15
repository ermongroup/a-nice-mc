import numpy as np
from objectives.bayes_logistic_regression import BayesianLogisticRegression


class German(BayesianLogisticRegression):
    def __init__(self, name='german', batch_size=32):
        data = np.load('data/german/data.npy')
        labels = np.load('data/german/labels.npy')

        # Normalize the f**king data!!!
        dm = np.mean(data, axis=0)
        ds = np.std(data, axis=0)
        data = (data - dm) / ds

        super(German, self).__init__(data, labels, batch_size=batch_size)
        self.name = name

