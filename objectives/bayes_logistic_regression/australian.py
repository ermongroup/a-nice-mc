import numpy as np
from objectives.bayes_logistic_regression import BayesianLogisticRegression


class Australian(BayesianLogisticRegression):
    def __init__(self, name='australian', batch_size=32):
        data = np.load('data/australian/data.npy')
        labels = np.load('data/australian/labels.npy')

        # Normalize the f**king data!!!
        dm = np.mean(data, axis=0)
        ds = np.std(data, axis=0)
        data = (data - dm) / ds

        super(Australian, self).__init__(data, labels, batch_size=batch_size)
        self.name = name
