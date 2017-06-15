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

    @staticmethod
    def mean():
        return np.array([
            0.00573914,  0.01986144, -0.15868089,  0.36768475,  0.72598995,  0.08102263,
            0.25611847,  1.68464095,  0.19636668,  0.65685423, -0.14652498,  0.15565136,
            -0.32924402,  1.6396836,  -0.31129081
        ])

    @staticmethod
    def std():
        return np.array([
            0.12749956,  0.13707998,  0.13329148,  0.12998348,  0.14871537,  0.14387384,
            0.16227234,  0.14832425,  0.16567627,  0.26399282,  0.12827283,  0.12381153,
            0.14707848,  0.56716324,  0.15749387,
        ])