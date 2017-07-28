import numpy as np
from a_nice_mc.objectives.bayes_logistic_regression import BayesianLogisticRegression


class Heart(BayesianLogisticRegression):
    def __init__(self, name='heart', batch_size=32):
        data = np.load('data/heart/data.npy')
        labels = np.load('data/heart/labels.npy')

        # Normalize the f**king data!!!
        dm = np.mean(data, axis=0)
        ds = np.std(data, axis=0)
        data = (data - dm) / ds

        super(Heart, self).__init__(data, labels, batch_size=batch_size)
        self.name = name

    @staticmethod
    def mean():
        return np.array([
            -0.13996868,  0.71390106,  0.69571619,  0.43944853,  0.36997702, -0.27319424,
            0.31730518, -0.49617367,  0.40516419, 0.4312388,   0.26531786, 1.10337417,
            0.70054367, -0.25684964
        ])

    @staticmethod
    def std():
        return np.array([
            0.22915648,  0.24545612,  0.20457998,  0.20270157,  0.21040644,  0.20094482,
            0.19749419,  0.24134014,  0.20230987,  0.25595334,  0.23709087,  0.24735325,
            0.20701178,  0.19771984
        ])
