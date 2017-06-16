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

    @staticmethod
    def mean():
        return np.array([
            -0.73619639,  0.419458, -0.41486377,  0.12679717, -0.36520298, -0.1790139,
            -0.15307771,  0.01321516,  0.18079792, - 0.11101034, - 0.22463548,  0.12258933,
            0.02874339, -0.13638893, -0.29289896,  0.27896283, -0.29997425,  0.30485174,
            0.27133239,  0.12250612, -0.06301813, -0.09286941, -0.02542205, -0.02292937,
            -1.20507437
        ])

    @staticmethod
    def std():
        return np.array([
            0.09370191,  0.1066482,   0.097784,    0.11055009,  0.09572253,  0.09415687,
            0.08297686,  0.0928196,   0.10530122,  0.09953667,  0.07978824,  0.09610339,
            0.0867488,   0.09550436,  0.11943925,  0.08431934,  0.10446487,  0.12292658,
            0.11318609,  0.14040756,  0.1456459,   0.09062331,  0.13020753,  0.12852231,
            0.09891565
        ])
