class Energy(object):
    def __init__(self):
        pass

    def __call__(self, z):
        raise NotImplementedError(str(type(self)))

    @staticmethod
    def mean():
        return None

    @staticmethod
    def std():
        return None

    def _vector_to_model(self, v):
        return v

    @staticmethod
    def statistics(z):
        return z

    def evaluate(self, z, path=None):
        raise NotImplementedError(str(type(self)))
