import numpy as np


class Buffer(object):
    def __init__(self, data):
        self.data = data
        self.perm = np.random.permutation(self.data.shape[0])
        self.pointer = 0

    def insert(self, data):
        self.data = np.concatenate([self.data, data], axis=0)
        self.perm = np.random.permutation(self.data.shape[0])
        self.pointer = 0

    def set(self, data):
        self.data = data
        self.perm = np.random.permutation(self.data.shape[0])
        self.pointer = 0

    def discard(self, ratio=0.5):
        assert ratio > 0, ratio < 1
        sz = self.data.shape[0]
        self.perm = np.random.permutation(sz)
        self.data = self.data[self.perm[:int(sz * (1 - ratio))]]
        self.perm = np.random.permutation(self.data.shape[0])
        self.pointer = 0

    def __call__(self, batch_size):
        if self.pointer + batch_size >= self.data.shape[0]:
            self.pointer = 0
            self.perm = np.random.permutation(self.data.shape[0])
        start = self.pointer
        end = self.pointer + batch_size
        self.pointer = end

        return self.data[self.perm[start:end]]