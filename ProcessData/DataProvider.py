import numpy as np

class DataProvider:
    def __init__(self, D, L, seed=0):
        # read data
        # shuffle data
        # np.random.seed(seed)
        self.idx = np.random.permutation(D.shape[1])
        # counter
        counter = 0

        self.D = D
        self.L = L

    def get(self, K, counter):
        nTrain = int(self.D.shape[1] * 1.0 / K)

        idxTrain = self.idx[counter * nTrain: (counter + 1) * nTrain]
        idxTest = list(set(np.arange(self.D.shape[1])) - (set(idxTrain)))
        DTR = self.D[:, idxTrain]
        DTE = self.D[:, idxTest]
        LTR = self.L[idxTrain]
        LTE = self.L[idxTest]
        # data = generateData(counter)

        return DTR, DTE, LTR, LTE
