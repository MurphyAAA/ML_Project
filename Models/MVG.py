import pdb

import numpy as np
import scipy

from DataProc.ConfusionMatrix import ConfusionMatrix
from .Model import MLModel


class MVG(MLModel):
    def __init__(self, name="MVG"):
        MLModel.__init__(self, name)
        self.mu = None
        self.sigma = None

    def _vcol(v):
        return v.reshape((v.size, 1))

    def _vrow(v):
        return v.reshape((1, v.size))

    # log-density for a set of N
    def _logpdf_GAU_ND(x, mu, C):
        M = x.shape[0]
        a = M * np.log(2 * np.pi)
        _, b = np.linalg.slogdet(C)  # log|C|
        xc = (x - mu)
        c = np.dot(xc.T, np.linalg.inv(C))
        c = np.dot(c, xc)
        c = np.diagonal(c)
        return (-1.0 / 2.0) * (a + b + c)

    #overriding abstract method
    def train(self, DTR, LTR):
        DTRPos = DTR[:, LTR == 1]
        DTRNeg = DTR[:, LTR == 0]
        # calculate hyper parameter
        muPos = DTRPos.mean(1)
        muPos = MVG._vcol(muPos)
        DTRC0 = DTRPos - muPos
        CPos = np.dot(DTRC0, DTRC0.T) / DTRC0.shape[1]

        muNeg = DTRNeg.mean(1)
        muNeg = MVG._vcol(muNeg)
        DTRC1 = DTRNeg - muNeg
        CNeg = np.dot(DTRC1, DTRC1.T) / DTRC1.shape[1]

        self.mu = [muPos, muNeg]
        self.sigma = [CPos, CNeg]
        # apply parameter on evaluation set and get accuracy and error rate

    # overriding abstract method

    def score(self, DTE):
        assert self.mu is not None
        assert self.sigma is not None

        logPos = MVG._logpdf_GAU_ND(DTE, self.mu[0], self.sigma[0])
        logNeg = MVG._logpdf_GAU_ND(DTE, self.mu[1], self.sigma[1])
        score = logPos - logNeg

        return score

    def estimate(self, score, th):
        return (score > th).astype(np.int32)  # 1 for class positive, 0 for class negative

    def evaluation(self, DTE):
        score = self.score(DTE)
        return self.estimate(score, self.threshold)

    def validation(self, DTE, LTE):
        predictLabel = self.evaluation(DTE)
        cm = ConfusionMatrix(predictLabel, LTE)
        return cm.acc, cm.err


def main():
    from DataProc import load
    from DataProc.Prepare import gaussianize
    from DataProc.Prepare import normalize
    from DataProc import DataProvider
    from DataProc import KFold
    # load data from given path
    D, L = load('../Train.txt')
    # preprocessing the dataset
    normalize(gaussianize(D))
    dp = DataProvider(D, L)
    model = MVG()
    pi = dp.pi
    t = -np.log(pi / (1 - pi))
    model.set_threshold(t)
    model.train(dp.Data, dp.Label)
    print("Theoretical threshold: %.2f." % t)
    actDcf = model.actDcf(D, L, dp.pi)
    minDcf = model.minDcf(D, L, dp.pi)
    acc, err = model.validation(dp.Data, dp.Label)
    print("Model acc is %.4f%%" % (acc * 100))

    print(actDcf)
    print(minDcf)


if __name__ == "__main__":
    main()