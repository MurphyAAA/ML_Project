## MVG ##

import numpy as np
import scipy
from Models.Model import MLModel

class NaiveBayes(MLModel):
    __NAME__ = "NaiveBayes"
    def __init__(self):
        self.mu = None
        self.sigma = None

    def vcol(v):
        return v.reshape((v.size, 1))

    def vrow(v):
        return v.reshape((1, v.size))

    # log-density for a set of N
    def logpdf_GAU_ND(x, mu, C):
        M = x.shape[0]
        a = M * np.log(2 * np.pi)
        _, b = np.linalg.slogdet(C)  # log|C|
        xc = (x - mu)
        c = np.dot(xc.T, np.linalg.inv(C))
        c = np.dot(c, xc)
        c = np.diagonal(c)
        return (-1.0 / 2.0) * (a + b + c)

    # overriding abstract method
    def train (self, DTR, LTR):
        DTR0 = DTR[:, LTR == 0]
        DTR1 = DTR[:, LTR == 1]
        # calculate hyper parameter
        mu0 = DTR0.mean(1)
        mu0 = NaiveBayes.vcol(mu0)
        DTRC0 = DTR0 - mu0
        C0 = np.dot(DTRC0, DTRC0.T) / DTRC0.shape[1]

        mu1 = DTR1.mean(1)
        mu1 = NaiveBayes.vcol(mu1)
        DTRC1 = DTR1 - mu1
        C1 = np.dot(DTRC1, DTRC1.T) / DTRC1.shape[1]
        identity = np.identity(DTR.shape[0])
        C0 = C0 * identity
        C1 = C1 * identity
        self.mu = [mu0, mu1]
        self.sigma = [C0, C1]
        # apply parameter on evaluation set and get accuracy and error rate

    # overriding abstract method
    def validation(self, DTE, LTE):

        logGau0 = NaiveBayes.logpdf_GAU_ND(DTE, self.mu[0], self.sigma[0])
        logGau1 = NaiveBayes.logpdf_GAU_ND(DTE, self.mu[1], self.sigma[1])

        jointLogDen0 = logGau0+np.log(2/3)
        jointLogDen1 = logGau1+np.log(1/3)

        logSJoint = np.vstack((jointLogDen0, jointLogDen1))
        logSMarginal = NaiveBayes.vrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        Spost = np.exp(logSPost)
        predictLabel = np.argmax(Spost, axis=0)
        res = predictLabel == LTE
        corr = res.sum()
        wrong = res.size - corr
        acc = corr / len(res)
        err = wrong / len(res)
        return acc, err
