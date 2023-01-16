from abc import ABC, abstractmethod
import numpy as np

from DataProc.ConfusionMatrix import ConfusionMatrix


class MLModel (ABC):

    def __init__(self, name):
        self.name = name
        self.threshold = 0

    def set_threshold(self, threshold):
        self.threshold = threshold

    @abstractmethod
    def train(self, x, y):
        pass
    @abstractmethod
    def validation(self, x, y):
        pass
    @abstractmethod
    def evaluation(self, x):
        pass

    @abstractmethod
    def score(self, x):
        pass

    @abstractmethod
    def estimate(self, score, th):
        pass

    @abstractmethod
    def score(self, x):
        pass

    def dcf(self, score, label, pi, th):
        predictLabel = self.estimate(score, th)
        cm = ConfusionMatrix(predictLabel, label)
        return cm.computeCDF(pi)

    def actDcf(self, data, label, pi):
        th = -np.log(pi / (1 - pi))
        score = self.score(data)
        return self.dcf(score, label, pi, th)

    def minDcfByScore(self, score, label, pi):
        scoreList = score.tolist()
        scoreList.append(np.inf)
        scoreList.append(-np.inf)
        dcf = [self.dcf(score, label, pi, t) for t in scoreList]
        from operator import itemgetter
        index, minV = min(enumerate(dcf), key=itemgetter(1))
        # print("min threshold: %.2f" % scoreList[index])
        return minV

    def minDcf(self, data, label, pi):
        score = self.score(data)
        return self.minDcfByScore(score, label, pi)

