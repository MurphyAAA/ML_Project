import pdb

import numpy as np


class ConfusionMatrix:
    def __init__(self, predicts, real_value):
        assert predicts.size == real_value.size

        self.num = predicts.size
        self.tp = ((predicts == 1) * (real_value == 1)).sum()
        self.tn = ((predicts == 0) * (real_value == 0)).sum()
        self.fp = ((predicts == 1) * (real_value == 0)).sum()
        self.fn = ((predicts == 0) * (real_value == 1)).sum()

        assert self.fn + self.fp + self.tp + self.tn == self.num

        self.acc = (self.tn + self.tp) / self.num
        self.err = (self.fp + self.fn) / self.num
        self.fnr = self.fn / (self.fn + self.tp)
        self.fpr = self.fp / (self.fp + self.tn)

    def computeCDF(self, pi):
        dcf = pi * self.fnr + (1 - pi) * self.fpr
        return dcf
