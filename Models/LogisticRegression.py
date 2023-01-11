import numpy as np
import scipy
from Models.Model import MLModel


class LR(MLModel):
    __NAME__ = "LogicalRegression"
    __LAMBDA__ = 1e-9

    def __init__(self):
        self.w = None
        self.b = None

    def vcol(v):
        return v.reshape((v.size, 1))

    def vrow(v):
        return v.reshape((1, v.size))

    def logreg_obj_wrap(DTR, LTR,l):
        # v contains all model par w,b
        def logreg_obj(v):
            w = v[0:-1]
            b = v[-1]
            w_norm = np.linalg.norm(w)
            w = LR.vcol(w)
            reg_term = (l/ 2) * (w_norm ** 2)
            negz = -1 * (2 * LTR - 1)
            fx = np.dot(w.T, DTR) + b
            logJ = np.logaddexp(0, negz * fx)
            mean_logJ = logJ.mean()
            res = reg_term + mean_logJ
            res = res.reshape(res.size, )
            return res

        return logreg_obj

    # overriding abstract method
    def train(self, DTR, LTR):
        # DTR0 = DTR[:, LTR == 0]
        # DTR1 = DTR[:, LTR == 1]
        logreg_obj = LR.logreg_obj_wrap(DTR, LTR, LR.__LAMBDA__)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad=True)
        self.w = x[0:-1]
        self.b = x[-1]

    # overriding abstract method
    def validation(self, DTE, LTE):

        self.w = LR.vrow(self.w)
        s = np.dot(self.w, DTE) + self.b
        s = s.reshape(s.size, )
        predictLabel = []
        for i in s:
            if i > 0:
                predictLabel.append(1)
            else:
                predictLabel.append(0)

        res = predictLabel == LTE
        corr = res.sum()
        wrong = res.size - corr
        acc = corr / len(res)
        err = wrong / len(res)
        return acc, err
