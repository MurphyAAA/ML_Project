import pdb
import matplotlib.pyplot as plt
from DataProc import *
from DataProc.ReadData import load
from DataProc.Prepare import gaussianize
from DataProc.Prepare import gaussianize1
from DataProc.Prepare import plot_hist
from DataProc.Prepare import normalize
import numpy as np
import Models


def main():
    # load data from given path
    D, L = load('Train.txt') # 一个sample是一列
    plot_hist(D,L)
    # preprocessing the dataset

    D = normalize(gaussianize(D))
    dp = DataProvider(D, L)
    kf = KFold(k=5)
    mvg = Models.MVG()
    dcf, mindcf = kf.dcf(mvg, dp)
    print("DCF=%f, minDcf=%f" % (dcf, mindcf))

  #  acc_nb, nb = kf.run(Models.NaiveBayes, dp)
  #   acc_td, td = kf.run(Models.TiedCov, dp)


    # print("Best acc for model {} is {:.2f} %.".format(mvg.__NAME__, acc_mvg * 100))
"""
    print("Best acc for model {} is {:.2f} %.".format(nb.__NAME__, acc_nb * 100))
    print("Best acc for model {} is {:.2f} %.".format(td.__NAME__, acc_td * 100))

# logistic Regression
    lambdas = [1e-6, 1e-3, 1e-1, 1]

    best_acc = 0
    best_lam = 0

    for lam in lambdas:
        Model = Models.LR
        Model.__LAMBDA__ = lam
        acc_lr, lr = kf.run(Model, dp)
        print("hyper parameter lambdas {} : best acc for model {} is {:.2f} %.".
              format(lam, lr.__NAME__,  acc_lr * 100))
        if acc_lr > best_acc:
            best_acc = acc_lr
            best_lam = lam
    print("Best acc for model {} with hyper parameter lambdas {} is {:.2f} %.".
          format(lr.__NAME__, best_lam, best_acc * 100))

"""
if __name__ == '__main__':
    # argparse
    main()
