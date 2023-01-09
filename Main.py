import pdb

from preProcess import *
from preProcess.ReadData import load
from preProcess.Prepare import gaussianize
from preProcess.Prepare import normalize
import Models


def main():
    # load data from given path
    D, L = load('Train.txt')
    # preprocessing the dataset
    normalize(gaussianize(D))

    dp = DataProvider(D, L)
    kf = KFold(k=5)
    acc_mvg, mvg = kf.run(Models.MVG, dp)
    acc_nb, nb = kf.run(Models.NaiveBayes, dp)
    acc_td, td = kf.run(Models.TiedCov, dp)
    print("Best acc for model {} is {:.2f} %.".format(mvg.__NAME__, acc_mvg * 100))
    print("Best acc for model {} is {:.2f} %.".format(nb.__NAME__, acc_nb * 100))
    print("Best acc for model {} is {:.2f} %.".format(td.__NAME__, acc_td * 100))


if __name__ == '__main__':
    # argparse
    main()
