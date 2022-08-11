import sys

import m_WineDetect as wd;

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np


def cross_validation_SVM():
    X, LTR = wd.load("./Train.txt")
    # T 变成index可以直接提取一组数据
    sc = StandardScaler()
    DTR_std = sc.fit_transform(X)

    LTR = LTR.reshape(1, -1)
    # Kernel_SVM_RBF = wd.train_SVM(DTR_std, LTR, C=1, sigma=0.39, K=0)
    K = 4

    # print("DTR_std:", DTR_std)
    # print("LTR:", LTR)  # 1839
    N = X.shape[1]
    copyX = DTR_std.copy();
    copyY = LTR.copy();
    min = sys.maxsize;
    res = {}
    for i in range(K):

        num = int(N / K)
        print(num)
        X_test = copyX[:, i * num:(i + 1) * num]
        X_train = np.delete(copyX, slice(i * num, (i + 1) * num), 1);

        Y_test = copyY[:, i * num:(i + 1) * num]
        Y_train = np.delete(copyY, slice(i * num, (i + 1) * num), 1);

        # print(" X_train:", X_train, " X_test", X_test)
        # print("Y_train:", Y_train, "Y_test:", Y_test)
        for s in np.arange(0.01, 0.1, 0.01):
            SVMresult = wd.train_SVM(X_train, Y_train, C=1, sigma=s, K=0)
            cur = {SVMresult.dl[0]: s}
            res = cur if SVMresult.dl[0] < min else res;

            SVMresult.Kernel_SVM_RBF(X_test, Y_test)



        print("min s ",res.values)


    #
    # KF = KFold(n_splits=3)  # 建立4折交叉验证方法 查一下KFold函数的参数
    # bestgamma = 0;
    # minDL = sys.maxsize;
    # for train_index, test_index in KF.split(DTR_std):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = DTR_std[train_index], DTR_std[test_index]
    #     #对quality 相应也要切割
    #     Y_train, Y_test = LTR[train_index], LTR[test_index]
    #     #print(X_train, Y_train)
    #
    #     for s in np.arange(0.01,1,0.01):
    #
    #         print(s)
    #         Kernel_SVM_RBF = wd.train_SVM(X_train.T, Y_train, C=1, sigma=s, K=0)
    #         # minDL= min (minDL, curDL)
    #         print("-----------")

    # DTE, LTE = load("./Test.txt")
    # print(np.arange(0,1,0.01)) [0 0.01 0.02....]
