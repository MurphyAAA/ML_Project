import pdb

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.optimize
import sklearn.datasets


def vcol(v):
    return v.reshape((v.size, 1))


def vrow(v):
    return v.reshape((1, v.size))


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]
    L = L[L != 0]
    L[L == 2] = 0
    return D, L


def split_db_2tol(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)  # 2/3的数据当做训练集，1/3当做测试
    np.random.seed(seed)  # 设置一个种子

    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


# def train_SVM_linear(DTR, LTR, C, gamma, K=1):
def train_SVM_linear(DTR, LTR, C, K=1):
    DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = np.dot(DTREXT.T, DTREXT)
    ## Dist = vcol((DTR**2).sum(0)) + vrow((DTR**2).sum(0)) - 2 * np.dot(DTR.T,DTR)
    # Dist = np.zeros(DTR.shape[0],DTR.shape[0])
    # for i in range(DTR.shape[0]):
    #     for j in range(DTR.shape[1]):
    #         xi = DTR[:,i]
    #         xj = DTR[:,j]
    #         Dist[i,j] = np.linalg.norm(xi-xj)**2
    # H = np.exp(-gamma*Dist) + K
    print("---------")
    H = vcol(Z) * vrow(Z) * H


    def JDual(alpha):
        Ha = np.dot(H, vcol(alpha))
        aHa = np.dot(vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)  # 损失函数，梯度

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    def JPrimal(w):
        S = np.dot(vrow(w), DTREXT)
        loss = np.maximum(np.zeros(S.shape), 1 - Z * S).sum()
        return 0.5 * np.linalg.norm(w) ** 2 + C * loss

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        np.zeros(DTR.shape[1]),
        bounds=[(0, C)] * DTR.shape[1],
        factr=1.0,
        maxiter=100000,
        maxfun=100000)
    wStar = np.dot(DTREXT, vcol(alphaStar) * vcol(Z))  # wStar 为 (feature+K 行，1列) 的列向量
    print(JPrimal(wStar))
    print(JDual(alphaStar)[0])

    return wStar

    # print(alphaStar.shape)
    # mloss,_ = JDual(alphaStar)


    # print("gap ",JPrimal(wStar) - JDual(alphaStar)[0])


def train_SVM(DTR, LTR, C, gamma, K=1):  #非线性 使用 核函数
    # DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    # H = np.dot(DTREXT.T, DTREXT)
    Dist = np.zeros((DTR.shape[1],DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            xi = DTR[:,i]
            xj = DTR[:,j]
            Dist[i,j] = np.linalg.norm(xi-xj)**2
    kernel = np.exp(-gamma*Dist) + K**0.5
    H = vcol(Z) * vrow(Z) * kernel

    def JDual(alpha):
        Ha = np.dot(H, vcol(alpha))
        aHa = np.dot(vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)  # 损失函数，梯度

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad



    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        np.zeros(DTR.shape[1]),
        bounds=[(0, C)] * DTR.shape[1],
        factr=1.0,
        maxiter=100000,
        maxfun=100000)
    # wStar = np.dot(DTR, vcol(alphaStar) * vcol(Z))  # wStar 为 (feature+K 行，1列) 的列向量

    print('Dual loss ',JDual(alphaStar)[0])

    return alphaStar




if __name__ == '__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2tol(D, L)
    wStar = train_SVM_linear(DTR, LTR, 1, 1)
    gamma = 1
    #alphaStar = train_SVM(DTR,LTR,C=1,gamma=gamma,K=0)

    b = wStar[-1]
    wStar = wStar[0:-1]
    score = np.dot(wStar.T, DTE) + b
    score = np.dot(wStar.T, DTE)
    score = score.ravel()
    predict = np.zeros(score.size,dtype=np.int32)
    pre = []
    for idx in range(score.size):
        predict[idx] = 1 if score[idx]>0 else 0

        if predict[idx] == LTE[idx]:
            pre.append(True)
        else:
            pre.append(False)

    corr = pre.count(True)
    wrong = pre.count(False)
    acc = corr / len(pre)
    err = wrong / len(pre)
    print("acc:", acc * 100, "%")
    print("err:", err * 100, "%")

    # Dist = np.zeros((DTR.shape[1], DTE.shape[1]))
    # Z = np.zeros(LTR.shape)
    # Z[LTR == 1] = 1
    # Z[LTR == 0] = -1
    # predict = np.zeros(DTE.shape[1],dtype=np.int32)
    # for i in range(DTE.shape[1]):
    #     xi = DTE[:, i]
    #     S=0
    #     for j in range(DTR.shape[1]):
    #
    #         xj = DTR[:, j]
    #         Dist = np.linalg.norm(xi - xj) ** 2
    #         kernel = np.exp(-gamma * Dist)
    #         S += alphaStar[j] * Z[j] * kernel
    #         # print(S)
    #
    #     predict[i] = 1 if S>0 else 0
    #
    # pre = []
    # for idx in range(DTE.shape[1]):
    #
    #     if predict[idx] == LTE[idx]:
    #         pre.append(True)
    #     else:
    #         pre.append(False)
    #
    # corr = pre.count(True)
    # wrong = pre.count(False)
    # acc = corr / len(pre)
    # err = wrong / len(pre)
    # print("acc:", acc * 100, "%")
    # print("err:", err * 100, "%")