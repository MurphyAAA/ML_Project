import sys

import m_WineDetect as wd;
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import scipy.special
import scipy.optimize
import pdb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np


def train_SVM(DTR, LTR, C, sigma=1, K=1):  # 非线性 使用 核函数
    #
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    def gaussian_kernel_3(X, Y, sigma=sigma):
        '''
        输入
            X       每一行是一个feature，每一列是一个sample，一个sample有11个特征
            Y       每一行是一个feature，每一列是一个sample，一个sample有11个特征
            sigma   浮点数, 高斯核的超参数
        输出
            K
        '''
        # X = np.array(X)
        # Y = np.array(Y) xi,yi是列项量
        dist = np.sum(X * X, axis=0, keepdims=True) \
               + np.sum(Y * Y, axis=0, keepdims=True) \
               - 2 * np.dot(X.T, Y)

        # print("Dist - dist ", Dist-dist)

        return np.exp(-dist / (2 * sigma ** 2)) + K ** 0.5

    # Dist = np.zeros((DTR.shape[1], DTR.shape[1]))
    # for i in range(DTR.shape[1]):
    #     for j in range(DTR.shape[1]):
    #         xi = DTR[:, i]
    #         xj = DTR[:, j]
    #         Dist[i, j] = np.linalg.norm(xi - xj) ** 2
    # # kernel = np.exp(-0.5 * Dist) + K ** 0.5
    # kernel = np.exp(- Dist / (2 * sigma ** 2)) + K ** 0.5

    kernel = gaussian_kernel_3(DTR, DTR, sigma)
    # print("kernel",kernel)
    # print("kernel - kernel2",kernel - kernel2)
    H = wd.vcol(Z) * wd.vrow(Z) * kernel

    def JDual(alpha):  # 对偶
        Ha = np.dot(H, wd.vcol(alpha))
        aHa = np.dot(wd.vrow(alpha), Ha)
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

    # print('Dual loss ', JDual(alphaStar)[0])
    dualLoss = JDual(alphaStar)[0];

    def Kernel_SVM_RBF(DTE, LTE):
        minErr = sys.maxsize;
        def gaussian_kernel_2(X, y, sigma=sigma):
            '''
            输入
                X       二维浮点数组, 第一维长度num是样本数量, 第二维长度dim是特征维数
                y       一维浮点数组, 长度dim是特征维数
                sigma   浮点数, 高斯核的超参数
            输出
                K       一维浮点数组, 长度为dim, 其中第i个表示kernel(X[i], y)

            '''
            X = np.array(X)
            y = np.array(y).reshape(-1)
            D = X - y
            return np.exp(-np.sum(D * D, axis=1) / (2 * sigma ** 2)) + K ** 0.5

        predict = np.zeros(DTE.shape[1], dtype=np.int32)
        # for i in range(DTE.shape[1]):
        #     xi = DTE[:, i]
        #     S = 0
        #     for j in range(DTR.shape[1]):
        #         xj = DTR[:, j]
        #         Dist = np.linalg.norm(xi - xj) ** 2
        #         kernel = np.exp(- Dist/ (2 * sigma ** 2)) + K ** 0.5
        #         # kernel = np.exp(- 0.5 * Dist) + K ** 0.5
        #         S += alphaStar[j] * Z[j] * kernel
        #         # print(S)
        #
        #     predict[i] = 1 if S > 0 else 0

        # 优化
        for i in range(DTE.shape[1]):
            xi = DTE[:, i]
            kernel = gaussian_kernel_2(DTR.T, xi.ravel(), sigma)
            S = (wd.vrow(alphaStar) * wd.vrow(Z) * kernel).sum()
            predict[i] = 1 if S > 0 else 0

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        pre = []

        for idx in range(DTE.shape[1]):

            if predict[idx] == LTE[idx]:
                # return True
                pre.append(True)
                if LTE[idx] == 0:
                    TN += 1
                else:
                    TP += 1
            else:
                # return False
                pre.append(False)
                if LTE[idx] == 0:
                    FN += 1
                else:
                    FP += 1

        corr = pre.count(True)
        wrong = pre.count(False)
        acc = corr / len(pre)
        curErr = wrong / len(pre)
        # minErr = curErr if curErr<minErr else minErr

        # print("acc:", acc * 100, "%")
        print("err:", curErr * 100, "%")
        # print('TN: ', TN, 'FP: ', FP)
        # print('FN: ', FN, 'TP: ', TP)
        # print('total: ', (TN + FP + FN + TP))

    return Kernel_SVM_RBF, dualLoss


def getData(DTR_std, LTR, i, N, kfold):
    num = int(N / kfold)
    copyX = DTR_std.copy();
    copyY = LTR.copy().reshape(1, -1);
    X_test = copyX[:, i * num:(i + 1) * num]
    X_train = np.delete(copyX, slice(i * num, (i + 1) * num), 1);

    Y_test = copyY[:, i * num:(i + 1) * num]
    Y_train = np.delete(copyY, slice(i * num, (i + 1) * num), 1);

    return X_train, Y_train, X_test, Y_test


def cross_validatin(DTR_std, LTR, kfold, c, sigma, K=1):
    best_loss = sys.maxsize
    N = DTR_std.shape[1]
    for ik in range(kfold):
        training_set_X, training_set_Y, test_set_X, test_set_Y = getData(DTR_std, LTR, ik, N, kfold)
        TestFunc, dualLoss = train_SVM(training_set_X, training_set_Y, C=c, sigma=sigma, K=1)

        TestFunc(test_set_X, test_set_Y.ravel())

        # loss = validataion(alpha, validation_set)
        best_loss = dualLoss if dualLoss < best_loss else best_loss

    return best_loss


def parameter_exploration(DTR_std, LTR, kfold=5, K=1):
    for sigma in np.arange(0.01, 0.1, 0.01):
        ##pdb.set_trace()
        for c in range(1, 5):
            print("------")
            print("C: ", c, "sigma: ", sigma)
            best_loss  = cross_validatin(DTR_std, LTR, kfold, c, sigma, K)
            print("minLoss: ", best_loss)
            print("------")


if __name__ == '__main__':
    DTR, LTR = wd.load("./Train.txt")
    DTE, LTE = wd.load("./Test.txt")

    sc = StandardScaler()
    DTR_std = sc.fit_transform(DTR)  # 给feature归一化

    parameter_exploration(DTR_std, LTR, kfold=5, K=1)
