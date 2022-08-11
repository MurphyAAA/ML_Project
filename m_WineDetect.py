import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import scipy.special
import scipy.optimize
from sklearn.preprocessing import StandardScaler


def vcol(v):  # 转为列向量
    return v.reshape((v.size, 1))


def vrow(v):
    return v.reshape((1, v.size))


def load(filename):
    DList = []
    labelsList = []
    with open(filename) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:11]
                attrs = vcol(np.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
        return np.hstack(DList), np.array(labelsList, dtype=np.float32)


def plot_scatter(D, L, title):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    hFea = {
        0: 'fixed acidity',
        1: 'volatile acidity',
        2: 'citric acid',
        3: 'residual sugar',
        4: 'chlorides',
        5: 'free sulfur dioxide',
        6: 'total sulfur dioxide',
        7: 'density',
        8: 'pH',
        9: 'sulphates',
        10: 'alcohol'
    }
    # for xFea in range(11):
    xFea = 0
    yFea = 1
    zFea = 4
    #     for yFea in range(11):
    #         if(xFea != yFea):
    # print(xFea,yFea)
    # for angle in np.arange(0, 360, 10):
    fig = plt.figure()

    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # ax.view_init(elev=angle, azim=350)
    # ax.view_init()

    # plt.xlabel(hFea[xFea])
    # plt.ylabel(hFea[yFea])
    # plt.scatter(D0[xFea, :], abs(D0[yFea, :]), label='low level')  # X轴：D0[dIdx1, :]  Y轴： D0[dIdx2, :]
    # plt.scatter(D1[xFea, :], abs(D1[yFea, :]), label='high level')
    #
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    plt.title(title)
    ax.set_xlabel(hFea[xFea])
    ax.set_ylabel(hFea[yFea])
    ax.set_zlabel(hFea[zFea])

    ax.scatter(D0[xFea, :], abs(D0[yFea, :]), D0[zFea, :], label='low level')
    ax.scatter(D1[xFea, :], abs(D1[yFea, :]), D1[zFea, :], label='high level')

    # plt.savefig('scatter_%d_%d.pdf' )
    plt.show()


# 高斯概率密度函数
def logpdf_GAU_ND(x, mu, C):  # x: 未去中心化的原数据
    M = x.shape[0]  # 特征数
    a = M * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(C)  # log|C|
    xc = (x - mu)
    c = np.dot(xc.T, np.linalg.inv(C))
    c = np.dot(c, xc)
    c = np.diagonal(c)
    return (-1.0 / 2.0) * (a + b + c)


def LDA(D, L, nF, nC=2):
    N = D.shape[1]
    mu = D.mean(1)
    mu = vcol(mu)

    SWc = np.zeros((nF, nF))
    SB = np.zeros((nF, nF))
    for i in range(nC):
        DC = D[:, L == i]
        nc = DC.shape[1]
        muc = DC.mean(1)
        muc = vcol(muc)
        DCc = DC - muc
        C = np.dot(DCc, DCc.T)
        SWc += C

        M = muc - mu
        M = np.dot(M, M.T)
        M *= nc
        SB += M

    SW = SWc / N
    SB /= N

    s, U = scipy.linalg.eigh(SB, SW)
    m = 2
    W = U[:, ::-1][:, 0:m]
    Dp = np.dot(W.T, D)  # 返回投影后的数据集
    ## plot_scatter(DP,LTR,'LDA')
    return Dp


def logreg_obj_warp(DTR, LTR, l):
    def logreg_obj(v):
        w = v[0:-1]
        b = v[-1]
        w_norm = np.linalg.norm(w)
        w = vcol(w)
        reg_term = (l / 2) * (w_norm ** 2)
        negz = -1 * (2 * LTR - 1)
        fx = np.dot(w.T, DTR) + b
        logJ = np.logaddexp(0, negz * fx)
        mean_logJ = logJ.mean()
        res = reg_term + mean_logJ
        res = res.reshape(res.size, )
        return res

    return logreg_obj


def train_SVM(DTR, LTR, C, sigma=1, K=1,):  # 非线性 使用 核函数
    #
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    global dl
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

        return np.exp(-dist / (2 * sigma ** 2))+ K ** 0.5



    # Dist = np.zeros((DTR.shape[1], DTR.shape[1]))
    # for i in range(DTR.shape[1]):
    #     for j in range(DTR.shape[1]):
    #         xi = DTR[:, i]
    #         xj = DTR[:, j]
    #         Dist[i, j] = np.linalg.norm(xi - xj) ** 2
    # # kernel = np.exp(-0.5 * Dist) + K ** 0.5
    # kernel = np.exp(- Dist / (2 * sigma ** 2)) + K ** 0.5

    kernel = gaussian_kernel_3(DTR,DTR,sigma)
    # print("kernel",kernel)
    # print("kernel - kernel2",kernel - kernel2)
    H = vcol(Z) * vrow(Z) * kernel


    def JDual(alpha):  # 对偶
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

    print('Dual loss ', JDual(alphaStar)[0])
    #JDual(alphaStar)[0];




    def Kernel_SVM_RBF(DTE, LTE):

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

        #优化
        for i in range(DTE.shape[1]):
            xi = DTE[:,i]
            kernel = gaussian_kernel_2(DTR.T,xi.ravel(),sigma)
            S = (vrow(alphaStar) * vrow(Z) * kernel).sum()
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
        err = wrong / len(pre)
        print("acc:", acc * 100, "%")
        print("err:", err * 100, "%")
        print('TN: ', TN, 'FP: ', FP)
        print('FN: ', FN, 'TP: ', TP)
        print('total: ', (TN + FP + FN + TP))

    return Kernel_SVM_RBF


if __name__ == '__main__':
    DTR, LTR = load("./Train.txt")
    DTE, LTE = load("./Test.txt")

    sc = StandardScaler()
    DTR_std = sc.fit_transform(DTR)  # 给feature归一化
    DTE_std = sc.fit_transform(DTE)

    # plot_scatter(DTR, LTR, '原数据')
    # print(D)

    # DTE0 = DTE[:,LTE == 0]
    # DTE1 = DTE[:,LTE == 1]
    #
    # TR_NC0 = DTR0.shape[1]
    # TR_NC1 = DTR1.shape[1]

    ## PCA ##
    # mu = vcol(DTR.mean(1))
    # DTRc = DTR - mu
    # C = np.dot(DTRc,DTRc.T) / LTR.size
    # U,s,Vh = np.linalg.svd(C)
    # m = 6
    # P = U[:,0:m]
    # DP = np.dot(P.T,DTR)
    # plot_scatter(DP,LTR)

    ## LDA ##
    # DTRp = LDA(DTR,LTR,11,2) #降维，降维前11个特征，2类
    # DTEp = LDA(DTE,LTE,11,2)

    ## MVG ##
    # DTR0 = DTRp[:, LTR == 0]
    # DTR1 = DTRp[:, LTR == 1]
    #
    # mu0 = DTR0.mean(1)
    # mu0 = vcol(mu0)
    #
    # mu1 = DTR1.mean(1)
    # mu1 = vcol(mu1)
    #
    # DTRC0 = DTR0 - mu0
    # DTRC1 = DTR1 - mu1
    #
    # C0 = np.dot(DTRC0, DTRC0.T) / DTRC0.shape[1]
    # C1 = np.dot(DTRC1, DTRC1.T) / DTRC1.shape[1]
    #
    # # identity = np.identity(DTR.shape[0])  # 与 高斯密度分布的区别，就这里认为特征之间相互无关所以除了对角线都是0
    # # C0 = C0 * identity
    # # C1 = C1 * identity
    #
    # tlogll0 = logpdf_GAU_ND(DTEp,mu0,C0)
    # tlogll1 = logpdf_GAU_ND(DTEp,mu1,C1)
    #
    # molecular0 = tlogll0+np.log(2/3)
    # molecular1 = tlogll1+np.log(1/3)
    # logS = np.vstack((molecular0,molecular1))
    # logSMarginal = vrow(scipy.special.logsumexp(logS,axis=0))
    # logSPost = logS - logSMarginal
    # Spost = np.exp(logSPost)
    #
    # predictLabel = np.argmax(Spost,axis=0)
    # res = []
    # for i in range(predictLabel.size):
    #     if (predictLabel[i] == LTE[i]):
    #         res.append(True)
    #     else:
    #         res.append(False)
    # corr = res.count(True)
    # wrong = res.count(False)
    # acc = corr / len(res)
    # err = wrong / len(res)
    # print(res)
    # print("acc:", acc * 100, "%")
    # print("err:", err * 100, "%")

    ## Tied MVG ##
    # DTR0 = DTR[:, LTR == 0]
    # DTR1 = DTR[:, LTR == 1]
    #
    # mu0 = DTR0.mean(1)
    # mu0 = vcol(mu0)
    #
    # mu1 = DTR1.mean(1)
    # mu1 = vcol(mu1)
    #
    # DTRC0 = DTR0 - mu0
    # DTRC1 = DTR1 - mu1
    #
    # C = (np.dot(DTRC0,DTRC0.T) + np.dot(DTRC1,DTRC1.T)) / DTR.shape[1]
    # # identity = np.identity(DTR.shape[0])  # 与 高斯密度分布的区别，就这里认为特征之间相互无关所以除了对角线都是0
    # # C = C * identity
    #
    #
    # tlogll0 = logpdf_GAU_ND(DTE,mu0,C)
    # tlogll1 = logpdf_GAU_ND(DTE,mu1,C)
    #
    # molecular0 = tlogll0+np.log(2/3)
    # molecular1 = tlogll1+np.log(1/3)
    # logS = np.vstack((molecular0,molecular1))
    # logSMarginal = vrow(scipy.special.logsumexp(logS,axis=0))
    # logSPost = logS - logSMarginal
    # Spost = np.exp(logSPost)
    #
    # predictLabel = np.argmax(Spost,axis=0)
    # res = []
    # for i in range(predictLabel.size):
    #     if (predictLabel[i] == LTE[i]):
    #         res.append(True)
    #     else:
    #         res.append(False)
    # corr = res.count(True)
    # wrong = res.count(False)
    # acc = corr / len(res)
    # err = wrong / len(res)
    # print(res)
    # print("acc:", acc * 100, "%")
    # print("err:", err * 100, "%")

    ## BLR ##
    # l = 1e-6
    # logreg_obj = logreg_obj_warp(DTR,LTR,l)
    # x,f,d = scipy.optimize.fmin_l_bfgs_b(logreg_obj,np.zeros(DTR.shape[0] + 1),approx_grad=True)
    # w = x[0:-1] #训练好的w
    # b = x[-1]   #训练好的b
    #
    # w = vrow(w)
    # s = np.dot(w,DTE)+b
    # s = s.reshape(s.size,)
    #
    # predict = []
    # TT=0 #应该是T 预测为T
    # TF=0 #应该是T 预测为F
    # FT=0 #应该是F 预测为T
    # FF=0 #应该是F 预测为F
    # for i in s:
    #     if i > 0:
    #         predict.append(1)  # 预测为1
    #     else:
    #         predict.append(0)  # 预测为0
    # # print(res)
    # res = []
    # for i in range(len(predict)):
    #     # print(i)
    #     if predict[i] == LTE[i]:
    #         res.append(True)  # 预测正确的
    #         if predict[i] == 1:
    #             TT += 1
    #         else:
    #             FF += 1
    #     else:
    #         res.append(False)  # 预测错误的
    #         if predict[i] == 1:
    #             FT += 1
    #         else:
    #             TF += 1
    # corNum = res.count(True)
    # errNum = res.count(False)
    # err_rate = errNum / len(res)
    # print("l:", l, " Error Rate:", err_rate)
    # print('TT: ',TT,'TF: ',TF)
    # print('FT: ',FT,'FF: ',FF)
    # print('total: ',(TT+TF+FT+FF))

    # SVM

    # for i in range(DTR_std.shape[1]):
    # i=0
        # DTR_std_new = np.delete(DTR_std.copy(),i,axis=1)
        # LTR_new = np.delete(LTR.copy(),i,axis=0)
        # DTE_new = DTR_std[:,i:i+1].copy()
        # LTE_new = np.int32(LTR[i].copy())

    Kernel_SVM_RBF = train_SVM(DTR_std, LTR, C=1, sigma=0.39, K=0)
    Kernel_SVM_RBF(DTE_std, LTE)
        # Kernel_SVM_RBF = train_SVM(DTR_std_new, LTR_new, C=1, sigma=1, K=1)
        # print("start predict ",i)
        # res.append(Kernel_SVM_RBF(DTE_new,LTE_new))

    # print("result: ")
    # print(res)

