
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

def gaussianize(D):
    # cubic root
    res = np.cbrt(D)
    for i in range(11):
        print(">> ",res[i,:].mean(), res[i,:].std())
    return np.cbrt(D)
def gaussianize1(D):
    # 一行一个feature
    ylist=[]
    for ind in range(11):
        # rank(D[ind,:]) 第ind个feature的rank的数组
        res = stats.norm.ppf(rank(D[ind,:]),loc=0,scale=1)
        ylist.append(res)
        print(res.mean(),res.std())
    # y: 11 * 1839
    y = np.vstack(ylist)
    return y
def rank(x):
    ranks = stats.rankdata(x,method='min') # 一定要是min才是均匀分布！ 不是均匀分布没办法转回高斯，对于一个feature 共N个samples，计算每个sample这个feature的值有多少比他小的，（将N个样本的feature按从小到大排序）
    return (ranks+1) / (len(x) + 2)

def normalize(D):
    return stats.zscore(D, axis=1)

    # Z-score mean
    # sum = 0;
    # for i in range(D.shape[1]):
    #     sum += D[:, i:i+1]
    # len = float(D.shape[1])
    # mu = sum / len
    # std = sqrt(sum/(len - 1))


def corrlationAnalysis(D):
    data = {}
    for i in range(11):
        data[i] = D[i]

    df = pd.DataFrame(data)
    corr_matrix = df.corr();
    sns.heatmap(corr_matrix, cmap="YlGnBu")
    print(corr_matrix)
    plt.show()


def plot_hist(D, L):
    # D0 = normalize(gaussianize(D[:, L == 0]))
    D0 = gaussianize1(D[:, L == 0])
    # D1 = normalize(gaussianize(D[:, L == 1]))
    D1 = gaussianize1(D[:, L == 1])


    # D0 = D[:, L == 0]
    # D1 = D[:, L == 1]
    hFea = {
        0: "fixed acidity",
        1: "volatile acidity",
        2: "citric acid",
        3: "residual sugar",
        4: "chlorides",
        5: "free sulfur dioxide",
        6: "total sulfur dioxide",
        7: "density",
        8: "pH",
        9: "sulphates",
        10: "alcohol"
    }

    for ind in range(11):
        plt.figure()
        plt.xlabel(hFea[ind])

        plt.hist(D0[ind, :], bins=30, density=True, alpha=0.4, label='low quality')
        plt.hist(D1[ind, :], bins=30, density=True, alpha=0.4, label='high quality')
        plt.legend()
        plt.tight_layout()
        plt.savefig('hist_%d.pdf' % ind)
    plt.show()

