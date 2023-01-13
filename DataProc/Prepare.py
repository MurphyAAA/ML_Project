
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

def gaussianize(D):
    # cubic root
    return np.cbrt(D)


def normalize(D):
    # return stats.zscore(D, axis=1)
    res = (D-D.mean()) / D.std()
    return res
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
    D0 = normalize(gaussianize(D[:, L == 0]))
    D1 = normalize(gaussianize(D[:, L == 1]))

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

