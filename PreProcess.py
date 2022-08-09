# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 19:08:49 2022

@author: 吴佳琦
"""

import pandas as pd
import numpy as np
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
        return np.hstack(DList), np.array(labelsList, dtype=np.int32)

if __name__ == "__main__":
    # header = ["a","b","c","d","e","f","g","h","i","j","k","l"]
    # data_import=pd.read_csv("./Train.txt",names=header);
    # data_import=pd.read_csv("./Train.txt");

    x,y = data_import = load('./Train.txt')

    # x = data_import[:-1]
    # y = data_import[-1]


    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x) #给feature归一化
    # ytest = data_import["l"].values.reshape(-1,1)
    # ytest = vcol(y)
    # ans = np.concatenate((x_train_scaled , ytest), axis=1)

    # np.savetxt('stdData', ans,delimiter=',')
