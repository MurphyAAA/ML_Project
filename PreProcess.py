# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 19:08:49 2022

@author: 吴佳琦
"""

import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
  # creating instance of StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score


header = ["a","b","c","d","e","f","g","h","i","j","k","l"]
data_import=pd.read_csv("E:\POLITO_CS\MLPR\Projects\Train.txt",names=header);

x = data_import.drop("l",axis = 1).values
y = data_import["l"].values.reshape(-1,1)


sc = StandardScaler()
x_train_scaled = sc.fit_transform(x)
ytest = data_import["l"].values.reshape(-1,1)
ans = np.concatenate((x_train_scaled , ytest), axis=1)
np.savetxt('stdData', ans,delimiter=',')
