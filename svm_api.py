import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import scipy.special
import scipy.optimize
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC




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


if __name__ == '__main__':
    DTR, LTR = load("./Train.txt")
    DTE, LTE = load("./Test.txt")
    DTR = DTR.T
    DTE = DTE.T
    sc = StandardScaler()
    DTR_std = sc.fit_transform(DTR)  # 给feature归一化

    DTE_std = sc.transform(DTE)

    #SVM
    # Kernel_SVM_RBF = train_SVM(DTR_std,LTR,C = 1,gamma=1,K=0)
    # Kernel_SVM_RBF(DTE,LTE)
    svc_clf = SVC(C=1.0,
              kernel='rbf',
              degree=3,
              gamma='auto',
              coef0=0.0, shrinking=True,
              probability=False,
              tol=0.001, cache_size=200,
              class_weight=None,
              verbose=False, max_iter=-1,
              decision_function_shape='ovr',
              break_ties=False,random_state=0)
    svc_clf.fit(DTR_std, LTR)
    svc_clf_predictions = svc_clf.predict(DTE_std)
    c = confusion_matrix(LTE, svc_clf_predictions)
    a = accuracy_score(LTE, svc_clf_predictions)
    p = precision_score(LTE, svc_clf_predictions)
    r = recall_score(LTE, svc_clf_predictions)

#TP,FP
#FN,TN
    print('Confusion Matrix:\n',c)
    print('Accuracy:', a * 100) # TP+TN / TP+TN+FP+FN
    print('Precision:', p * 100)
    print('Recall:', r * 100)