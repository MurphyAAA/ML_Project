import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def mcol(v):  # 转为列向量
    return v.reshape((v.size, 1))

def load(filename):
    DList = []
    labelsList = []
    with open(filename) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:12]
                attrs = mcol(np.array([float(i) for i in attrs] ))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
        return np.hstack(DList),np.array(labelsList,dtype=np.int32)



# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    D,L = load("./Train.txt")
    print(D)

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
