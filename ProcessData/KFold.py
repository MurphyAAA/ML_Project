import numpy as np


class KFold:
    def __init__(self, k):
        self.k = k

    def run(self, model_class, data_provider):
        HighestAcc = 0.0
        model = None
        for i in range(self.k):
            md = model_class()
            DTR, DTE, LTR, LTE = data_provider.get(self.k, i)
            md.train(DTR, LTR)
            acc, err = md.validation(DTE, LTE)
            print("Model {}  acc: {}%".format(model_class.__NAME__, acc * 100))
            print("Model {} err rate: {}%".format(model_class.__NAME__, err * 100))
            if acc > HighestAcc:
                HighestAcc = acc
                model = md
        return HighestAcc, model
