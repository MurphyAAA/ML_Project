import numpy as np
from DataProc import DataProvider
from Models.Model import MLModel


class KFold:
    def __init__(self, k):
        self.k = k

    def dcf(self, model: MLModel, data_provider: DataProvider):
        dcf = 0
        score = []
        label = []
        for i in range(self.k):
            DTR, DTE, LTR, LTE = data_provider.get(self.k, i)
            model.train(DTR, LTR)
            dcf += model.actDcf(DTE, LTE, data_provider.pi)
            score.append(model.score(DTE))
            label.append(LTE)
        minDcf = model.minDcfByScore(np.array(score), np.array(label), data_provider.pi)
        return dcf, minDcf

    def plotDcf(self, model: MLModel, data_provider: DataProvider):
        # p: -4, 4
        # calc pi
        #
        # threshold = np.linspace(-3, 3, 21)
        # DCF = []
        # for t in threshold:
        #     MVG = Models.MVG()
        #     dcf = MVG.validation_actdcf(DTE, LTE, t)
        #     DCF.append(dcf)
        #
        # plt.plot(threshold, DCF, label='DCF', color='r')

        pass

def main():
    from DataProc import load
    from DataProc.Prepare import gaussianize
    from DataProc.Prepare import normalize
    from DataProc import DataProvider
    # load data from given path
    D, L = load('../Train.txt')
    # preprocessing the dataset
    normalize(gaussianize(D))
    dp = DataProvider(D, L)
    from Models import MVG
    model = MVG()
    kf = KFold(5)
    dcf, minDcf = kf.dcf(model, dp)
    print("DCF=%f, minDcf=%f"%(dcf, minDcf))


if __name__ == '__main__':
    main()