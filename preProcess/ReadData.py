import numpy

# turn data into vector
def mcol(v):
    return v.reshape(v.size, 1)


def load(file):
    attrList = []
    labelList = []

    with open(file) as f:
        for line in f:
            try:
                attribute = line.split(",")[0:11]
                # change format from string into flat
                attribute = mcol(numpy.array([float(i) for i in attribute]))
                # create column vector [1,2]=>[[[1] [2]]
                attrList.append(attribute)
                label = line.split(',')[-1]
                labelList.append(label)
                # vector into matrix
            except:
                pass
        return numpy.hstack(attrList), numpy.array(labelList, dtype=numpy.int32)




#if __name__ == '__main__':
    # D, L = load('../Train.txt')
    # print(D)
    # print(L)

