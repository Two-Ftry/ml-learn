import numpy

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = numpy.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances  = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def img2vector(filename):
    file = open(filename)
    mat = numpy.zeros((1, 1024))
    for i in range(32):
        line = file.readline()
        for j in range(32):
            mat[0, 32*i+j] = int(line[j])
    file.close()
    return mat

# def handleWriting():
#     return

mat = img2vector('./2-kNN/trainingDigits/0_0.txt')
# print(mat[0,0:31])