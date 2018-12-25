from numpy import *
import operator

def createDataSet ():
    group = array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1],
    ])
    lables = ['A', 'A', 'B', 'B']
    return group, lables

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # print('1', tile(inX, (dataSetSize, 1)))
    # print('2', diffMat)
    sqDiffMat = diffMat**2
    # print('3', sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    # print('4', sqDistances)
    distances  = sqDistances**0.5
    # print('5', distances)
    sortedDistIndicies = distances.argsort()
    # print('6', sortedDistIndicies)
    classCount={}
    for i in range(k):
        # print('6-0', i, sortedDistIndicies[i])
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # print('6-1', voteIlabel, classCount[voteIlabel])
    # print('7', classCount)
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)
    # print('8', sortedClassCount)
    return sortedClassCount[0][0]

ds = createDataSet()

r1 = classify0([[1.0, 0.8]], ds[0], ds[1], 3)
r2 = classify0([[0.1, 0.1]], ds[0], ds[1], 3)

print('9', r1)
print('9', r2)
