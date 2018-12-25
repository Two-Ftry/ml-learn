# -*- coding: UTF-8 -*-
from numpy import *
import operator
import sys

def createDataSet():
    group = array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    row = dataSet.shape[0]
    inXs = tile(inX, (row, 1))
    diffMat = inXs - dataSet
    sqDiffMat = diffMat**2
    sumDiffMat = sqDiffMat.sum(axis=1)
    distances = sumDiffMat**0.5
    # print(distances)
    sortedDistIndicies = distances.argsort()
    # print(sortedDistIndicies)
    classifyCount = {}
    for i in range(k):
        label = labels[sortedDistIndicies[i]]
        classifyCount[label] = classifyCount.get(label, 0) + 1
    sortedClassCount = sorted(classifyCount.iteritems(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# print('result', r1)

def main(inputValue, k=3):
    ds = createDataSet()
    result = classify0([inputValue], ds[0], ds[1], k)
    print('result', result)

len = len(sys.argv)
if len < 3:
    print('parameters must larger than 3')
else:
    x = sys.argv[1]
    y = sys.argv[2]
    k = 3
    if len >= 4:
        k = sys.argv[3]
    if x and y:
        main([float(x), float(y)], int(k))
    else:
        print('parameters is wrong')