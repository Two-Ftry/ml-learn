# -*- coding: UTF-8 -*-
import numpy
import operator
import sys

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
# 读取文件
labelMap = {
    'largeDoses': 3,
    'smallDoses': 2,
    'didntLike': 1
}
def file2matrix(filename):
    # file = open(filename)
    file = getResourceAsStream(filename)
    lineList = file.readlines()
    file.close()
    numberOfLines = len(lineList)
    mat = numpy.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in lineList:
        line= line.strip()
        lineItems = line.split('\t')
        mat[index,:] = lineItems[0:3]
        classLabelVector.append(labelMap[lineItems[-1]])
        index += 1
    return mat, classLabelVector

# 归一化处理
def autoNorm(dataSet):
    maxVals = dataSet.max(0)
    minVals = dataSet.min(0)
    ranges = maxVals - minVals
    normDataSet = numpy.zeros(numpy.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - numpy.tile(minVals, (m, 1))
    normDataSet = normDataSet/numpy.tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: {0}, the real answer is: {1}'.format(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print('the total error rate is: {0}'.format(errorCount/float(numTestVecs)))



def main():
    datingClassTest()

main()
