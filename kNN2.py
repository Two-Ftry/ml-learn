# -*- coding: UTF-8 -*-
import numpy
# 读取文件
labelMap = {
    'largeDoses': 3,
    'smallDoses': 2,
    'didntLike': 1
}
def file2matrix(filename):
    file = open(filename)
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

datingDataMat, datingLabels = file2matrix('./datingTestSet.txt')

# 归一化处理
def Norm(dataSet):
    maxValue = dataSet.max(0)
    minValue = dataSet.min(0)
    print(maxValue, minValue)

Norm(datingDataMat)

# 分析数据：使用matplotlib创建散点图
'''
import matplotlib
import matplotlib.pyplot as plt 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0 * numpy.array(datingLabels), 15.0 * numpy.array(datingLabels))
plt.show()
'''