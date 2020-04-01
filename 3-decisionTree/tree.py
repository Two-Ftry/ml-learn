from math import log

# 计算熵
def caclShannonEnt(dataSet):
    labels = {}
    for data in dataSet:
        currentLabel = data[-1]
        if currentLabel not in labels.keys():
            labels[currentLabel] = 0
        labels[currentLabel] += 1
    shannonEnt = 0.0
    numEntries = len(dataSet)
    for key in labels:
        prob = labels[key]/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for data in dataSet:
        if (data[axis] == value):
            reduceFeatVec = data[:axis]
            reduceFeatVec.extend(data[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet) - 1
    baseEntropy = caclShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        uniqueFeatures = set(featureList) 
        # newEntropy某一列的熵
        newEntropy = 0.0
        for value in uniqueFeatures:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * caclShannonEnt(subDataSet)
        # 判断这个特征的熵是否更好
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature   
        

    return bestFeature

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

dataMat, labels = createDataSet()
# result = caclShannonEnt(dataMat)
# print('%f' % (result))
r = splitDataSet(dataMat, 0, 1)
print(r)