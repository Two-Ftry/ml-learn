import math
import operator

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


#  计算熵
def caclEntropy(dataSet):
    totalCount = len(dataSet)
    labelClass = {}
    for item in dataSet:
        label = item[-1]
        if label not in labelClass:
            labelClass[label] = 0
        labelClass[label] += 1

    entropy = 0.0
    for key in labelClass:
        count = labelClass[key]
        prob = count / totalCount
        entropy -= prob * math.log(prob,2)
    return entropy


# caclEntropy(dataMat)

# 划分数据集，选择最好的划分方式
def splitDataSet(dataSet, axis, value):
    newDataSet = []
    for item in dataSet:
        if item[axis] == value:
            newItem = item[:axis]
            newItem.extend(item[axis:])
            newDataSet.append(newItem)
    return newDataSet

# splitDataSet(dataMat, 0, 0)

def getColValueList(dataSet, colIndex):
    values = []
    for item in dataSet:
        values.append(item[colIndex])
    return values

def chooseBestFeatureToSplit(dataSet):
    featureCount = len(dataSet[0]) - 1

    baseEntropy = caclEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for featureIndex in range(featureCount):
        valueList = getColValueList(dataSet, featureIndex)
        valueSet = set(valueList)
        newEntropy = 0.0
        for valueItem in valueSet:
            subDataSet = splitDataSet(dataSet, featureIndex, valueItem)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * caclEntropy(subDataSet)
        
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = featureIndex
    return bestFeature

# chooseBestFeatureToSplit(dataMat)


# 投票表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

# 构建决策树
def createTree(dataSet, labels):
    tree = {}
    bestFeature = chooseBestFeatureToSplit(dataSet)
    tree[labels[bestFeature]] = {}

    # 填充value key
    values = getColValueList(dataSet, bestFeature)
    valueSet = set(values)
    for valueItem in valueSet:
        subDataSet = splitDataSet(dataSet, bestFeature, valueItem)
        labelList = getColValueList(subDataSet, len(subDataSet[0]) - 1)
        labelSet = set(labelList)
        if len(labelSet) == 1:
            tree[labels[bestFeature]][valueItem] = labelList[0]
        elif len(subDataSet) == 1:
            tree[labels[bestFeature]][valueItem] = majorityCnt(labelList)
        else:
            newLabels = labels[:bestFeature]
            newLabels.extend(labels[bestFeature + 1:])
            newDataSet = []
            for item in subDataSet:
                tempItem = item[:bestFeature]
                tempItem.extend(item[bestFeature+1:])
                newDataSet.append(tempItem)
            subTree = createTree(newDataSet, newLabels)
            tree[labels[bestFeature]][valueItem] = subTree

    return tree

resultTree = createTree(dataMat, labels)

# print(resultTree)

# 使用决策树进行分类
def classify(inputTree, featLabels, testVec):
    classifyLabel = list(inputTree.keys())[0]
    index = featLabels.index(classifyLabel)
    value = testVec[index]
    result = ''
    subTree = inputTree[classifyLabel][value]
    if type(subTree).__name__ == 'dict':
        result = classify(subTree, featLabels, testVec)
    else:
        result = subTree
    return result


# testResult = classify(resultTree, labels, [1, 1])
# print('@@@@@', testResult)

# 决策树的存储
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, mode='wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

storeTree(resultTree, 'classifierStorage.txt')

print(grabTree('classifierStorage.txt'))