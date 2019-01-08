import numpy as np 

def classify0 (inX, dataSet, labels, k):
    # 1、计算距离
    lines = dataSet.shape[0]
    inMat = np.tile(inX, lines)
    subtractMat = inMat - dataSet
    squareMat = subtractMat ** 2
    sumMat = np.sum(squareMat, axis=1)
    distanceMat = sumMat ** 0.5
    # 2、排序距离
    
    return

# def img2vector():
#     return

# def handleWriting():
#     return
a = np.array([
    [1, 2],
    [3, 4],
    [3,4]
])

b = np.array([
    [1, 2],
    [3, 4],
    [3,1]
])

print(np.sum(a, axis=1))