import numpy
import random
def loadDataSet(path, cols = 2):
    dataMat = []
    labelMat = []
    fr = open(path)
    for line in fr.readlines():
        lineArr = line.strip().split()
        tempArray = []
        for colIndex in range(cols):
            tempArray.append(float(lineArr[colIndex]))
        dataMat.append(tempArray)
        labelMat.append(int(lineArr[cols].split('.')[0]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj


dataArr, labelArr = loadDataSet('./data/testSet.txt')


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = numpy.mat(dataMatIn); labelMat = numpy.mat(classLabels).transpose()
    b = 0; m,n = numpy.shape(dataMatrix)
    alphas = numpy.mat(numpy.zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            #   fXi: 预测的类别
            fXi = float(numpy.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            print(dataMatrix[i,:])
            # Ei：计算误差
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            # 该数据向量可以被优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(numpy.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                # 0 < alpha < C
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print('L==H'); continue
                # eta为alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                # 
                if eta >= 0: print('eta>=0'); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print('j not moving enough'); continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print('iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print('iteration number: %d' % (iter))
    return b,alphas

b, alpha = smoSimple(dataArr, labelArr, 0.6, 0.001, 4)
print(b)
print(alpha)