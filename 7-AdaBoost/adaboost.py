import numpy

def loadSimpData():
    datMat = numpy.matrix([[1., 2.1], 
        [2. , 1.1],
        [1.3 , 1.],
        [1. , 1.],
        [2. , 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

datMat, classLabels = loadSimpData()

def stumpClassify (dataMatrix, dimen, threshVal, threshIneq):
    retArray = numpy.ones((numpy.shape(dataMatrix)[0], 1))
    # print(dataMatrix[:, dimen] <= threshVal)
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    # print(retArray)
    return retArray

# 单层决策器
def buildStump(dataArr, classLabels, D):
    dataMatrix = numpy.mat(dataArr)
    labelMat = numpy.mat(classLabels).T
    m, n = numpy.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = numpy.mat(numpy.zeros((m, 1)))
    minError = numpy.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = numpy.mat(numpy.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # AdaBoost和分类器交互的地方
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

D = numpy.mat(numpy.ones((5, 1))/5)
bestStump, minError, bestClasEst = buildStump(datMat, classLabels, D)

def adaBoostTrainDS (dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m  = numpy.shape(dataArr)[0]
    D = numpy.mat(numpy.ones((m, 1))/m)
    aggClassEst = numpy.mat(numpy.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print('D: %s' % (D.T))
        alpha = float(0.5*numpy.log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst:%s', classEst.T)
        expon = numpy.multiply(-1*alpha*numpy.mat(classLabels).T, classEst)
        D = numpy.multiply(D, numpy.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print('aggClassEst' % (aggClassEst.T))
        aggErrors = numpy.multiply(numpy.sign(aggClassEst) != numpy.mat(classLabels).T, numpy.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        if errorRate == 0.0:
            break
    return weakClassArr

weakClassArr = adaBoostTrainDS(datMat, classLabels)

def adaClassify(datToClass, classifierArr):
    dataMatrix = numpy.mat(datToClass)
    m = numpy.shape(dataMatrix)[0]
    aggClassEst = numpy.mat(numpy.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return numpy.sign(aggClassEst)

print('-----------------------------')

print(adaClassify(datMat, weakClassArr))