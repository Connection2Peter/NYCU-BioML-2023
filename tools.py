##### Import
import numpy as np
import random, hashlib, itertools



##### Functions
def GetSHA256(input_string):
    input_bytes = input_string.encode('utf-8')

    return hashlib.sha256(input_bytes).hexdigest()

def ShowDistribution(y):
    RET = {}

    for i in y:
        if i not in RET.keys():
            RET[i] = 0

        RET[i] += 1
    
    return RET

def BalancingXY(X, y):
    maxLen = len(X)
    minNum = min(ShowDistribution(y).values())

    D1X, D1y, D2X, D2y = [], [], [], []
    newX, newy = [], []

    for i in range(maxLen):
        if y[i] == 0:
            D1X.append(X[i])
            D1y.append(y[i])
        else:
            D2X.append(X[i])
            D2y.append(y[i])

    GetD1s = random.sample(range(len(D1X)), minNum)
    GetD2s = random.sample(range(len(D2X)), minNum)

    for i in GetD1s:
        newX.append(D1X[i])
        newy.append(D1y[i])
    
    for i in GetD2s:
        newX.append(D2X[i])
        newy.append(D2y[i])

    return np.array(newX),  np.array(newy)

def BalancingRatio(X, y, ratio):
    maxLen = len(X)
    minNum = min(ShowDistribution(y).values())

    D1X, D1y, D2X, D2y = [], [], [], []
    newX, newy = [], []

    for i in range(maxLen):
        if y[i] == 0:
            D1X.append(X[i])
            D1y.append(y[i])
        else:
            D2X.append(X[i])
            D2y.append(y[i])

    selNum = int(minNum*ratio)
    lenD1, lenD2 = len(D1X), len(D2X)

    if lenD1 > lenD2:
        GetD1s = random.sample(range(lenD1), selNum)
        GetD2s = random.sample(range(lenD2), minNum)
    else:
        GetD1s = random.sample(range(lenD1), minNum)
        GetD2s = random.sample(range(lenD2), selNum)

    for i in GetD1s:
        newX.append(D1X[i])
        newy.append(D1y[i])
    
    for i in GetD2s:
        newX.append(D2X[i])
        newy.append(D2y[i])

    return np.array(newX),  np.array(newy)

def Label2Possibility(y):
    RET = []

    for i in y:
        if i == 0:
            RET.append([1, 0])
        else:
            RET.append([0, 1])
    
    return np.array(RET)

def Possibility2Label(y):
    RET = []

    for i in y:
        if i[0] > i[1]:
            RET.append(0)
        else:
            RET.append(1)
    
    return np.array(RET)

def GetAllCombination(FeatureSets):
    RET = []
    numFeatureSet = len(FeatureSets)

    for i in range(1, numFeatureSet+1):
        RET += list(itertools.combinations(FeatureSets, i))
    
    return RET
