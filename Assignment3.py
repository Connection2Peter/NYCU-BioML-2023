##### Import
import sys
import numpy as np
from lib import cmdline
from lib import dataset
from lib import encoder
from lib import classifier



##### Argument
nSplit, nTree = 5, 500
Config = cmdline.ArgumentParser().parse_args()
errMsg = cmdline.ArgumentCheck(Config)

if errMsg != "":
    exit(errMsg)



##### Functions
def GetQ1Feature(Encoder):
    X1, y1 = Encoder.ToOneHot()
    X2, y2 = Encoder.ToAAC()
    X3, y3 = Encoder.ToPWM()
    X4, y4 = Encoder.ToPSSM()
    X5, y5 = Encoder.ToBLOSUM62()

    DBs = {
        "OneHot" : {"X": X1, "y" : y1, "Name" : "One-hot encoding"},
        "AAC"    : {"X": X2, "y" : y2, "Name" : "Amino acid composition"},
        "PWM"    : {"X": X3, "y" : y3, "Name" : "Positional Weighted Matrix"},
        "PSSM"   : {"X": X4, "y" : y4, "Name" : "Position-specific scoring matrix"},
        "BLOSUM" : {"X": X5, "y" : y5, "Name" : "BLOSUM62"},
    }

    return DBs

def GetQ2Classifier():
    Clfs = {
        "RF" : {"Model" :  classifier.RandomForest(nTree), "Name" : "Random Forest"},
        "RF" : {"Model" :  classifier.RandomForest(nTree), "Name" : "Random Forest"},
        "RF" : {"Model" :  classifier.RandomForest(nTree), "Name" : "Random Forest"},
        "RF" : {"Model" :  classifier.RandomForest(nTree), "Name" : "Random Forest"},
        "RF" : {"Model" :  classifier.RandomForest(nTree), "Name" : "Random Forest"},
    }

    return Clfs



##### Main
DataSplit = dataset.SplitNfold(nSplit)

### Q1
Datas = GetQ1Feature(encoder.Encode(Config.positive_data, Config.negative_data))

RF = classifier.RandomForest(nTree)

print("### 1. Performance Comparison of Different Feature Encoding Methods")
print("Feature", "\t".join(["Sn", "Sp", "Acc", "MCC", "AUC"]))
for k, Vs in Datas.items():
    X, y = Vs["X"], Vs["y"]
    Evas = []

    for trainIdx, testIdx in DataSplit.split(X, y):
        X_train, X_test = X.iloc[trainIdx], X.iloc[testIdx]
        y_train, y_test = y.iloc[trainIdx], y.iloc[testIdx]

        RF.fit(X_train, y_train.values.ravel())

        Evas.append(dataset.Evaluation(y_test, RF.predict(X_test)))

    print("{}\t{}".format(k, "\t".join(["{:.3f}".format(100*v) for v in np.mean(np.array(Evas), axis=0)])))

print("### 2. Performance Comparison of Different Supervised Learning Methods")


print("### Done !")
