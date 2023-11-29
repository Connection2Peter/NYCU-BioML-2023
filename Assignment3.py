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
        "OneHot" : {"X": X1, "y" : y1},
        "AAC"    : {"X": X2, "y" : y2},
        "PWM"    : {"X": X3, "y" : y3},
        "PSSM"   : {"X": X4, "y" : y4},
        "BLOSUM" : {"X": X5, "y" : y5},
    }

    return DBs

def GetQ2Classifier():
    Clfs = {
        "DT"  : {"Model" : classifier.DecisionTree(), "Name" : "Decision Tree"},
        "RF"  : {"Model" : classifier.RandomForest(nTree), "Name" : "Random Forest"},
        "SVM" : {"Model" : classifier.SupportVectorMachine(), "Name" : "Support Vector Machine"},
        "XGB" : {"Model" : classifier.XGBoost(nTree), "Name" : "XGBoost"},
        "MLP" : {"Model" : classifier.MultilayerPerceptron(), "Name" : "Multilayer Perceptron"},
    }

    return Clfs



##### Main
DataSplit = dataset.SplitNfold(nSplit)
FeatureEncoder = encoder.Encode(Config.positive_data, Config.negative_data)

### Q1
print("### 1. Performance Comparison of Different Feature Encoding Methods")
print("Feature", "\t".join(["Sn", "Sp", "Acc", "MCC", "AUC"]))

RF = classifier.RandomForest(nTree)
Datas = GetQ1Feature(FeatureEncoder)

for k, Vs in Datas.items():
    X, y = Vs["X"], Vs["y"]
    Evas = []

    for trainIdx, testIdx in DataSplit.split(X, y):
        X_train, X_test = X.iloc[trainIdx], X.iloc[testIdx]
        y_train, y_test = y.iloc[trainIdx], y.iloc[testIdx]

        RF.fit(X_train, y_train.values.ravel())

        Evas.append(dataset.Evaluation(y_test, RF.predict(X_test)))

    print("{}\t{}".format(k, "\t".join(["{:.3f}".format(100*v) for v in np.mean(np.array(Evas), axis=0)])))

del(Datas, RF)

### Q2
print("### 2. Performance Comparison of Different Supervised Learning Methods")
print("Method", "\t".join(["Sn", "Sp", "Acc", "MCC", "AUC"]))

MeanROCs = []
X, y = FeatureEncoder.ToPSSM()
Clfs = GetQ2Classifier()

for k, Vs in Clfs.items():
    model = Vs["Model"]
    Evas, ROCs = [], []

    for trainIdx, testIdx in DataSplit.split(X, y):
        X_train, X_test = X.iloc[trainIdx], X.iloc[testIdx]
        y_train, y_test = y.iloc[trainIdx], y.iloc[testIdx]

        model.fit(X_train, y_train.values.ravel())

        y_pred = model.predict(X_test)
        
        ROCs.append(dataset.ROC(y_test, y_pred))
        Evas.append(dataset.Evaluation(y_test, y_pred))

    Means = np.mean(np.array(Evas), axis=0)
    Temps = np.mean(np.array(np.array(ROCs)), axis=0)
    MeanROCs.append([Vs["Name"], [Temps[0], Temps[1], round(Means[4], 8)]])

    print("{}\t{}".format(k, "\t".join(["{:.3f}".format(100*v) for v in Means])))

dataset.ROCs(MeanROCs)

print("### Done !")
