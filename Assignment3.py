##### Import
import sys
import warnings
import numpy as np
from lib import cmdline
from lib import dataset
from lib import encoder
from lib import classifier
from lib import rocplot
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


##### Argument
nSplit, nTree = 5, 500
Config = cmdline.ArgumentParser().parse_args()
errMsg = cmdline.ArgumentCheck(Config)

if errMsg != "":
    exit(errMsg)



##### Functions
# def GetQ1Feature(Encoder):
#     X1, y1 = Encoder.ToOneHot()
#     X2, y2 = Encoder.ToAAC()
#     X3, y3 = Encoder.ToPWM()
#     X4, y4 = Encoder.ToPSSM()
#     X5, y5 = Encoder.ToBLOSUM62()
#     X6, y6 = Encoder.ToEAAC()
#     X7, y7 = Encoder.ToCKSAAP()
#     X8, y8 = Encoder.ToDPC()
#     X9, y9 = Encoder.ToDDE()

#     DBs = {
#         "OneHot" : {"X": X1, "y" : y1},
#         "AAC"    : {"X": X2, "y" : y2},
#         "PWM"    : {"X": X3, "y" : y3},
#         "PSSM"   : {"X": X4, "y" : y4},
#         "BLOSUM" : {"X": X5, "y" : y5},
#         "EAAC"   : {"X": X6, "y" : y6},
#         "CKSAAP" : {"X": X7, "y" : y7},
#         "DPC"    : {"X": X8, "y" : y8},
#         "DDE"    : {"X": X9, "y" : y9},
#     }

#     return DBs

def GetQ2Classifier():
    Clfs = {
       # "DT"  : {"Model" : classifier.DecisionTree(), "Name" : "Decision Tree"},
       # "RF"  : {"Model" : classifier.RandomForest(nTree), "Name" : "Random Forest"},
       # "SVM" : {"Model" : classifier.SupportVectorMachine(), "Name" : "Support Vector Machine"},
       # "XGB" : {"Model" : classifier.XGBoost(nTree), "Name" : "XGBoost"},
       # "MLP" : {"Model" : classifier.MultilayerPerceptron(), "Name" : "Multilayer Perceptron"},
       "VC"  : {"Model" : classifier.VoteClassifier(nTree, [
        ('rf1', classifier.RandomForest(nTree)),
        ('rf2', classifier.RandomForest(nTree, criterion = "entropy")),
        ('rf3', classifier.RandomForest(nTree, criterion = "log_loss")),
        ('svm', classifier.SupportVectorMachine()),
        ('cb', classifier.CatBoost(10000)),
        ('gb', classifier.GradientBoosting()),
        ]), "Name" : "Ensemble Learning"},
       # "Ada" : {"Model" : classifier.AdaBoost(), "Name" : "AdaBoost"},
       # "GB"  : {"Model" : classifier.GradientBoosting(), "Name" : "Gradient Boosting"},
       # "ET"  : {"Model" : classifier.ExtraTrees(), "Name" : "Extra Trees"},
       # "GNB" : {"Model" : classifier.GaussianNaiveBayes(), "Name" : "Gaussian Naive Bayes"},
       # "KNN" : {"Model" : classifier.KNeighbors(), "Name" : "K Nearest Neighbors"},
       # "CB"  : {"Model" : classifier.CatBoost(10000), "Name" : "CatBoost"},
    }

    return Clfs



##### Main
DataSplit = dataset.SplitNfold(nSplit)
FeatureEncoder = encoder.Encode(Config.positive_data, Config.negative_data)


## Q1
'''
RF = classifier.RandomForest(nTree)
Datas = GetQ1Feature(FeatureEncoder)

print("### 1. Performance Comparison of Different Feature Encoding Methods")
print("Feature", "\t".join(["Sn", "Sp", "Acc", "MCC", "AUC"]))

RocPlot = rocplot.Roc_curve()

for k, Vs in Datas.items():
    X, y = Vs["X"], Vs["y"]
    Evas = []

    for trainIdx, testIdx in DataSplit.split(X, y):
        X_train, X_test = X.iloc[trainIdx], X.iloc[testIdx]
        y_train, y_test = y.iloc[trainIdx], y.iloc[testIdx]

        RF.fit(X_train, y_train.values.ravel())

        Evas.append(dataset.Evaluation(y_test, RF.predict(X_test)))
        RocPlot.append(y_test, RF.predict_proba(X_test)[:, 1])

    RocPlot.add("RF")

    print("{}\t{}".format(k, "\t".join(["{:.3f}".format(100*v) for v in np.mean(np.array(Evas), axis=0)])))

RocPlot.plot()
del(Datas, RF)
'''

### Q2

RocPlot = rocplot.Roc_curve()

X, y = FeatureEncoder.ToEAAC()
Clfs = GetQ2Classifier()

for k, Vs in Clfs.items():
    model = Vs["Model"]
    Evas, ROCs = [], []

    for trainIdx, testIdx in DataSplit.split(X, y):
        X_train, X_test = X.iloc[trainIdx], X.iloc[testIdx]
        y_train, y_test = y.iloc[trainIdx], y.iloc[testIdx]

        model.fit(X_train, y_train.values.ravel())

        y_pred = model.predict(X_test)
        RocPlot.append(y_test, model.predict_proba(X_test)[:, 1])
        Evas.append(dataset.Evaluation(y_test, y_pred))

    RocPlot.add(Vs["Name"])
    Means = np.mean(np.array(Evas), axis=0)
    Temps = np.mean(np.array(np.array(ROCs)), axis=0)

    print("{}\t{}".format(k, "\t".join(["{:.3f}".format(100*v) for v in Means])))

RocPlot.plot("output.png")

print("### Done !")
