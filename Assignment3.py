##### Import
import sys
import warnings
import numpy as np
from lib import cmdline
from lib import dataset
from lib import encoder
from lib import classifier
from lib import rocplot


warnings.filterwarnings('ignore')


##### Argument
nSplit, nTree = 5, 500
Config = cmdline.ArgumentParser().parse_args()
errMsg = cmdline.ArgumentCheck(Config)

if errMsg != "":
    exit(errMsg)



##### Functions
def GetQ1Feature(Encoder):

    X1, y1   = Encoder.ToEAAC()
    # X2, y2   = Encoder.ToKSCTriad()
    # X3, y3   = Encoder.ToCTriad()
    # X4, y4   = Encoder.ToCTDD()
    X5, y5   = Encoder.ToZSCALE()
    # X6, y6   = Encoder.ToGTPC()
    # X7, y7   = Encoder.ToGDPC()
    # X8, y8   = Encoder.ToEGAAC()
    X9, y9   = Encoder.ToBINARY()
    # X10, y10 = Encoder.ToCKSAAGP()
    # X11, y11 = Encoder.ToCKSAAP()
    # X12, y12 = Encoder.ToCTDC()
    # X13, y13 = Encoder.ToEAAC()
    # X14, y14 = Encoder.ToDPC()
    # X15, y15 = Encoder.ToDDE()
    # X16, y16 = Encoder.ToTPC()
    # X17, y17 = Encoder.ToGAAC()
    # X18, y18 = Encoder.ToCTDT()
    # X21, y21 = Encoder.ToPWM_p()
    # X22, y22 = Encoder.ToPWM_n()
    # X23, y23 = Encoder.ToPWM_all()
    # X24, y24 = Encoder.ToPWM_d()
    X25, y25 = Encoder.ToPWM_d2()
    X26, y26 = Encoder.ToPWM_d3()
    # X27, y27 = Encoder.ToElectric()
    # X28, y28 = Encoder.ToPolor()
    # X29, y29 = Encoder.ToAromatic()

    DBs = {
        "EAAC"     : {"X": X1,  "y": y1 },
        # "KSCTriad" : {"X": X2,  "y": y2 },
        # "CTriad"   : {"X": X3,  "y": y3 },
        # "CTDD"     : {"X": X4,  "y": y4 },
        "ZSCALE"   : {"X": X5,  "y": y5 },
        # "GTPC"     : {"X": X6,  "y": y6 },
        # "GDPC"     : {"X": X7,  "y": y7 },
        # "EGAAC"    : {"X": X8,  "y": y8 },
        "BINARY"   : {"X": X9,  "y": y9 },
        # "CKSAAGP"  : {"X": X10, "y": y10},
        # "CKSAAP"   : {"X": X11, "y": y11},
        # "CTDC"     : {"X": X12, "y": y12},
        # "EAAC"     : {"X": X13, "y": y13},
        # "DPC"      : {"X": X14, "y": y14},
        # "DDE"      : {"X": X15, "y": y15},
        # "TPC"      : {"X": X16, "y": y16},
        # "GAAC"     : {"X": X17, "y": y17},
        # "CTDT"     : {"X": X18, "y": y18},
        # "PWM_p "   : {"X": X21, "y" : y21},
        # "PWM_n "   : {"X": X22, "y" : y22},
        # "PWM_a "   : {"X": X23, "y" : y23},
        # "PWM_d "   : {"X": X24, "y" : y24},
        "PWM_d2"   : {"X": X25, "y" : y25},
        "PWM_d3"   : {"X": X26, "y" : y26},
        # "Eletr   : {"X": X29, "y" : y29},
    }

    return DBs

def GetQ2Classifier():
    Clfs = {
       # "DT"  : {"Model" : classifier.DecisionTree(), "Name" : "Decision Tree"},
    #    "RF"  : {"Model" : classifier.RandomForest(nTree), "Name" : "Random Forest"},
    #    "SVM" : {"Model" : classifier.SupportVectorMachine(), "Name" : "Support Vector Machine"},
    #    "XGB" : {"Model" : classifier.XGBoost(nTree), "Name" : "XGBoost"},
       # "MLP" : {"Model" : classifier.MultilayerPerceptron(), "Name" : "Multilayer Perceptron"},
       "VC"  : {"Model" : classifier.VoteClassifier([
        ('rf1', classifier.RandomForest(nTree)),
        ('rf2', classifier.RandomForest(nTree, criterion = "entropy")),
        ('rf3', classifier.RandomForest(nTree, criterion = "log_loss")),
        ('svm', classifier.SupportVectorMachine()),
        ('cb', classifier.CatBoost(nTree)),
        ('gb', classifier.GradientBoosting()),
        ]), "Name" : "Ensemble Learning"},
       # # "Ada" : {"Model" : classifier.AdaBoost(), "Name" : "AdaBoost"},
    #    "GB"  : {"Model" : classifier.GradientBoosting(), "Name" : "Gradient Boosting"},
       # "ET"  : {"Model" : classifier.ExtraTrees(), "Name" : "Extra Trees"},
       # "GNB" : {"Model" : classifier.GaussianNaiveBayes(), "Name" : "Gaussian Naive Bayes"},
       # "KNN" : {"Model" : classifier.KNeighbors(), "Name" : "K Nearest Neighbors"},
    #    "CB"  : {"Model" : classifier.CatBoost(nTree), "Name" : "CatBoost"},
    }

    return Clfs



##### Main
DataSplit = dataset.SplitNfold(nSplit)
FeatureEncoder = encoder.Encode(Config.positive_data, Config.negative_data)


## Q1
# RF = classifier.CatBoost(nTree)
# Datas = GetQ1Feature(FeatureEncoder)

# print("### 1. Performance Comparison of Different Feature Encoding Methods")
# print("Feature", "\t".join(["Sn", "Sp", "Acc", "MCC", "AUC"]))

# RocPlot = rocplot.Roc_curve()

# for k, Vs in Datas.items():
#     X, y = Vs["X"], Vs["y"]
#     Evas = []

#     for trainIdx, testIdx in DataSplit.split(X, y):
#         X_train, X_test = X.iloc[trainIdx], X.iloc[testIdx]
#         y_train, y_test = y.iloc[trainIdx], y.iloc[testIdx]

#         RF.fit(X_train, y_train.values.ravel())

#         Evas.append(dataset.Evaluation(y_test, RF.predict(X_test), RF.predict_proba(X_test)[:, 1]))
#         RocPlot.append(y_test, RF.predict_proba(X_test)[:, 1])

#     RocPlot.add(k)

#     print("{}\t{}".format(k, "\t".join(["{:.3f}".format(100*v) for v in np.mean(np.array(Evas), axis=0)])))

# RocPlot.plot("top_5.png")
# del(Datas, RF)

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
        Evas.append(dataset.Evaluation(y_test, y_pred, model.predict_proba(X_test)[:, 1]))

    RocPlot.add(Vs["Name"])
    Means = np.mean(np.array(Evas), axis=0)
    Temps = np.mean(np.array(np.array(ROCs)), axis=0)

    print("{}\t{}".format(k, "\t".join(["{:.3f}".format(100*v) for v in Means])))

RocPlot.plot("output.png")

print("### Done !")
