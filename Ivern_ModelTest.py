##### Import
import os, sys
import numpy as np
from lib import tools
from lib import dataset
from lib import encoder
from lib import classifier
from lib import Ivern_encoder
from sklearn.metrics import roc_curve, auc




##### Main
ratio = 0.2
nTree = 500
nSplit = 5
WindowsSizes = [30]
EAACwin = [8]
BalanceRatios = [1]

Clfs = {
    # "LightGBM" : classifier.RandomForest(500),
    # "ET" : classifier.ExtraTrees(nTree),
    "CB": classifier.CatBoost(nTree),
}

ColNames = ["Model", "Sn", "Sp", "Acc", "MCC", "AUC", "Ratio", "Window", "dbDistribution"]

print("\t".join(ColNames))

fOut = open("Ivern_ModelTest.txt", "w")
fOut.write("\t".join(ColNames))

for k, model in Clfs.items():
  for balanceRatio in BalanceRatios:
    for window in WindowsSizes:
      for eaac in EAACwin:
        FeatureEncoder = Ivern_encoder.Iencoder("dataset/Connection/testNR/positive_" + str(window) + ".nr050.fasta", "dataset/Connection/testNR/negative_" + str(window) + ".nr050.fasta")
        X, y = FeatureEncoder.ToEAAC_np(eaac)
        X, y = tools.BalancingRatio(X, y, balanceRatio)
        dataSplit = dataset.SplitNfold(nSplit)

        X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        optimal_threshold_index = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_threshold_index]
        print('Optimal Threshold:', optimal_threshold)

        y_pred = []
        for i in y_probs:
          if i < optimal_threshold:
            y_pred.append(0)
          else:
            y_pred.append(1)

        Metrics = dataset.Evaluation(y_test, y_pred)

        Line = k + "\t" + "\t".join(["{:.3f}".format(100*v) for v in Metrics]) + "\t{}\t{}\t{}".format(balanceRatio, window, tools.ShowDistribution(y))

        print(Line)
        fOut.write(Line)


print("Done !")
