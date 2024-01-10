# Experiment 1
## Compare relation between number of window size and ML method
### Same setting
### 1. Balancing ratio
### 2. Using SSEPSSM as feature set

##### Import
import os, sys
import numpy as np
from lib import tools
from lib import dataset
from lib import encoder
from lib import classifier



##### Argument
if len(sys.argv) != 3:
	exit("Usage: python " + sys.argv[0] + " <input> <output>")

pathInput = sys.argv[1]
pathOutput = sys.argv[2]



##### Main
ratio = 0.2

WindowsSizes = range(10, 100, 10)
BalanceRatios = range(1, 4, 1)
Methods = {
	"RF"  : classifier.RandomForest(1000),
	"MLP" : classifier.MultilayerPerceptron(),
	"XGB" : classifier.XGBoost(1000),
	"CAT" : classifier.CatBoost(1000),
	"SVM" : classifier.SupportVectorMachine(),
}

Encoder = encoder.EntireSeqEncoder(pathInput)
Encoder.LoadFromSSEPSSM(pathOutput)

ColNames = ["Method", "Windows", "Sn", "Sp", "Acc", "MCC", "AUC", "Ratio", "dbDistribution"]
print("\t".join(ColNames))

for balanceRatio in BalanceRatios:
	for window in WindowsSizes:
		for name, model in Methods.items():
			X, y = Encoder.toSeqKnerDB3D(window)
			X, y = tools.BalancingRatio(X, y, balanceRatio/2)
	
			X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)
	
			clf = model
			clf.fit(X_train, y_train)
	
			Metrics = [str(round(m, 5)) for m in dataset.Evaluation(y_test, model.predict(X_test))]
	
			print("{}\t{}\t".format(name, window) + "\t".join(Metrics) + "\t{}\t{}".format(balanceRatio/2, tools.ShowDistribution(y)))

print("Done !")
