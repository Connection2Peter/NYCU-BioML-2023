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
n = range(5, 60, 5)
nTree = [1000, 5000, 10000]

ratio = 0.2

Encoder = encoder.EntireSeqEncoder(pathInput)
Encoder.LoadFromSSEPSSM(pathOutput)

#X, y = Encoder.toSeqKnerDB3D(n)
#X, y = tools.BalancingXY(X, y)

#print("Shape of X :", X.shape)
#print("Shape of y :", y.shape)

#print(tools.ShowDistribution(y))

#X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

#print("Shape of X_train / y_train :", X_train.shape, y_train.shape)
#print("Shape of X_test / y_test :", X_test.shape, y_test.shape)

print("BalancingXY")
print("Method\tTree\tWindows", "\t".join(["Sn", "Sp", "Acc", "MCC", "AUC"]))
for numTree in nTree:
	for i in n:
		X, y = Encoder.toSeqKnerDB3D(i)
		X, y = tools.BalancingXY(X, y)

		X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

		model = classifier.CatBoost(numTree)
		model.fit(X_train, y_train)

		print("CatBoost {} {}".format(numTree, i), dataset.Evaluation(y_test, model.predict(X_test)))

#model = classifier.RandomForest(nTree)
#model.fit(X_train, y_train)

#print(dataset.Evaluation(y_test, model.predict(X_test)))

print("Done !")
