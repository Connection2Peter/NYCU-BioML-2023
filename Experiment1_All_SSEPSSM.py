##### Import
import os, sys
import numpy as np
from lib import nns
from lib import tools
from lib import dataset
from lib import encoder
from lib import classifier
from tensorflow.keras.callbacks import EarlyStopping



##### Argument
if len(sys.argv) != 2:
	exit("Usage: python " + sys.argv[0] + " <input>")



##### Main
ratio = 0.2
batch = 256
numClass = 2
num_tree = 1000
numEpoch = 10000
WindowsSizes = range(10, 51, 10)
BalanceRatios = [i/2 for i in range(1, 4, 1)]
Methods = {
	"RF"  : classifier.RandomForest(num_tree),
	"XGB" : classifier.XGBoost(num_tree),
	"CAT" : classifier.CatBoost(num_tree),
#	"SVM" : classifier.SupportVectorMachine(),
	"DNN" : "",
}

ColNames = ["Method", "Windows", "Sn", "Sp", "Acc", "MCC", "AUC", "Ratio", "dbDistribution"]
print("\t".join(ColNames))

Encoder = encoder.EntireSeqEncoder()
Encoder.SeqSSEPSSMs = dataset.LoadObject(sys.argv[1])

for balanceRatio in BalanceRatios:
	for window in WindowsSizes:
		for name, model in Methods.items():
			X, y = Encoder.toSeqKmerDB2D(window)
			X, y = tools.BalancingRatio(X, y, balanceRatio)
			X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

			if name == "DNN":
				model = nns.DNN_2(X.shape[1:], numClass)
				model.fit(
					X_train,
					tools.Label2Possibility(y_train),
					batch_size=batch,
					epochs=numEpoch,
					shuffle=True,
					callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)],
					verbose=None,
				)
			else:
				model.fit(X_train, y_train)

			y_pred = model.predict(X_test)

			if name == "DNN":
				Metrics = [str(round(m, 5)) for m in dataset.Evaluation(y_test, np.argmax(model.predict(X_test), axis=1))]
			else:
				Metrics = [str(round(m, 5)) for m in dataset.Evaluation(y_test, y_pred)]

			Line = "{}\t{}\t".format(name, window) + "\t".join(Metrics) + "\t{}\t{}".format(balanceRatio, tools.ShowDistribution(y))

			print(Line)

			del(model)

print("Done !")
