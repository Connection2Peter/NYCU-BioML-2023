# Experiment 1
## Compare relation between number of window size and ML method
### Same setting
### 1. Balancing ratio
### 2. Using SSEPSSM as feature set

##### Import
import os, sys
import numpy as np
from lib import nns
from lib import tools
from lib import dataset
from lib import encoder
from tensorflow.keras.callbacks import EarlyStopping



##### Argument
if len(sys.argv) != 3:
	exit("Usage: python " + sys.argv[0] + " <input> <output>")

pathInput = sys.argv[1]
pathOutput = sys.argv[2]



##### Main
ratio = 0.2
batch = 512
num_class = 2

WindowsSizes = range(10, 100, 10)
BalanceRatios = range(1, 3, 1)
Methods = {"DNN" : 10000, "CNN" : 10000, "RNN" : 10000}

Encoder = encoder.EntireSeqEncoder(pathInput)
Encoder.LoadFromSSEPSSM(pathOutput)

ColNames = ["Method", "Windows", "Sn", "Sp", "Acc", "MCC", "AUC", "Ratio", "dbDistribution"]

fOut = open("Experiment1_NN_NoNorm.txt", "w")
fOut.write("\t".join(ColNames) + "\n")

print("\t".join(ColNames))

for balanceRatio in BalanceRatios:
	for window in WindowsSizes:
		for name, numEpoch in Methods.items():
			if name == "DNN":
				X, y = Encoder.toSeqKmerDB2D(window)
				X, y = tools.BalancingRatio(X, y, balanceRatio/2)

				model = nns.DNN_2(X.shape[1:], num_class)
			elif name == "CNN":
				X, y = Encoder.toSeqKmerDB3D(window)
				X, y = tools.BalancingRatio(X, y, balanceRatio/2)

				model = nns.CNN_1(X.shape[1:], num_class)
			elif name == "RNN":
				X, y = Encoder.toSeqKmerDB3D(window)
				X, y = tools.BalancingRatio(X, y, balanceRatio/2)

				model = nns.RNN_1(X.shape[1:], num_class)

			X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

			early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

			model.fit(X_train, tools.Label2Possibility(y_train), batch_size=batch, epochs=numEpoch, shuffle=True, callbacks=[early_stopping],verbose=None)
	
			y_pred = model.predict(X_test)

			Metrics = [str(round(m, 5)) for m in dataset.Evaluation(y_test, np.argmax(y_pred, axis=1))]

			Line = "{}\t{}\t".format(name, window) + "\t".join(Metrics) + "\t{}\t{}".format(balanceRatio/2, tools.ShowDistribution(y))

			print(Line)

			fOut.write(Line + "\n")

			del(model)

print("Done !")
