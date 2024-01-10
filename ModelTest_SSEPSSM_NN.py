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
batch = 16
numClass = 2
numEpoch = 10000
WindowsSizes = range(10, 51, 10)
BalanceRatios = [i/2 for i in range(1, 20, 1)]

ColNames = ["Layer1", "Layer2", "Sn", "Sp", "Acc", "MCC", "AUC", "Ratio", "Window", "dbDistribution"]
AllNodes = [
    [200, 200],
    [500, 300],
    [1000, 128],
    [2000, 128],
    [200, 256],
]

print("\t".join(ColNames))

Encoder = encoder.EntireSeqEncoder()
Encoder.SeqSSEPSSMs = dataset.LoadObject(sys.argv[1])

fOut = open("ModelTest_SSEPSSM_NN.txt", "w")
fOut.write("\t".join(ColNames))

for balanceRatio in BalanceRatios:
	for window in WindowsSizes:
		for Nodes in AllNodes:
			X, y = Encoder.toSeqKmerDB2D(window)
			X, y = tools.BalancingRatio(X, y, balanceRatio)
			X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

			model = nns.DNN_3(X.shape[1:], numClass, Nodes)
			model.fit(
				X_train,
				tools.Label2Possibility(y_train),
				batch_size=batch,
				epochs=numEpoch,
				shuffle=True,
				callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)],
				verbose=None,
			)

			y_pred = model.predict(X_test)

			Metrics = [str(round(m, 5)) for m in dataset.Evaluation(y_test, np.argmax(model.predict(X_test), axis=1))]

			Line = "{}\t{}\t".format(Nodes[0], Nodes[0]) + "\t".join(Metrics) + "\t{}\t{}\t{}".format(balanceRatio, window, tools.ShowDistribution(y))

			print(Line)
			fOut.write(Line)

			del(model)

print("Done !")
