##### Import
import sys
import numpy as np
from lib import tools
from lib import cmdline
from lib import dataset
from lib import classifier



##### Argument
ratio = 0.2
dRatio = 1

if len(sys.argv) != 2:
    exit("Usage: python " + sys.argv[0] + " <input>")



##### Main
X, y = dataset.LoadObject(sys.argv[1])
X, y = tools.BalancingRatio(X, y, dRatio)
X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

ColNames = ["nTree", "Sn", "Sp", "Acc", "MCC", "AUC"]
print("\t".join(ColNames))

fOut = open("Experiment3_CAT.txt", "w")
fOut.write("\t".join(ColNames) + "\n")

for nTree in range(100, 50001, 100):
    model = classifier.CatBoost(nTree)
    model.fit(X_train, y_train.ravel())

    evaluation = dataset.Evaluation(model.predict(X_test), y_test)
    line = "{}\t".format(nTree) + "\t".join([str(round(i, 5)) for i in evaluation])
    print(line)

    fOut.write(line + "\n")

    del(model)
