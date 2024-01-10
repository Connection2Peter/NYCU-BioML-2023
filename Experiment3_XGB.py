##### Import
import sys
import numpy as np
from lib import tools
from lib import cmdline
from lib import dataset
from lib import classifier
from lib import Ivern_encoder as Iencoder



##### Argument
ratio = 0.2
Config = cmdline.ArgumentParser().parse_args()
errMsg = cmdline.ArgumentCheck(Config)

if errMsg != "":
    exit(errMsg)



##### Main
EncodeberIvern = Iencoder.Iencoder(Config.positive_data, Config.negative_data)
Features = {
    "EGAAC"  : EncodeberIvern.ToEGAAC(),
    "BINARY" : EncodeberIvern.ToBINARY(),
    "EAAC"   : EncodeberIvern.ToEAAC(),
}

AllSets = []
for i in Features.values():
    setSin = np.array(i[0].values.tolist())
    AllSets.append(setSin)

X = dataset.Normalize2D(np.concatenate(AllSets, axis=1))
y = np.array(Features["EAAC"][1].values.tolist()).reshape(-1)

X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

model = classifier.XGBoostHPO()
model.fit(X_train, y_train.ravel())

print(model.best_params_)

ColNames = ["nTree", "Sn", "Sp", "Acc", "MCC", "AUC"]
print("\t".join(ColNames))

evaluation = dataset.Evaluation(model.predict(X_test), y_test)
print("{}\t".format(nTree) + "\t".join([str(round(i, 5)) for i in evaluation]))
