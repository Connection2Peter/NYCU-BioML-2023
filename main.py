##### Import
from lib import cmdline
from lib import dataset
from lib import encoder
from lib import classifier
import numpy as np



##### Argument
nSplit, nTree = 5, 500
Config = cmdline.ArgumentParser().parse_args()
errMsg = cmdline.ArgumentCheck(Config)

if errMsg != "":
    exit(errMsg)



##### Main
DataSplit = dataset.SplitNfold(nSplit)
AAencoder = encoder.Encoder(Config.positive_data, Config.negative_data)

X, y = AAencoder.ToBLOSUM62()

RF = classifier.RandomForest(nTree)

for trainIdx, testIdx in DataSplit.split(X, y):
    X_train, X_test = X.iloc[trainIdx], X.iloc[testIdx]
    y_train, y_test = y.iloc[trainIdx], y.iloc[testIdx]

    RF.fit(X_train, y_train.values.ravel())

    print(dataset.Accuracy(y_test, RF.predict(X_test)))

