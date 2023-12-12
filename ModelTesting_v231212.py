##### Import
import os, sys
from lib import cmdline
from lib import dataset
from lib import encoder
from lib import classifier



##### Argument
nTree = 500
ratio = 0.1
Config = cmdline.ArgumentParser().parse_args()
errMsg = cmdline.ArgumentCheck(Config)

if errMsg != "":
    exit(errMsg)



##### Main
FeatureEncoder = encoder.Encode(Config.positive_data, Config.negative_data)

X, y = FeatureEncoder.ToPSSM()

X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

model = classifier.CatBoost(30000)
model.fit(X_train, y_train)

model2 = classifier.RandomForest(nTree)
model2.fit(X_train, y_train)

model3 = classifier.XGBoost(nTree)
model3.fit(X_train, y_train)

print(dataset.Evaluation(y_test, model.predict(X_test)))
print(dataset.Evaluation(y_test, model2.predict(X_test)))
print(dataset.Evaluation(y_test, model3.predict(X_test)))
print("Done !")
