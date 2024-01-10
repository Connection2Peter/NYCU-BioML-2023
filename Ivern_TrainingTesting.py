##### Import
import os, sys
from lib import cmdline
from lib import dataset
from lib import Ivern_encoder
from lib import classifier
from lib import tools



##### Argument
nTree = 500
ratio = 0.1
balanceRatio = 1.25
EAACwin = 5
Config = cmdline.ArgumentParser_TrainTest().parse_args()
errMsg = cmdline.ArgumentCheck_TrainTest(Config)

if errMsg != "":
    exit(errMsg)



##### Main
FeatureEncoder = Ivern_encoder.Iencoder(Config.positive_data, Config.negative_data)

X, y = FeatureEncoder.ToEAAC_np(EAACwin)
X, y = tools.BalancingRatio(X, y, balanceRatio)

X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

model = classifier.CatBoost(nTree)
model.fit(X_train, y_train)

classifier.Save(model, Config.output_model)

print(dataset.Evaluation(y_test, model.predict(X_test)))
print("Done !")
