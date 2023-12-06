##### Import
import os, sys
from lib import cmdline
from lib import dataset
from lib import encoder
from lib import classifier



##### Argument
nTree = 500
ratio = 0.1
Config = cmdline.ArgumentParser_TrainTest().parse_args()
errMsg = cmdline.ArgumentCheck_TrainTest(Config)

if errMsg != "":
    exit(errMsg)



##### Main
FeatureEncoder = encoder.Encode(Config.positive_data, Config.negative_data)

X, y = FeatureEncoder.ToPSSM()

X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

model = classifier.XGBoost(nTree)
model.fit(X_train, y_train)

classifier.Save(model, Config.output_model)

print(dataset.Evaluation(y_test, model.predict(X_test)))
print("Done !")
