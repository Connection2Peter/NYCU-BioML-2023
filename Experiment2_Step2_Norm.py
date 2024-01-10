##### Import
import sys
import numpy as np
from lib import tools
from lib import cmdline
from lib import dataset
from lib import classifier
from lib import Brian_encoder as Bencoder
from lib import Ivern_encoder as Iencoder



##### Argument
ratio = 0.2
nTree = 500
Config = cmdline.ArgumentParser().parse_args()
errMsg = cmdline.ArgumentCheck(Config)

if errMsg != "":
    exit(errMsg)



##### Main
EncodeberIvern = Iencoder.Iencoder(Config.positive_data, Config.negative_data)
EncodeberBrian = Bencoder.Encode(Config.positive_data, Config.negative_data)
Features = {
    "EGAAC"  : EncodeberIvern.ToEGAAC(),
    "BINARY" : EncodeberIvern.ToBINARY(),
    "EAAC"   : EncodeberIvern.ToEAAC(),
    "PWM_d2" : EncodeberBrian.ToPWM_d2(),
    "PWM_d3" : EncodeberBrian.ToPWM_d3(),
}

ColNames = ["Sn", "Sp", "Acc", "MCC", "AUC", "Combination"]
print("\t".join(ColNames))

fOut = open("Experiment2_Step2_Norm.txt", "w")
fOut.write("\t".join(ColNames) + "\n")

for SetCombs in tools.GetAllCombination(Features.keys()):
    AllSets = []
    for i in range(len(SetCombs)):
        setSin = np.array(Features[SetCombs[i]][0].values.tolist())
        AllSets.append(setSin)

    X = dataset.Normalize2D(np.concatenate(AllSets, axis=1))
    y = np.array(Features[SetCombs[0]][1].values.tolist()).reshape(-1)
    
    X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

    model = classifier.RandomForest(nTree)
    model.fit(X_train, y_train.ravel())

    evaluation = dataset.Evaluation(model.predict(X_test), y_test)
    line = "\t".join([str(round(i, 5)) for i in evaluation]) + "\t" + "+".join(SetCombs)
    print(line)

    fOut.write(line + "\n")

    del(model)
