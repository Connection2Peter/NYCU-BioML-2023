##### Import
import sys
import numpy as np
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
EncodeberBrian = Bencoder.Encode(Config.positive_data, Config.negative_data)
EncodeberIvern = Iencoder.Iencoder(Config.positive_data, Config.negative_data)

Features = {
    "PWM_p " : EncodeberBrian.ToPWM_p(),
    "PWM_n " : EncodeberBrian.ToPWM_n(),
    "PWM_a " : EncodeberBrian.ToPWM_all(),
    "PWM_d " : EncodeberBrian.ToPWM_d(),
    "PWM_d2" : EncodeberBrian.ToPWM_d2(),
    "PWM_d3" : EncodeberBrian.ToPWM_d3(),
    "Charge" : EncodeberBrian.ToElectric(),
    "Polor " : EncodeberBrian.ToPolor(),
    "Armatc" : EncodeberBrian.ToAromatic(),
    "EAAC"   : EncodeberIvern.ToEAAC(),
    "CTDC"   : EncodeberIvern.ToCTDC(),
    "DPC"    : EncodeberIvern.ToDPC(),
    "DDE"    : EncodeberIvern.ToDDE(),
    "KSCT"   : EncodeberIvern.ToKSCTriad(),
    "CTriad" : EncodeberIvern.ToCTriad(),
    "CTDD"   : EncodeberIvern.ToCTDD(),
    "ZSCALE" : EncodeberIvern.ToZSCALE(),
    "GTPC"   : EncodeberIvern.ToGTPC(),
    "GDPC"   : EncodeberIvern.ToGDPC(),
    "EGAAC"  : EncodeberIvern.ToEGAAC(),
    "BINARY" : EncodeberIvern.ToBINARY(),
    "CKSAAGP": EncodeberIvern.ToCKSAAGP(),
    "CKSAAP" : EncodeberIvern.ToCKSAAP(),
    "CTDC"   : EncodeberIvern.ToCTDC(),
    "DPC"    : EncodeberIvern.ToDPC(),
    "DDE"    : EncodeberIvern.ToDDE(),
    "GAAC"   : EncodeberIvern.ToGAAC(),
    "CTDT"   : EncodeberIvern.ToCTDT(),
    "PSSM"   : EncodeberIvern.ToPSSM(),
}

ColNames = ["Method", "Sn", "Sp", "Acc", "MCC", "AUC"]
print("\t".join(ColNames))

for name, DBs in Features.items():
    X_train, X_test, y_train, y_test = dataset.SplitDataset(dataset.Normalize2D(DBs[0]), DBs[1], ratio)

    model = classifier.RandomForest(nTree)
    model.fit(X_train, y_train.values.ravel())

    evaluation = dataset.Evaluation(model.predict(X_test), y_test)
    print("{}\t".format(name) + "\t".join([str(round(i, 5)) for i in evaluation]))

    del(model)
