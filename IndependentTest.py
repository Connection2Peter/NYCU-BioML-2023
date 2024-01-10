##### Import
import os, sys
from lib import cmdline
from lib import dataset
from lib import encoder
from lib import classifier
import pandas as pd



##### Argument
n = 10
Config = cmdline.ArgumentParser_IndependentTest().parse_args()
errMsg = cmdline.ArgumentCheck_IndependentTest(Config)

if errMsg != "":
    exit(errMsg)



##### Main
model = classifier.Load(Config.model_path)
Itest = encoder.IndependentTest(Config.input_file)
Itest.TSV2Kmers(n)

for i, feature in enumerate(Itest.ToPSSM()):
    output = model.predict(pd.DataFrame([feature]))[0]==1
    print(f"{Itest.IDs[i]}\t{Itest.Pos[i]}\t{output}")
