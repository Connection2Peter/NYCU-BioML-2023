#### Import
import os, sys
from lib import cmdline
from lib import dataset
from lib import Brian_encoder as Bencoder
from lib import classifier
import pandas as pd



##### Argument
win = 10
eaac = 5
Config = cmdline.ArgumentParser_IndependentTest().parse_args()
errMsg = cmdline.ArgumentCheck_IndependentTest(Config)

if errMsg != "":
    exit(errMsg)



##### Main
#model = classifier.Load(Config.model_path)
Itest = Bencoder.IndependentTest(Config.input_file)


##### Main
model = classifier.Load(Config.model_path)

Itest.TSV2Kmers(win)
Features = Itest.ToEAAC(eaac)

fOut = open("第06組_2.csv", "w")

output = list(model.predict(pd.DataFrame(Features)))
for i, pred in enumerate(output):
  output = ""
  if pred == 0:
    output = "1"
  else:
    output = "0"

  fOut.write(f"{Itest.IDs[i]},{Itest.Pos[i]+1},{output}\n")
  print(f"{Itest.IDs[i]},{Itest.Pos[i]+1},{output}")
  
