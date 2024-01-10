#### Import
import os, sys
from lib import cmdline
from lib import dataset
from lib import Brian_encoder as Bencoder
from lib import classifier



##### Argument
n = 10
w = 5
Config = cmdline.ArgumentParser_IndependentTest().parse_args()
errMsg = cmdline.ArgumentCheck_IndependentTest(Config)

if errMsg != "":
    exit(errMsg)



##### Main
#model = classifier.Load(Config.model_path)
Itest = Bencoder.IndependentTest(Config.input_file)
Itest.TSV2Kmers(n)

print ( Itest.ToEAAC(w))
#print( Itest.Kmers)
