##### Import
import os, sys
from lib import cmdline
from lib import dataset
from lib import classifier



##### Argument
n = 10
Config = cmdline.ArgumentParser_IndependentTest().parse_args()
errMsg = cmdline.ArgumentCheck_IndependentTest(Config)

if errMsg != "":
    exit(errMsg)



##### Main
Seqs = dataset.LoadTSV2List(Config.input_file)
model = classifier.Load(Config.model_path)

for seq in Seqs[1:]:
	Ks = dataset.Seq2Kmer(seq[1], n)

	for k in Ks:
		print(model.predict(k))

	exit()