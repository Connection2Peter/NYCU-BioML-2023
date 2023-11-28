##### Import
from lib import cmdline
from lib import encoder
from lib import classifier
import pandas as pd



##### Argument
Config = cmdline.ArgumentParser().parse_args()
errMsg = cmdline.ArgumentCheck(Config)

if errMsg != "":
	exit(errMsg)



##### Main
AAencoder = encoder.Encoder(Config.positive_data, Config.negative_data)

X, y = AAencoder.ToPSSM()

print(pd.DataFrame(X))