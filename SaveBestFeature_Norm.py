##### Import
import sys
import numpy as np
from lib import cmdline
from lib import dataset
from lib import Ivern_encoder as Iencoder



##### Argument
ratio = 0.2
Config = cmdline.ArgumentParser_TrainTest().parse_args()
errMsg = cmdline.ArgumentCheck_TrainTest(Config)

if errMsg != "":
    exit(errMsg)



##### Main
X, y = Iencoder.Iencoder(Config.positive_data, Config.negative_data).ToEAAC()
X, y = dataset.Normalize2D(X.values.tolist()), y.values.reshape(-1).tolist()

print("X.shape: {}".format(np.shape(X)))
print("y.shape: {}".format(np.shape(y)))

dataset.SaveObject((X, y), Config.output_model)
