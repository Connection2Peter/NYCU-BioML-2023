##### Import
import os, sys
import numpy as np
from lib import tools
from lib import cmdline
from lib import dataset
from lib import encoder
from lib import classifier
from lib import transformer
from tensorflow.keras.callbacks import EarlyStopping



##### Argument
if len(sys.argv) != 2:
    exit("Usage: python " + sys.argv[0] + " <ssepssm>")



##### Setting
### Dataset
ratio = 0.3
val_ratio = 0.5
num_class = 2
num_feature = 60
# Preprocessing
scale = 10000

### Model
# Number of feature in whole sentense
# e.g. In this case, maxlen will be (2*k+1)*num_feature
maxlen = None
# Max number of encode index of word
# e.g. If encode word as 0~1500, vocab_size=1500
vocab_size = scale+1
# Numbers of attention head
num_heads = 3
# A word will mapping to ff_dim & decode in embed_dim size array
ff_dim = 128
embed_dim = 128

### Train
epochs = 1000
batch_size = 50



##### Main
window = 10
Encoder = encoder.EntireSeqEncoder()
Encoder.SeqSSEPSSMs = dataset.LoadObject(sys.argv[1])

ColNames = "\t".join(["Sn", "Sp", "Acc", "MCC", "AUC", "balance"])

fOut = open("ModelTest_TFM.txt", "w")
fOut.write(ColNames)

for bRatio in range(1, 13):
    X, y = Encoder.ReEncodeToSeqKmerDB2D(window, scale)

    X, y = tools.BalancingRatio(X, y, bRatio)
    
    print("Shape of X :", X.shape)
    print("Shape of y :", y.shape)

    X_train, X_tmp, y_train, y_tmp = dataset.SplitDataset(X, y, ratio)
    X_test, X_val, y_test, y_val = dataset.SplitDataset(X_tmp, y_tmp, val_ratio)

    print("Info of X_train & y_train :", X_train.shape, y_train.shape, tools.ShowDistribution(y_train))
    print("Info of X_test  & y_test  :", X_test.shape, y_test.shape, tools.ShowDistribution(y_test))
    print("Info of X_val   & y_val   :", X_val.shape, y_val.shape, tools.ShowDistribution(y_val))

    maxlen = num_feature*(2*window+1)

    model = transformer.TFM(
        maxlen=maxlen,
        num_heads=num_heads,
        ff_dim=ff_dim,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        num_class=num_class,
    )
    model.summary()
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)],
    )

    Metrics = dataset.Evaluation(y_test, np.argmax(model.predict(X_test), axis=1))
    
    line = "\t".join([str(round(m, 5)) for m in Metrics]) + "\t" + str(window)
    fOut.write(line)

    print(line)

    del(model)

print("Done !")
