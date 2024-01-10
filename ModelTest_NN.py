##### Import
import os, sys
import numpy as np
from lib import transformer
from lib import tools
from lib import dataset
from lib import encoder



##### Argument
if len(sys.argv) != 3:
	exit("Usage: python " + sys.argv[0] + " <input> <output>")

pathInput = sys.argv[1]
pathOutput = sys.argv[2]



##### Main
kmer = 40
label = 2
ratio = 0.3
val_ratio = 0.1
maxlen = 10000
num_feature = 60 * (kmer * 2 + 1)
Batch = 2
Epoch = 20
num_heads = 12
num_class = 2
vocab_size = 10000
ff_dim = 128
embed_dim = 128

Encoder = encoder.EntireSeqEncoder(pathInput)
Encoder.LoadFromSSEPSSM(pathOutput)

X, y = Encoder.toSeqKmerDB2DNorm(kmer, vocab_size)
X, y = tools.BalancingXY(X, y)

print("Shape of X :", X.shape)
print("Shape of y :", y.shape)

X_train, X_tmp, y_train, y_tmp = dataset.SplitDataset(X, y, ratio)
X_test, X_val, y_test, y_val = dataset.SplitDataset(X_tmp, y_tmp, val_ratio)

print("Shape of X_train / y_train :", X_train.shape, y_train.shape)
print("Shape of X_test / y_test :", X_test.shape, y_test.shape)
print("Shape of X_val / y_val :", X_val.shape, y_val.shape)

model = transformer.TFM(
	maxlen=maxlen,
	num_heads=num_heads,
	vocab_size=vocab_size+1,
    embed_dim=embed_dim,
	ff_dim=ff_dim,
	num_class=num_class,
	num_feature=num_feature,
)
model.summary()
model.fit(X_train, y_train, batch_size=Batch, epochs=Epoch, shuffle=True)

y_pred = model.predict(X_test)

print(dataset.Evaluation(y_test, np.argmax(y_pred, axis=1)))

print("Done !")
