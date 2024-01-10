##### Import
import os, sys
import numpy as np
from lib import tools
from lib import dataset
from lib import encoder
from lib import ssepssm
from lib import transformer

from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


##### Argument
if len(sys.argv) != 3:
	exit("Usage: python " + sys.argv[0] + " <input> <output>")

pathInput = sys.argv[1]
pathOutput = sys.argv[2]



##### Main
ratio = 0.3
val_ratio = 0.9
maxlen = 5000
vocab_size = 100
len_feature = 3 # 60
ff_dim = 32 # 64
embed_dim = 32 # 64
epochs = 5 # 5
num_heads = 3 # 3
batch_size = 2 # 16

Encoder = encoder.EntireSeqEncoder(pathInput)
Encoder.LoadFromSSEPSSM(pathOutput)

X, y = Encoder.toSeqDB3D(maxlen, vocab_size)
X = X[:, :, :len_feature]
print("Shape of X :", X.shape)
print("Shape of y :", y.shape)
exit()
X_train, X_tmp, y_train, y_tmp = dataset.SplitDataset(X, y, ratio)
X_val, X_test, y_val, y_test = dataset.SplitDataset(X_tmp, y_tmp, val_ratio)

print("Shape of X_train / y_train :", X_train.shape, y_train.shape)
print("Shape of X_test / y_test :", X_test.shape, y_test.shape)
print("Shape of X_val / y_val :", X_val.shape, y_val.shape)

model = transformer.Transformer3D(
	maxlen=maxlen,
	num_heads=num_heads,
	ff_dim=ff_dim,
	embed_dim=embed_dim,
	vocab_size=vocab_size,
    len_feature=len_feature,
)
model.summary()
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

Reals, Preds, Accs = [], [], []
for i in range(len(X_val)):
	Reals.append(y_val[i])
	Preds.append(transformer.SeqProbMat2ClassMat(model.predict(X_val[i:i+1])[0]))
	Accs.append(transformer.Evualte(Preds[i], Reals[i]))

transformer.ExportPredRealDB2Csv(Reals, Preds, Accs, "PredReal.csv")

print("="*30)
print("Validation DB :", X_val.shape)
print("Mean Accuracy :", np.mean(Accs))
print("="*30)

print("Done !")
