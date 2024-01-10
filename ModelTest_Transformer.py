##### Import
import os, sys
import numpy as np
from lib import cmdline
from lib import dataset
from lib import encoder
from lib import classifier
from lib import transformer



##### Argument
# conda activate BioML2023
# cls & python ModelTesting_v231212.py dataset\Connection\RawDataset\train_plant_15000.txt
if len(sys.argv) != 2:
	exit("Usage: python " + sys.argv[0] + " <file>")

file = sys.argv[1]



##### Main
ratio = 0.3
val_ratio = 0.1
maxlen = 5000
vocab_size = 25
ff_dim = 64 # 64
embed_dim = 64 # 64
epochs = 20 # 5
num_heads = 12 # 3
batch_size = 4 # 16

Encoder = encoder.EntireSeqEncoder(file)
X, y = Encoder.toSeqDB(maxlen)

print("Shape of X :", X.shape)
print("Shape of y :", y.shape)
exit()

X_train, X_tmp, y_train, y_tmp = dataset.SplitDataset(X, y, ratio)
X_val, X_test, y_val, y_test = dataset.SplitDataset(X_tmp, y_tmp, val_ratio)

print("Shape of X_train / y_train :", X_train.shape, y_train.shape)
print("Shape of X_test / y_test :", X_test.shape, y_test.shape)
print("Shape of X_val / y_val :", X_val.shape, y_val.shape)
exit()
model = transformer.Transformer(
	maxlen=maxlen,
	num_heads=num_heads,
	ff_dim=ff_dim,
	embed_dim=embed_dim,
	vocab_size=vocab_size,
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
