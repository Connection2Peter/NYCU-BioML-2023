##### Import
import os, sys
from lib import cmdline
from lib import dataset
from lib import encoder
from lib import classifier
from lib import transformer



##### Argument
if len(sys.argv) != 2:
	exit("Usage: python " + sys.argv[0] + " <file>")

file = sys.argv[1]



##### Main
ratio = 0.2
maxlen = 5000
epochs = 10
vocab_size = 23
batch_size = 24

Encoder = encoder.EntireSeqEncoder(file)
X, y = Encoder.toSeqDB(maxlen)

print("Shape of X :", X.shape)
print("Shape of y :", y.shape)

X_train, X_test, y_train, y_test = dataset.SplitDataset(X, y, ratio)

print("Shape of X_train :", X_train.shape)
print("Shape of X_test :", X_test.shape)

model = transformer.Transformer(
	vocab_size=vocab_size,
	maxlen=maxlen,
	embed_dim=32,
	num_heads=2,
	ff_dim=32,
)
model.summary()
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

print(dataset.Evaluation(y_test, model.predict(X_test)))
print("Done !")
