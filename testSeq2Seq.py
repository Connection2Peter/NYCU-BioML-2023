import numpy as np
from lib import transformer
from tensorflow import keras

maxlen = 200
vocab_size = 20000

(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_val))
print(np.shape(y_val))

x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)

print(np.shape(x_train))

model = transformer.Transformer(
	vocab_size=vocab_size,
	maxlen=maxlen,
	embed_dim=32,
	num_heads=2,
	ff_dim=32,
)

model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))

print(history.history)
print(model.predict(x_val[:1]))
