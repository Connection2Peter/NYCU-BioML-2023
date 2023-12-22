import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def TFM(maxlen, num_heads, vocab_size, embed_dim, ff_dim, num_class):
    inputs = layers.Input(shape=(maxlen))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_class, activation="softmax")(x)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def Transformer(maxlen, num_heads, vocab_size, embed_dim, ff_dim):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
#    x = layers.GlobalAveragePooling1D()(x)
#    x = layers.Dropout(0.1)(x)
#    x = layers.Dense(20, activation="relu")(x)
#    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    #model.compile(optimizer=optimizer"adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def Transformer3D(maxlen, num_heads, vocab_size, embed_dim, ff_dim, len_feature):
    vocab_size += 1
    inputs = layers.Input(shape=(maxlen, len_feature))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.Reshape((maxlen, len_feature*embed_dim))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    #model.compile(optimizer=optimizer"adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def SeqProbMat2ClassMat(ProbMat):
    return [np.argmax(i) for i in ProbMat]

def Evualte(pred, real):
    same = 0
    macLen = len(pred)

    for i in range(macLen):
        if pred[i] == real[i]:
            same += 1

    return same / macLen

def ExportDB2Csv(DB, saveTo):
    df = pd.DataFrame(DB)

    df.to_csv(saveTo, index=False, header=False)

def ExportPredRealDB2Csv(Real, Pred, Acc, saveTo):
    with open(saveTo, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        for i in range(len(Real)):
            csv_writer.writerow([str(i), str(Acc[i]), "Real"] + list(map(str, Real[i])))
            csv_writer.writerow([str(i), str(Acc[i]), "Pred"] + list(map(str, Pred[i])))
