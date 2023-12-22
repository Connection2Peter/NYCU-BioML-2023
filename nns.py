##### Import
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import GRU, Dense, Attention, LSTM, MultiHeadAttention
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Lambda, InputLayer
from tensorflow.keras.models import Model, Sequential



##### Functions

## DNN
def DNN_1(Shape, num_class):
    inputs = layers.Input(shape=Shape)
    x = layers.Dense(2000, activation="relu")(inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(200, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(8, activation="relu")(x)
    outputs = layers.Dense(num_class, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=optimizers.RMSprop(0.001, decay=1e-6),
        metrics=['accuracy']
    )

    return model

def DNN_2(Shape, num_class):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=Shape))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_class, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def DNN_3(Shape, num_class, Nodes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=Shape))
    model.add(layers.Dense(Nodes[0], activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(Nodes[1], activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_class, activation='softmax'))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

## CNN
def CNN_1(Shape, num_class):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=Shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

## RNN
def RNN_1(Shape, num_class):
    model = Sequential()
    model.add(LSTM(64, input_shape=Shape))
    model.add(Dense(num_class, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

## GRU
def GRU_1(Shape, num_class):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.GRU(units=128, input_shape=Shape))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
