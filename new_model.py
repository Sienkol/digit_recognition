import numpy as np
import struct
from array import array
from os.path import join
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Tworzenie modelu sekwencyjnego
model = Sequential()

# Dodawanie warstwy wejściowej z 784 neuronami (rozmiar obrazu 28x28 pikseli)
# Dodawanie dwóch warstw ukrytych po 16 neuronów każda
# Dodawanie warstwy wyjściowej z 10 neuronami (odpowiadającymi cyfrom od 0 do 9)
model.add(Dense(units=784, activation="relu", input_shape=(28, 28)))
model.add(Dense(units=196, activation="relu"))
model.add(Flatten())
model.add(Dense(units=49, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

# Wczytywanie wag z pliku
# model.load_weights("model_weights2.h5")

# Kompilowanie modelu
model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Dopasowanie modelu do danych
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Ewaluacja modelu
model.evaluate(x_test, y_test)

# Zapisywanie wag do pliku
model.save_weights("model_weights2.h5")

model.save("model2.h5")