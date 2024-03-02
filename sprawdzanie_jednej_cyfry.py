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

model = Sequential()

# Dodawanie warstwy wejściowej z 784 neuronami (rozmiar obrazu 28x28 pikseli)
# Dodawanie dwóch warstw ukrytych po 16 neuronów każda
# Dodawanie warstwy wyjściowej z 10 neuronami (odpowiadającymi cyfrom od 0 do 9)
model.add(Dense(units=784, activation="relu", input_shape=(28, 28)))
model.add(Dense(units=16, activation="relu"))
model.add(Flatten())
model.add(Dense(units=16, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

# Wczytywanie wag z pliku
model.load_weights("model_weights.h5")

# Kompilowanie modelu
model.compile(
    optimizer="sgd",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

index = np.random.randint(0, len(x_test))
x_sample = x_test[index]

predictions = []

for _ in range(10):
    # Przewidzenie dla próbki
    prediction = model.predict(x_sample.reshape(1, 28, 28))
    # Dodanie predykcji do listy
    predictions.append(np.argmax(prediction))

# Wyświetlenie predykcji
print(f"Predykcje: {predictions}")
print(y_test[index])

fig, axes = plt.subplots(1, 1, figsize=(10, 10))
axes.imshow(x_test[index], cmap="gray")
axes.set_title(f"Cyfra: {y_test[index]}")
plt.show()
