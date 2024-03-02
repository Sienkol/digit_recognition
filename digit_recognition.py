import numpy as np
import struct
from array import array
from os.path import join
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

indexes = np.random.randint(0, len(x_test), 10)
print(indexes)
fig, axes = plt.subplots(1, 10, figsize=(25, 25))
for i, index in enumerate(indexes):
    axes[i].imshow(x_test[index], cmap="gray")
    axes[i].set_title(f"Cyfra: {y_test[index]}")
plt.show()
