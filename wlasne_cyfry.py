import numpy as np
import os
import cv2
from array import array
from os.path import join
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

model = tf.keras.models.load_model("model2.h5")

img_number = 0
while os.path.isfile(f"cyfry/{img_number}.png"):
    try:
        img = cv2.imread(f"cyfry/{img_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediciton = model.predict(img)
        print(f"cyfra {img_number} to {np.argmax(prediciton)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("blad")
    finally:
        img_number += 1
