import os

os.environ["TFF_CPP_MIN_LOG_LVL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

pickle_in = open("x_train.pickle","rb")
x_train = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y_train = pickle.load(pickle_in)

pickle_in = open("x_test.pickle","rb")
x_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

pickle_in = open("x_pre.pickle","rb")
x_pre = pickle.load(pickle_in)

pickle_in = open("y_pre.pickle","rb")
y_pre = pickle.load(pickle_in)

x_train = x_train/255.0
x_test = x_test/255.0
x_pre = x_pre/255.0

model = keras.models.Sequential()
model.add(layers.InputLayer(input_shape=(64, 64, 3)))
model.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='elu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='elu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='elu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=128, activation='elu'))
model.add(layers.Dense(units=100, activation='softmax'))

model.save('model.h5')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tb_callback_dir", histogram_freq=1)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.fit(x_train,y_train, epochs=15, batch_size=8, callbacks=[tensorboard_callback])

model.evaluate(x_test, y_test, batch_size=8)

pre_number = 47

predictions=model.predict(x_pre[pre_number-1].reshape(-1,64,64,3))
best_pre_index = np.argmax(predictions, axis=1)

print(predictions)
print(best_pre_index)


# PLOT

def plot_main_image(predictions_array,image,class_main):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title("Obszar z dnia 31.01.2017 r.")
    image = image[class_main].reshape(64,64,3)
    plt.imshow(image)
    plt.xlabel("Przypisana klasa:""{}\n""Obliczone prawdopodobieństwo:"" {:0.4f}".format(class_main+1,
                                np.max(predictions_array)),color='black')

def plot_image(class_index, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.title("Obszar z dnia 06.02.2017 r.")

  plt.imshow(img)

  plt.xlabel("Numer analizowanego obszaru:""{}".format(class_index),
                                color='black')



num_rows = 1
num_cols = 1
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range (num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(pre_number, x_pre[pre_number-1])
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_main_image(predictions,x_train,best_pre_index)
plt.tight_layout()
plt.show()