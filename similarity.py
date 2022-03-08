import os
import pathlib

os.environ["TFF_CPP_MIN_LOG_LVL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pathlib
import matplotlib.pyplot as plt
import tensorflow_io as tifo
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Embedding output layer with L2 norm
from tensorflow_similarity.layers import MetricEmbedding
# Specialized metric loss
from tensorflow_similarity.losses import MultiSimilarityLoss
# Sub classed keras Model with support for indexing
from tensorflow_similarity.models import SimilarityModel
# Data sampler that pulls datasets directly from tf dataset catalog
from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler
# Nearest neighbor visualizer
from tensorflow_similarity.visualization import viz_neigbors_imgs
import tensorflow_similarity as tfsim




physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


img_height = 64
img_width = 64
batch_size = 3

import pickle

# MODEL
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

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))
print(len(x_pre))
print(len(y_pre))

x_train = x_train/255.0
x_test = x_test/255.0
x_pre = x_pre/255.0

def transform_data(item, label):


  # Create output tensor
  output = tf.stack([item, label], axis=0)
  return output, label

# Create a training dataset from the base dataset - for each batch map the input format to the goal format by passing the mapping function
train_dataset = tf.data.Dataset.map(transform_data(x_pre,y_pre))

inputs = layers.Input(shape=(64, 64, 3))
x = keras.layers.Conv2D(64, 3, activation="relu")(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(128, 3, activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D((4, 4))(x)
x = keras.layers.Conv2D(256, 3, activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(256, 3, activation="relu")(x)
x = keras.layers.GlobalMaxPool2D()(x)
outputs = tfsim.layers.MetricEmbedding(1)(x)

# Build a specialized Similarity model
model = SimilarityModel(inputs, outputs)

# Train Similarity model using contrastive loss
model.compile('adam', loss=MultiSimilarityLoss())
model.fit(train_dataset, epochs=5)

sx, sy = train_dataset.get_slice(0,100)
model.index(x=sx, y=sy, data=sx)

# Find the top 5 most similar indexed MNIST examples for a given example
qx, qy = train_dataset.get_slice(3, 1)
nns = model.single_lookup(qx[0])

# Visualize the query example and its top 5 neighbors
viz_neigbors_imgs(qx[0], qy[0], nns)