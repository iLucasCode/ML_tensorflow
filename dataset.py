import os

os.environ["TFF_CPP_MIN_LOG_LVL"] = "2"

import tensorflow as tf
import numpy as np
import cv2
import pickle

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def process_path(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image)
    return image

# TRAIN DATA

DATADIR = "C:/Users/klimc/PycharmProjects/Praca/raster/dane_640/input/class"
number = range(1,101)

CATEGORIES_TRAIN = []

for i in number:
    classes = str(i)
    print(classes)
    CATEGORIES_TRAIN.append(classes)

training_data = []

def create_training_data():
    for category in CATEGORIES_TRAIN:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES_TRAIN.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                training_data.append([img_array,class_num])
                print(img)
            except Exception as e:
                pass

create_training_data()

x_train = []
y_train = []
for features, label in training_data:
    x_train.append(features)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

pickle_out = open("x_train.pickle","wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

# TEST DATA
number_test = range(1,21)

CATEGORIES_TEST = []

for i in number_test:
    classes = str(i)
    print(classes)
    CATEGORIES_TEST.append(classes)

DATADIR_TEST = "C:/Users/klimc/PycharmProjects/Praca/raster/dane_640/input/class"
test_data = []
def create_test_data():
    for category in CATEGORIES_TEST:
        path = os.path.join(DATADIR_TEST, category)
        class_num = CATEGORIES_TEST.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                test_data.append([img_array,class_num])
            except Exception as e:
                pass

create_test_data()

import random
random.shuffle(test_data)

x_test = []
y_test = []
for features, label in test_data:
    x_test.append(features)
    y_test.append(label)

x_test = np.array(x_test)
y_test = np.array(y_test)

pickle_out = open("x_test.pickle","wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

# PREDICTION DATA

DATADIR_PRE = "C:/Users/klimc/PycharmProjects/Praca/raster/dane_640/pre2/img_100"
CATEGORIES = ["img"]
pre_data = []
def create_pre_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR_PRE, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array,(64,64))
                pre_data.append([img_array,class_num])
                print(img)
            except Exception as e:
                pass

create_pre_data()

x_pre = []
y_pre = []


for features, label in pre_data:
    x_pre.append(features)
    y_pre.append(label)


x_pre = np.array(x_pre).reshape(-1,64,64,3)
y_pre = np.array(y_pre)


pickle_out = open("x_pre.pickle","wb")
pickle.dump(x_pre, pickle_out)
pickle_out.close()

pickle_out = open("y_pre.pickle","wb")
pickle.dump(y_pre, pickle_out)
pickle_out.close()