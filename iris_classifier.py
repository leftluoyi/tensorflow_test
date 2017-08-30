import tensorflow as tf
import pandas as pd
import numpy as np

sess = tf.Session()

rawdata =  pd.read_csv("dataset/iris.csv", dtype={"clength":np.float32}).values
features = rawdata[:,0:4].astype(np.float32)
labels = pd.get_dummies(rawdata[:,4]).values

dataset = tf.contrib.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.repeat(1)
dataset = dataset.batch(50)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

while True:
    try:
        next_feature, next_label = sess.run(next_element)
        print("===================================")
        print(next_label)
    except tf.errors.OutOfRangeError:
        break

