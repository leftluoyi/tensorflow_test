import tensorflow as tf
import pandas as pd

bp = tf.random_uniform([4, 10])
rawdata = pd.read_csv("dataset/iris_dummy.csv", skip_blank_lines=1).values
features = rawdata[:, 0:4]
labels = rawdata[:, 4:7]
dataset = tf.contrib.data.Dataset.from_tensor_slices((features, labels))
# print(type(dataset))  # Do not try to print this !!!

dataset = dataset.shuffle(buffer_size=100)
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_feature, next_label = iterator.get_next()

sess = tf.Session()

print(sess.run(next_label))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_label))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_label))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])


