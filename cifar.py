import tensorflow as tf
import numpy as np
import pandas as pd

##### Load data #####

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle("dataset/cifar/data_batch_1")
data = np.hstack([data[b'data']/256, pd.get_dummies(data[b'labels']).values])

dataset = tf.contrib.data.Dataset.from_tensor_slices(data)
dataset = dataset.shuffle(buffer_size=1000)
batched_dataset = dataset.batch(100)
iterator = batched_dataset.make_one_shot_iterator()
next_data = iterator.get_next()

# print(pd.get_dummies(data[b'labels']).values[1:11,:].shape)
# print(data)

##### Train the model #####

sess = tf.Session()

# tmpdata = tf.cast(next_data, tf.float32)
# print(sess.run(tf.slice(tmpdata, [1, 3072], [-1,10])))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# x = tf.placeholder(tf.float32, shape=[None, 3072])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
tmpdata = tf.cast(next_data, tf.float32)
x = tf.slice(tmpdata, [1, 1], [-1,3072])
y_ = tf.slice(tmpdata, [1, 1], [-1,10])
x_image = tf.reshape(x, [-1,32,32,3])

with tf.name_scope("convolutional_layer"):
    W_conv1 = weight_variable([5,5,3,32])
    tf.summary.histogram("Convolutional layer weights", W_conv1)
    b_conv1 = bias_variable([32])
    tf.summary.histogram("Convolutional layer bias", b_conv1)
    conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)
    max1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    tf.summary.histogram("Max layer output", W_conv1)

with tf.name_scope("full_connected_layer"):
    W_fc = weight_variable([16 * 16 * 32, 1024])
    b_fc = bias_variable([1024])
    max1_flat = tf.reshape(max1, [-1, 16*16*32])
    fc = tf.nn.relu(tf.matmul(max1_flat, W_fc) + b_fc)

with tf.name_scope("readout_layer"):
    W_ro = weight_variable([1024, 10])
    b_ro = bias_variable([10])
    ro = tf.nn.softmax(tf.matmul(fc, W_ro) + b_ro)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=ro))
    tf.summary.scalar("loss", loss)
    train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("log/")
writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(100):
    # tmpdata = next_data
    # features = tf.slice(tmpdata, [1, 1], [-1,3072])
    # labels = tf.slice(tmpdata, [1, 3072], [-1,10])
    # print(features)
    # if i%5 == 0:
    #     s = sess.run(merged, {x:features, y_:labels})
    #     writer.add_summary(s, i)
    sess.run(train)