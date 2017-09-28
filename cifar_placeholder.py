import tensorflow as tf
import numpy as np
import pandas as pd

##### Load data #####

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

data = unpickle("dataset/cifar/data_batch_1")
# data = np.hstack([data[b'data']/256, pd.get_dummies(data[b'labels']).values])
features = data[b'data']/256
labels = pd.get_dummies(data[b'labels']).values

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

# dataset = tf.contrib.data.Dataset.from_tensor_slices(data)
dataset = dataset.shuffle(buffer_size=1000)
batched_dataset = dataset.batch(100)
iterator = batched_dataset.make_initializable_iterator()
(x, y_) = iterator.get_next()
x = tf.cast(x, tf.float32)
# y_ = tf.cast(y_, tf.float32)
x_image = tf.reshape(x, [-1,32,32,3])

sess = tf.Session()

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
    train = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("log/")
writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())

for _ in range(5):
    sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})
    for i in range(83):
        if i%5 == 0:
            s = sess.run(merged)
            writer.add_summary(s, i)
        sess.run(train)

########################################################################################################################
########################################################################################################################
########################################################################################################################
data_test = unpickle("dataset/cifar/test_batch")
features_test = data_test[b'data']/256
labels_test = pd.get_dummies(data_test[b'labels']).values
sess.run(iterator.initializer, feed_dict={features_placeholder: features_test, labels_placeholder: labels_test})
accuracy = tf.constant(0, dtype = tf.float32)
for _ in range(83):
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(ro,1))
    accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))/83
print(sess.run(accuracy))
