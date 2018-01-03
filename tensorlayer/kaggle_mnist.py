import tensorflow as tf
import tensorlayer as tl
import pandas as pd
import numpy as np
import csv

# data importing
data = pd.read_csv('data/mnist/kaggle_train.csv')
data_x = data.drop('label',1).values
data_y = data['label'].values

num = len(data.index)
trainset = np.random.choice(num, int(num * 0.9), replace=False)


train_X = data_x[trainset,:].astype(np.float32)
valid_X = np.delete(data_x, trainset, axis=0).astype(np.float32)
train_y = data_y[trainset]
valid_y = np.delete(data_y, trainset, axis=0)
test_X = pd.read_csv('data/mnist/kaggle_test.csv').values

# tensorflow initialize
sess = tf.InteractiveSession()
X = tf.placeholder(train_X.dtype, shape=[None, 784])
X = X / 256
y_ = tf.placeholder(train_y.dtype, shape=[None, ])

# tensorflow dense
network_dense = tl.layers.InputLayer(X, name='dense_input_layer')
network_dense = tl.layers.DenseLayer(network_dense, n_units=800, act=tf.nn.relu, name='dense_layer_1')
network_dense = tl.layers.DenseLayer(network_dense, n_units=200, act=tf.nn.relu, name='dense_layer_2')
network_dense = tl.layers.DenseLayer(network_dense, n_units=10, act=tf.identity, name='dense_layer_3')

y_dense = network_dense.outputs
cost_dense = tl.cost.cross_entropy(y_dense, y_, name='dense_cost')
correct_prediction_dense = tf.equal(tf.argmax(y_dense, 1), y_)
acc_dense = tf.reduce_mean(tf.cast(correct_prediction_dense, tf.float32))
train_op_dense = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                                        epsilon=1e-08, use_locking=False).minimize(cost_dense,
                                                                                   var_list=network_dense.all_params)

# tensorflow LeNET
X_image_conv = tf.reshape(X, [-1, 28, 28, 1])
network_conv = tl.layers.InputLayer(X_image_conv, name='conv_input_layer')
network_conv = tl.layers.Conv2dLayer(network_conv,
                                     act=tf.nn.relu,
                                     shape=[5, 5, 1, 32],  # 32 features for each 5x5 patch
                                     strides=[1, 1, 1, 1],
                                     padding='SAME',
                                     name='conv_layer_conv_1')  # output: (?, 28, 28, 32)
network_conv = tl.layers.PoolLayer(network_conv,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   pool=tf.nn.max_pool,
                                   name='conv_layer_pool_1', )
network_conv = tl.layers.Conv2dLayer(network_conv,
                                     act=tf.nn.relu,
                                     shape=[5, 5, 32, 64],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME',
                                     name='conv_layer_conv_2')
network_conv = tl.layers.PoolLayer(network_conv,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   pool=tf.nn.max_pool,
                                   name='conv_layer_pool_2', )
network_conv = tl.layers.FlattenLayer(network_conv, name='conv_flatten_layer')
network_conv = tl.layers.DenseLayer(network_conv, n_units=256, act=tf.nn.relu, name='conv_dense_1')
network_conv = tl.layers.DenseLayer(network_conv, n_units=10, act=tf.identity, name='conv_dense_2')

y_conv = network_conv.outputs
cost_conv = tl.cost.cross_entropy(y_conv, y_, name='conv_cost')
correct_prediction_conv = tf.equal(tf.argmax(y_conv, 1), y_)
acc_conv = tf.reduce_mean(tf.cast(correct_prediction_conv, tf.float32))
train_op_conv = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                                       epsilon=1e-08, use_locking=False).minimize(cost_conv,
                                                                                  var_list=network_conv.all_params)

# train
tl.layers.initialize_global_variables(sess)
tl.utils.fit(sess, network_dense, train_op_dense, cost_dense, X_train=train_X, y_train=train_y, x=X, y_=y_, X_val=valid_X, y_val=valid_y,
             acc=acc_dense, batch_size=500, n_epoch=1, print_freq=1,
             eval_train=False, tensorboard=False, tensorboard_epoch_freq=1)
tl.utils.fit(sess, network_conv, train_op_conv, cost_conv, X_train=train_X, y_train=train_y, x=X, y_=y_, X_val=valid_X, y_val=valid_y,
             acc=acc_conv, batch_size=500, n_epoch=50, print_freq=1,
             eval_train=False, tensorboard=False, tensorboard_epoch_freq=1)

# save models
tl.files.save_npz(network_dense.all_params , name='model_dense.npz')
tl.files.save_npz(network_conv.all_params , name='model_conv.npz')

# prediction
prediction_conv = tl.utils.predict(sess, network_conv, test_X, X, y_conv_op, batch_size=500)
prediction_dense = tl.utils.predict(sess, network_dense, test_X, X, y_dense_op, batch_size=500)

prediction_conv_flat = np.reshape(prediction_conv, [-1, 10], order='F')
prediction_dense_flat = np.reshape(prediction_dense, [-1, 10], order='F')

with open('predictions.csv', 'w', newline='') as csvfile:
    fieldnames = ['ImageId', 'Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(prediction_conv)):
        writer.writerow({'ImageId': i + 1, 'Label': prediction_conv[i]})
print("Write predictions finished")


sess.close()