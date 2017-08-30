import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv("dataset/iris.csv", delimiter=",")
proced_data = data.values

trainset = np.random.choice(150,135,replace=False)
traindata = proced_data[trainset,:]
testdata = np.delete(proced_data, trainset, axis=0)

with tf.name_scope("layer"):
    x_ = tf.placeholder(tf.float32, [None, 4])
    x = tf.nn.l2_normalize(x_, 0)
    y_ = tf.placeholder(tf.float32, [None, 3])
    with tf.name_scope("weight"):
        W = tf.Variable(tf.ones([4, 3]))
        tf.summary.histogram("weights", W)
    with tf.name_scope("bias"):
        b = tf.Variable(tf.ones([3]))
        tf.summary.histogram("bias", b)
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        tf.summary.scalar("loss", loss)
    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("log/")
writer.add_graph(sess.graph)

print("Start to train...")
for i in range(10000):
    xdata = traindata[:,0:4].reshape(-1, 4)
    ydata = pd.get_dummies(traindata[:,4]).values
    if i % 10 == 0:
        s = sess.run(merged, {x: xdata, y_:ydata})
        writer.add_summary(s, i)
    sess.run(train, {x: xdata, y_:ydata})

print("Start to evaluate...")
correct = tf.equal(tf.argmax(y_,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print("The accuracy is: %g" % sess.run(accuracy,{x: testdata[:,0:4].reshape(-1, 4), y_: pd.get_dummies(testdata[:,4]).values}))
