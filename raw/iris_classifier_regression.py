import tensorflow as tf
import pandas
import numpy as np

data = pandas.read_csv("dataset/iris.csv", delimiter=",")
proced_data = data.values

trainset = np.random.choice(150,135,replace=False)
traindata = proced_data[trainset,:]
testdata = np.delete(proced_data, trainset, axis=0)

with tf.name_scope("layer"):
    x = tf.placeholder(tf.float32, [None, 4])
    y_ = tf.placeholder(tf.float32, [None, 1])
    with tf.name_scope("weight"):
        W = tf.Variable(tf.ones([4, 1]))
        tf.summary.histogram("weights", W)
    with tf.name_scope("bias"):
        b = tf.Variable(tf.ones([1]))
        tf.summary.histogram("bias", b)
    y = tf.matmul(x,W) + b
    with tf.name_scope("logg"):
        loss = tf.reduce_sum(tf.pow(y - y_, 2))/proced_data.shape[0]
        tf.summary.scalar("loss", loss)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("log/")
writer.add_graph(sess.graph)

for i in range(1000):
    xdata = traindata[:,0:4].reshape(-1, 4)
    ydata = traindata[:,4].reshape(-1, 1)
    if i % 10 == 0:
        s = sess.run(merged, {x: xdata, y_:ydata})
        writer.add_summary(s, i)
    sess.run(train, {x: xdata, y_:ydata})

correct = tf.equal(tf.round(y) - y_,0)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print("The accuracy is: %g" % sess.run(accuracy,{x: testdata[:,0:4].reshape(-1, 4), y_: testdata[:,4].reshape(-1, 1)}))
