import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv("dataset/iris.csv", delimiter=",")
proced_data = data.values

trainset = np.random.choice(150,135,replace=False)
traindata = proced_data[trainset,:]
testdata = np.delete(proced_data, trainset, axis=0)

with tf.name_scope("layer"):
    x = tf.placeholder(tf.float32, [None, 4])
    x = tf.identity(x, name="input")
    y_ = tf.placeholder(tf.float32, [None, 3])
    with tf.name_scope("weight"):
        W = tf.get_variable("W", [4, 3])
        tf.summary.histogram("weights", W)
    with tf.name_scope("bias"):
        b = tf.get_variable("b", [3])
        tf.summary.histogram("bias", b)
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    y = tf.identity(y, name="output")
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        tf.summary.scalar("loss", loss)
    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("log/")
writer.add_graph(sess.graph)

builder = tf.saved_model.builder.SavedModelBuilder("tmp2/")

print("Starting to train...")
for i in range(5000):
    xdata = traindata[:,0:4].reshape(-1, 4)
    ydata = pd.get_dummies(traindata[:,4]).values
    if i % 10 == 0:
        s = sess.run(merged, {x: xdata, y_:ydata})
        writer.add_summary(s, i)
    sess.run(train, {x: xdata, y_:ydata})

print("Saving the model...")
builder.add_meta_graph_and_variables(sess, ["foo-tag"])
builder.save()

print("Starting to evaluate...")
correct = tf.equal(tf.argmax(y_,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print("The accuracy is: %g" % sess.run(accuracy,{x: testdata[:,0:4].reshape(-1, 4), y_: pd.get_dummies(testdata[:,4]).values}))

sess = tf.Session()
meta_graph_def = tf.saved_model.loader.load(sess, ['foo-tag'], "tmp2/")

print([n.name for n in tf.get_default_graph().as_graph_def().node])

x = sess.graph.get_tensor_by_name('layer/input:0') # 也可以用 x = sess.graph.get_tensor_by_name('layer/Placeholder:0')
y = sess.graph.get_tensor_by_name('layer/output:0')
print(sess.run(y, feed_dict={x: testdata[:,0:4]}))
