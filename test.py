import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1,28,28,1])

with tf.name_scope("convolutional_layer"):
    W_conv1 = weight_variable([5,5,1,32])
    tf.summary.histogram("Convolutional layer weights", W_conv1)
    b_conv1 = bias_variable([32])
    tf.summary.histogram("Convolutional layer bias", b_conv1)
    conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)
    max1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    tf.summary.histogram("Max layer output", W_conv1)

with tf.name_scope("full_connected_layer"):
    W_fc = weight_variable([14 * 14 * 32, 1024])
    b_fc = bias_variable([1024])
    max1_flat = tf.reshape(max1, [-1, 14*14*32])
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
for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i%5 == 0:
        s = sess.run(merged, {x:batch[0], y_:batch[1]})
        writer.add_summary(s, i)
    sess.run(train, {x:batch[0], y_:batch[1]})

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(ro,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))