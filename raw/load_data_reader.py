import tensorflow as tf
import pandas as pd

filename_queue = tf.train.string_input_producer(["dataset/iris_dummy.csv"])

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1.], [1.], [1.], [1.], [0], [0], [0]]
col1, col2, col3, col4, col5, col6, col7 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])
labels = tf.stack([col5, col6, col7])
features_batch, labels_batch = tf.train.shuffle_batch([ features, labels ], batch_size=10, capacity=20, min_after_dequeue=10)
# dataset = tf.contrib.data.Dataset.from_tensor_slices(dataset)
# dataset.make

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1200):
        # Retrieve a single instance:
        print(sess.run([ features_batch ]))
        # example = sess.run([features])

    coord.request_stop()
    coord.join(threads)