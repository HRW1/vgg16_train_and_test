import numpy as np
import tensorflow as tf
import os
import glob
from skimage import io, transform
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='0'

path = './picture/'
w = 224
h = 224
c = 3
classes = 2

def one_hot(labels,Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
    return one_hot_label

def normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image-mean))
    image = (image - mean)/np.sqrt(var)
    return image

def read_img(path):
    cate   = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs   = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the image: %s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            std_img = normalize(img)
            imgs.append(std_img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
data, label = read_img(path)

# print(data[0])
# plt.imshow(data[10])
# plt.show()

num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

one_hot_y = one_hot(label, 2)

ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
one_hot_y_t = one_hot_y[:s]

x_val   = data[:]
y_val   = label[:]
one_hot_y_v = one_hot_y[:]

def build_network(height, width, channel):
    parameters = []
    #RuntimeError: tf.placeholder() is not compatible with eager execution.
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, height, width, channel], name='input')
    one_hot_y = tf.compat.v1.placeholder(tf.int64, shape=[None, classes], name='labels_placeholder')

    # conv1_1
    with tf.name_scope('conv1_1') as scope:
        #AttributeError: module 'tensorflow' has no attribute 'truncated_normal'
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # pool1
    pool1 = tf.nn.max_pool(conv1_2,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool1')

    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv2_2
    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                    stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # pool2
    pool2 = tf.nn.max_pool(conv2_2,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool2')

    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # pool3
    pool3 = tf.nn.max_pool(conv3_3,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool3')

    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv4_3
    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # pool4
    pool4 = tf.nn.max_pool(conv4_3,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool4')

    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # pool5
    pool5 = tf.nn.max_pool(conv5_3,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool4')

    # fc1
    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc1w = tf.Variable(tf.random.truncated_normal([shape, 4096],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)
        parameters += [fc1w, fc1b]

    # fc2
    with tf.name_scope('fc2') as scope:
        fc2w = tf.Variable(tf.random.truncated_normal([4096, 4096],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=True, name='biases')
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l)
        parameters += [fc2w, fc2b]

    # fc3
    with tf.name_scope('fc3') as scope:
        fc3w = tf.Variable(tf.random.truncated_normal([4096, classes],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc3b = tf.Variable(tf.constant(1.0, shape=[classes], dtype=tf.float32),
                           trainable=True, name='biases')
        fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
        parameters += [fc3w, fc3b]

    finaloutput = tf.nn.softmax(fc3l, name="softmax")

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3l, labels=one_hot_y))
    optimize = tf.compat.v1.train.AdamOptimizer(0.001).minimize(cost)

    prediction_labels = tf.argmax(finaloutput, axis=1, name="output")

    correct_prediction = tf.equal(prediction_labels,tf.argmax(one_hot_y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return dict(
        x=x,
        one_hot_y=one_hot_y,
        optimize=optimize,
        accuracy=accuracy,
        finaloutput=finaloutput,
    )

def train_network(graph, batch_size, num_epochs, pb_file_path):
    input_queue = tf.compat.v1.train.slice_input_producer([x_train, y_train, one_hot_y_t], shuffle=False)
    img_batch, label_batch, one_hot_l_batch = tf.compat.v1.train.shuffle_batch(input_queue,
                                                    batch_size=4, capacity=2000,
                                                    min_after_dequeue=1000)
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        for epoch_index in range(num_epochs):
            for i in range(6):
                val, l, o_l = sess.run([img_batch, label_batch, one_hot_l_batch])
                print(l)
                _, acc, fo = sess.run([graph['optimize'], graph['accuracy'], graph['finaloutput']], feed_dict={
                    graph['x']: val,
                    graph['one_hot_y']: o_l
                })
                print("Epoch:[%4d] [%d / 6] accuracy:[%.8f]" % (epoch_index,i , acc))
                print(fo)

            _, acc = sess.run([graph['optimize'], graph['accuracy']], feed_dict={
                graph['x']: x_val,
                graph['one_hot_y']: one_hot_y_v
            })
            print("test accuracy: [%.8f]" % (acc))

            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
            with tf.compat.v1.gfile.FastGFile(pb_file_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
        coord.request_stop()
        coord.join(threads)

def main():
    batch_size = 12
    num_epochs = 50

    pb_file_path = "vggs.pb"

    g = build_network(height=224, width=224, channel=3)
    train_network(g, batch_size, num_epochs, pb_file_path)

main()