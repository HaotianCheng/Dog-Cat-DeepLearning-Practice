import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = 'C:\\Galaxy\\Tools\\Scripts\\DOGvsCAT\\all\\train'
TEST_DIR = 'C:\\Galaxy\\Tools\\Scripts\\DOGvsCAT\\all\\test'
IMG_SIZE = 64
LR = tf.Variable(0.0001, dtype=tf.float32)


def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.reshape(-1, IMG_SIZE*IMG_SIZE)
        img = img[0]
        img = img.astype(np.float32)
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.reshape(-1, IMG_SIZE * IMG_SIZE)
        img = img[0]
        img = img.astype(np.float32)
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


train_data = create_train_data()
# test_data = process_test_data()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE])
y = tf.placeholder(tf.float32, [None, 2])

x_image = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, 1])

w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_conv3 = weight_variable([5, 5, 64, 32])
b_conv3 = bias_variable([32])

h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3)+b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# w_conv4 = weight_variable([5, 5, 32, 32])
# b_conv4 = bias_variable([32])
#
# h_conv4 = tf.nn.relu(conv2d(h_pool3, w_conv4)+b_conv4)
# h_pool4 = max_pool_2x2(h_conv4)

w_fcl = weight_variable([8*8*32, 1024])
b_fcl = bias_variable([1024])

h_pool_flat = tf.reshape(h_pool3, [-1, 8*8*32])
h_fcl = tf.nn.relu(tf.matmul(h_pool_flat, w_fcl)+b_fcl)

keep_prob = tf.placeholder(tf.float32)
h_fcl_drop = tf.nn.dropout(h_fcl, keep_prob)

w_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

prediction = tf.nn.softmax(tf.matmul(h_fcl_drop, w_fc2)+b_fc2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
train_step = tf.train.AdagradOptimizer(LR).minimize(loss)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train = train_data[:-500]
test = train_data[-500:]

batch_size = 200
n_batch = len(train)//batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        # sess.run(tf.assign(LR, 0.001 * (0.95 ** epoch)))
        # for i in n_batch:
        shuffle(train)
        X = [i[0] for i in train[:batch_size]]
        Y = [i[1] for i in train[:batch_size]]
        sess.run(train_step, feed_dict={x: X, y: Y, keep_prob: 1.0})

        test_x = [i[0] for i in test]
        test_y = [i[1] for i in test]
        test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
        LOSS = sess.run(loss, feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
        print('Iter' + str(epoch) + ',Testing Accuracy ' + str(test_acc)+',Loss '+str(LOSS))

