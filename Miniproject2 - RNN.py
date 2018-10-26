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

n_inputs = 64
max_time = 64
lstm_size = 300
n_classes = 2


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


x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE])
y = tf.placeholder(tf.float32, [None, 2])

weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


def RNN(X, weights, biases):
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights)+biases)
    return results


prediction = RNN(x, weights, biases)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
optimise = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()


train = train_data[:-500]
test = train_data[-500:]
batch_size = 500
n_batch = len(train)//batch_size

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        shuffle(train)
        # sess.run(tf.assign(LR, 0.001 * (0.95 ** epoch)))
        for i in range(n_batch):
            X = np.array([i[0] for i in train[batch_size*i:batch_size*(i+1)]])
            Y = [i[1] for i in train[batch_size*i:batch_size*(i+1)]]
            sess.run(optimise, feed_dict={x: X, y: Y})

        test_x = np.array([i[0] for i in test])
        test_y = [i[1] for i in test]
        test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
        LOSS = sess.run(cross_entropy, feed_dict={x: test_x, y: test_y})
        print('Iter' + str(i) + ',Testing Accuracy ' + str(test_acc)+',Loss '+str(LOSS))

