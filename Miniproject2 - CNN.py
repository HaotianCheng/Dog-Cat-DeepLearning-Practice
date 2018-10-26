import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


TRAIN_DIR = 'C:\\Galaxy\\Tools\\Scripts\\DOGvsCAT\\all\\train'
TEST_DIR = 'C:\\Galaxy\\Tools\\Scripts\\DOGvsCAT\\all\\test'
IMG_SIZE = 64
LR = 1e-3

MODEL_NAME = 'DVC-{}-{}.model'.format(LR, '3conv')

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
        # img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


train_data = create_train_data()
test_data = process_test_data()

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], data_preprocessing=img_prep, data_augmentation=img_aug)

conv_1 = conv_2d(network, 32, 5, activation='relu', name='conv_1')
pool_1 = max_pool_2d(conv_1, 2)

conv_2 = conv_2d(pool_1, 64, 5, activation='relu', name='conv_2')
pool_2 = max_pool_2d(conv_2, 2)

conv_3 = conv_2d(pool_2, 128, 5, activation='relu', name='conv_3')
pool_3 = max_pool_2d(conv_3, 2)

conv_f = fully_connected(pool_3, 1024, activation='relu')
drop = dropout(conv_f, 0.8)

conv_F = fully_connected(drop, 2, activation='softmax')

optimise = regression(conv_F, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy')

model = tflearn.DNN(optimise, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-1000]
test = train_data[-1000:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X.astype('float64')
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_x = test_x.astype('float64')
test_y = [i[1] for i in test]

model.fit(X, Y, validation_set=(test_x, test_y), batch_size=500, n_epoch=23, run_id=MODEL_NAME, show_metric=True)

model.save(MODEL_NAME)

test_data = np.load('test_data.npy')

fig = plt.figure()

for num, data in enumerate(test_data[:50]):

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(10, 5, num + 1)
    orig = img_data
    DATA = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    DATA = DATA.astype('float64')

    model_out = model.predict([DATA])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

