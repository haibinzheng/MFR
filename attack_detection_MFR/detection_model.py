import numpy as np
import keras

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

x_train=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/data_013510/x_train_013510.npy')
y_train=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/data_013510/y_train_013510.npy')
x_test=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/data_013510/x_test_013510.npy')
y_test=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/data_013510/y_test_013510.npy')

y_train=keras.utils.to_categorical(y_train,2)
y_test=keras.utils.to_categorical(y_test,2)

network = input_data(shape=[None,5])
network = fully_connected(network, 200, activation = 'relu')
network = fully_connected(network, 100, activation = 'relu')
network = fully_connected(network,2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0,tensorboard_dir='/home/Bear/Doublenet1/')

model.load('/home/Bear/attack_detection/model_detection/model_vgg-200000')

