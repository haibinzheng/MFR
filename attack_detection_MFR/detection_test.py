import numpy as np
import keras
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf


x_train=np.load('/home/Bear/attack_detection/a good detector/x_train_all.npy')
y_train=np.load('/home/Bear/attack_detection/a good detector/y_train_all.npy')
x_test =np.load('/home/Bear/attack_detection/a good detector/x_test_all.npy')
y_test =np.load('/home/Bear/attack_detection/a good detector/y_test_all.npy')


y_train=keras.utils.to_categorical(y_train,2)
y_test=keras.utils.to_categorical(y_test,2)

network = input_data(shape=[None,5,10])
network = fully_connected(network, 200, activation = 'relu')
network = fully_connected(network,2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.00001)
# Training
model = tflearn.DNN(network, checkpoint_path='model_cnn',
                    max_checkpoints=1, tensorboard_verbose=0,tensorboard_dir='/home/Bear/attack_detection')
# model.fit(x_train, y_train, n_epoch=500, validation_set=(x_test, y_test),shuffle=True,
#           show_metric=True, batch_size=15, snapshot_step=500,
#           snapshot_epoch=False, run_id='cw')
# model.save('agooddetector_all')
x0=np.load('/home/Bear/attack_detection/cw_vgg16_cifar10/cifar10_all/5/x_test.npy')
y0=np.load('/home/Bear/attack_detection/cw_vgg16_cifar10/cifar10_all/5/y_test.npy')
y0=keras.utils.to_categorical(y0,2)
print(x0.shape)
print(y0.shape)
model.load('/home/Bear/attack_detection/agooddetector_all')
score = model.evaluate(x0,y0,batch_size=128)
print(score)


