import numpy as np
import keras
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

x_train=np.load('/home/Bear/attack_detection/cw2_cifar10/cw2_con0/cw2_con0_56/x_train.npy')
y_train=np.load('/home/Bear/attack_detection/cw2_cifar10/cw2_con0/cw2_con0_56/y_train.npy')
x_test=np.load('/home/Bear/attack_detection/cw2_cifar10/cw2_con0/cw2_con0_56/x_test.npy')
y_test=np.load('/home/Bear/attack_detection/cw2_cifar10/cw2_con0/cw2_con0_56/y_test.npy')


y_train=keras.utils.to_categorical(y_train,2)
y_test=keras.utils.to_categorical(y_test,2)

network = input_data(shape=[None,45,10])

network = fully_connected(network, 200, activation = 'relu')
network = fully_connected(network,2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0,tensorboard_dir='/home/Bear/attack_detection')

model.fit(x_train, y_train, n_epoch=200, validation_set=(x_test, y_test),shuffle=True,
          show_metric=True, batch_size=10, snapshot_step=500,
          snapshot_epoch=False, run_id='data')

