import numpy as np
import keras
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

x_train=np.load('/home/Bear/attack_detection/WorB/x_train18.npy')
y_train=np.load('/home/Bear/attack_detection/WorB/y_train18.npy')
x_test =np.load('/home/Bear/attack_detection/WorB/x_test18.npy')
y_test =np.load('/home/Bear/attack_detection/WorB/y_test18.npy')


y_train=keras.utils.to_categorical(y_train,2)
y_test=keras.utils.to_categorical(y_test,2)

network = input_data(shape=[None,18,10])
network = fully_connected(network, 200, activation = 'relu')
network = fully_connected(network,2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.00001)
print('WorB')
# Training
model = tflearn.DNN(network, checkpoint_path='model_cnn',
                    max_checkpoints=1, tensorboard_verbose=0,tensorboard_dir='/home/Bear/attack_detection')
model.fit(x_train, y_train, n_epoch=500, validation_set=(x_test, y_test),shuffle=True,
          show_metric=True, batch_size=15, snapshot_step=500,
          snapshot_epoch=False, run_id='cw')
model.save('WorB18')


