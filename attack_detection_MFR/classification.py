import numpy as np
import keras
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d

x_train=np.load('/home/Bear/attack_detection/w_b_fgsm_GBA_resnet/x_train.npy')
y_train=np.load('/home/Bear/attack_detection/w_b_fgsm_GBA_resnet/y_train.npy')
x_test =np.load('/home/Bear/attack_detection/w_b_fgsm_GBA_resnet/x_test.npy')
y_test =np.load('/home/Bear/attack_detection/w_b_fgsm_GBA_resnet/y_test.npy')


y_train=keras.utils.to_categorical(y_train,3)
y_test=keras.utils.to_categorical(y_test,3)

network = input_data(shape=[None,5,10])

# network = conv_1d(network, 64, 3, activation='relu', name='network')
# network = conv_1d(network, 64, 3, activation='relu', name='network')
# network = max_pool_1d(network, 2, strides=2, name = 'network')
#
# network = conv_1d(network, 128, 3, activation='relu', name='network')
# network = conv_1d(network, 128, 3, activation='relu', name='network')
# network = max_pool_1d(network, 2, strides=2, name = 'network')
#
# network = conv_1d(network, 256, 3, activation='relu', name='network')
# network = conv_1d(network, 256, 3, activation='relu', name='network')
# network = max_pool_1d(network, 2, strides=2, name = 'network')

network = fully_connected(network, 200, activation = 'relu')
network=dropout(network,0.7)
network = fully_connected(network, 100, activation = 'relu')
network = fully_connected(network,3, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.00001)
print('dim=197144')
# Training
model = tflearn.DNN(network, checkpoint_path='model_cnn',
                    max_checkpoints=1, tensorboard_verbose=0,tensorboard_dir='/home/Bear/attack_detection')
model.fit(x_train, y_train, n_epoch=500, validation_set=(x_test, y_test),shuffle=True,
          show_metric=True, batch_size=15, snapshot_step=500,
          snapshot_epoch=False, run_id='cw')
model.save('/home/Bear/attack_detection/classify/cwfgsmdf_vgg16/cwfgsmdf_vgg16_500_CNN_01242126')


