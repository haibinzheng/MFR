import numpy as np
import keras
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
td=5
x_train=np.load('/home/Bear/attack_detection/cw_vgg16_cifar10/cifar10_all/5/x_train.npy')
y_train=np.load('/home/Bear/attack_detection/cw_vgg16_cifar10/cifar10_all/5/y_train.npy')
x_test =np.load('/home/Bear/attack_detection/cw_vgg16_cifar10/cifar10_all/5/x_test.npy')
y_test =np.load('/home/Bear/attack_detection/cw_vgg16_cifar10/cifar10_all/5/y_test.npy')
# X=np.load('/home/Bear/attack_detection/WorB/X_cc_coda5.npy')
# Y=np.load('/home/Bear/attack_detection/WorB/Y_cc_coda5.npy')
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

y_train=keras.utils.to_categorical(y_train,2)
y_test=keras.utils.to_categorical(y_test,2)

network = input_data(shape=[None,td,10])
network = fully_connected(network, 200, activation = 'relu')
network = fully_connected(network,2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.00001)
print('wb_cc5_0325{}'.format(str(td)))
# Training
model = tflearn.DNN(network, checkpoint_path='model_cnn',
                    max_checkpoints=1, tensorboard_verbose=0,tensorboard_dir='/home/Bear/attack_detection')
# model.fit(x_train, y_train, n_epoch=500, validation_set=(x_test, y_test),shuffle=True,
#           show_metric=True, batch_size=15, snapshot_step=500,
#           snapshot_epoch=False, run_id='cw')
# model.save('wb_cc5_0325{}'.format(str(td)))
model.load('/tmp/attack_detection_cifar10model/model_5_1208/model_cnn-352000')
y_pred=model.predict(x_test)
print(y_test.shape)
print(y_pred.shape)
print(y_test[0],'*',y_pred[0])
y_test=np.argmax(y_test,axis=1)
y_pred=np.argmax(y_pred,axis=1)
print(y_test.shape)
print(y_pred.shape)
print(y_test[0],'*',y_pred[0])
print(accuracy_score(y_test,y_pred))




