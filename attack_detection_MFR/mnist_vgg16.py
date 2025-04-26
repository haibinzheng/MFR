from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
import cv2
from keras import datasets

from keras.applications.vgg16 import VGG16

from keras.optimizers import SGD

from keras.datasets import mnist

import numpy as np

# 迁移学习 使用VGG16进行手写数字识别
# 只迁移网络结构，不迁移权重

model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model = Flatten(name='Flatten')(model_vgg.output)
moel = Dense(10, activation='softmax')(model)

model_vgg_mnist = Model(inputs=model_vgg.input, outputs=model, name='vgg16')
model_vgg_mnist.summary()

# 迁移学习；网络结构与权重
#

ishape = 224
model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(ishape, ishape, 3))
for layers in model_vgg.layers:
    layers.trainable = False

model = Flatten()(model_vgg.output)
model = Dense(10, activation='softmax')(model)
model_vgg_mnist_pretrain = Model(inputs=model_vgg.input, outputs=model, name='vgg16_pretrain')
model_vgg_mnist_pretrain.summary()
# ==============================================================================
# Total params: 14,965,578.0
# Trainable params: 250,890.0
# Non-trainable params: 14,714,688.0
# ==============================================================================


sgd = SGD(lr=0.05, decay=1e-5)
model_vgg_mnist_pretrain.compile(optimizer=sgd, loss='categorical_crossentropy',
                                 metrics=['accuracy'])

# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# 转成VGG16需要的格式
# X_train = [cv2.cvtColor(cv2.resize(i, (ishape, ishape)), cv2.COLOR_GRAY2BGR) for i in X_train]
# X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')
#
# X_test = [cv2.cvtColor(cv2.resize(i, (ishape, ishape)), cv2.COLOR_GRAY2BGR) for i in X_test]
# X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')

# 预处理
# X_train.shape
# X_test.shape
#
# X_train /= X_train / 255
# X_test /= X_test / 255
X_train = np.load('/home/Bear/attack_detection/dataset_mnist/x_train.npy')
y_train = np.load('/home/Bear/attack_detection/dataset_mnist/y_train.npy')
X_test = np.load('/home/Bear/attack_detection/dataset_mnist/x_test.npy')
y_test = np.load('/home/Bear/attack_detection/dataset_mnist/y_test.npy')

np.where(X_train[0] != 0)


# 哑编码
def train_y(y):
    y_one = np.zeros(10)
    y_one[y] = 1
    return y_one


y_train_one = np.array([train_y(y_train[i]) for i in range(len(y_train))])
y_test_one = np.array([train_y(y_test[i]) for i in range(len(y_test))])

# 模型训练
model_vgg_mnist_pretrain.fit(X_train, y_train_one, validation_data=(X_test, y_test_one),
                             epochs=200, batch_size=128)

model_vgg_mnist_pretrain.save('1_vgg16_mnist_saved_models')