import os
import shutil
import sys
import random
import errno
import keras
from keras import models, layers, optimizers
import numpy as np
import keras
import cv2
from keras import backend as K
from keras.layers import Input, Dense, Dropout
from keras.applications import vgg16, resnet50, vgg19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import load_model
from keras_preprocessing import image
from keras.utils import to_categorical
import shutil
import foolbox
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

#%% load data
images = np.load('/public/zly/DGAN/CUB_200_2011/images_npy/trian_images_10_classes.npy')
labels = np.load('/public/zly/DGAN/CUB_200_2011/images_npy/trian_labels_10_classes.npy')
model = load_model('bird_model_vgg16_224_cub-200-2011_10_classes.h5')
#%%
fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))
attacks = foolbox.attacks.DeepFoolAttack(fmodel)
advs = []
for i in range(images.shape[0]):
    x = images[i:i+1]
    truth = labels[i:i+1]
    truth = np.argmax(truth, axis=1)
    x_adv = attacks(x, np.array(truth))
    print('Running Loop:{}'.format(i))
    if np.argmax(model.predict(x_adv)) != np.argmax(truth):
        advs.append(x_adv)
ad = np.array(advs).reshape(len(advs), 224, 224, 3)
np.save('/public/zly/DGAN/CUB_200_2011/attacks/DeepFool/advs.npy', ad)
#%%
robust_class = {"0": 4, "1": 4, "2": 1, "3": 8, "4": 9, "5": 8, "6": 0, "7": 8, "8": 4, "9": 4}
frajial_class = {"0": 2, "1": 9, "2": 6, "3": 2, "4": 2, "5": 2, "6": 2, "7": 2, "8": 0, "9": 1}
advs = np.load('/public/zly/DGAN/CUB_200_2011/attacks/DeepFool/advs.npy')
#%%
y_robust = []
y_fragile = []
for i in range(300):
    y_ture = np.argmax(labels[i:i+1])
    y_robust.append(to_categorical(robust_class.get('{}'.format(y_ture)), 10))
    y_fragile.append(to_categorical(frajial_class.get('{}'.format(y_ture)), 10))
y_fragile = np.array(y_fragile).reshape(len(y_fragile), -1)
y_robust = np.array(y_robust).reshape(len(y_robust), -1)

#%%
get_embedding_features = K.function(inputs=[model.input], outputs=[model.layers[-2].output])
features = []
for i in range(300):
    features.append(get_embedding_features([images[i:i+1]])[0])
np.save('/public/zly/DGAN/MFR_jin/CIFAR10_CLASS/vgg19_features_cub.npy', np.array(features))
#%%%
features = np.load('/public/zly/DGAN/MFR_jin/CIFAR10_CLASS/vgg19_features_cub.npy').reshape(300, -1)
#%%
C_core = 1
def loss_core(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true, y_pred) + C_core * keras.losses.categorical_crossentropy(y_robust, keras.layers.multiply([1 - y_true, y_pred]))
def core_model(input_shape):
    x = Dense(4096, name='fc1', activation='tanh')(input_shape)
    x = Dense(1024, name='fc2', activation='tanh')(x)
    x = Dense(512, name='fc3', activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, name='predictions_cifa10', activation='softmax')(x)
    model = Model(input_shape, x)
    model.compile(loss=loss_core, optimizer='adam', metrics=['accuracy'])
    return model
#%%
C_coa = 1
def loss_coa(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true, y_pred) + C_coa * keras.losses.categorical_crossentropy(y_fragile, keras.layers.multiply([1 - y_true, y_pred]))
def coa_model(input_shape):
    x = Dense(4096, name='fc1', activation='tanh')(input_shape)
    x = Dense(1024, name='fc2', activation='tanh')(x)
    x = Dense(512, name='fc3', activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, name='predictions_cifa10', activation='softmax')(x)
    model = Model(input_shape, x)
    model.compile(loss=loss_core, optimizer='adam', metrics=['accuracy'])
    return model
#%%
y_train = labels
model_core = core_model(input_shape=Input(shape=features.shape[1:]))
model_core.fit(features, y_train, batch_size=300, epochs=40)
model_core.save('/public/zly/DGAN/MFR_jin/CUB_models/vgg19_core.h5')
#%%
y_train = labels
model_coa = coa_model(input_shape=Input(shape=features.shape[1:]))
model_coa.fit(features, y_train, batch_size=300, epochs=40)
model_coa.save('/public/zly/DGAN/MFR_jin/CUB_models/vgg19_coa.h5')
#%%
denoise_img = []
for i in range(300):
    k = 0.5 # 2 # 降噪系数
    x = images[i:i+1]
    H, W, C = np.shape(x[0])
    img_big = cv2.resize(x[0], dsize=(int(W * k), int(H * k)))
    # img_small = cv2.resize()
    denoise_img.append(cv2.resize(img_big, (W, H)) - x[0])
denoise_img = np.array(denoise_img)
#%%
denoise_features = []
for i in range(300):
    denoise_features.append(get_embedding_features([denoise_img[i:i+1]])[0])
np.save('/public/zly/DGAN/MFR_jin/CIFAR10_CLASS/vgg19_features_cub_denoise.npy', np.array(denoise_features))
#%%%
denoise_features = np.load('/public/zly/DGAN/MFR_jin/CIFAR10_CLASS/vgg19_features_cub_denoise.npy').reshape(300, -1)
#%%
def denoise_model(input_shape):
    x = Dense(4096, name='fc1', activation='tanh')(input_shape)
    x = Dense(1024, name='fc2', activation='tanh')(x)
    x = Dense(512, name='fc3', activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, name='predictions_cifa10', activation='softmax')(x)
    model = Model(input_shape, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#%%
y_train = labels
model_denoise = denoise_model(input_shape=Input(shape=denoise_features.shape[1:]))
model_denoise.fit(denoise_features, y_train, batch_size=32, epochs=80)
model_denoise.save('/public/zly/DGAN/MFR_jin/CUB_models/vgg19_denoise.h5')
#%% detection model
def detection_model(input_shape):
    x = Dense(512*3, name='fc1', activation='relu')(input_shape)
    x = Dense(2, name='predictions', activation='softmax')(x)
    model = Model(input_shape, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#%%
get_embedding_core_features = K.function(inputs=[model_core.input], outputs=[model_core.get_layer('fc3').output])
get_embedding_coa_features = K.function(inputs=[model_coa.input], outputs=[model_coa.get_layer('fc3').output])
get_embedding_denoise_features = K.function(inputs=[model_denoise.input], outputs=[model_denoise.get_layer('fc3').output])
core_features_fc = []
coa_features_fc = []
denoise_features_fc = []
for i in range(300):
    core_features_fc.append(get_embedding_core_features([features[i:i+1]])[0])
    coa_features_fc.append(get_embedding_coa_features([features[i:i + 1]])[0])
    denoise_features_fc.append(get_embedding_denoise_features([features[i:i + 1]])[0])
#%%
features_adv = []
for i in range(250):
    features_adv.append(get_embedding_features([advs[i:i+1]])[0])
features_adv = np.array(features_adv).reshape(250, -1)
#%%
adv_core_features_fc = []
adv_coa_features_fc = []
adv_denoise_features_fc = []
for i in range(250):
    adv_core_features_fc.append(get_embedding_core_features([features_adv[i:i+1]])[0])
    adv_coa_features_fc.append(get_embedding_coa_features([features_adv[i:i + 1]])[0])
    adv_denoise_features_fc.append(get_embedding_denoise_features([features_adv[i:i + 1]])[0])
#%%
core_features_fc = np.array(core_features_fc).reshape(300, -1)
coa_features_fc = np.array(coa_features_fc).reshape(300, -1)
denoise_features_fc = np.array(denoise_features_fc).reshape(300, -1)
adv_coa_features_fc = np.array(adv_coa_features_fc).reshape(250, -1)
adv_core_features_fc = np.array(adv_core_features_fc).reshape(250, -1)
adv_denoise_features_fc = np.array(adv_denoise_features_fc).reshape(250, -1)
#%%
detection_train = []
detection_label = []
for i in range(300):
    detection_tmp = np.hstack((core_features_fc[i:i+1], coa_features_fc[i:i+1]))
    detection_tmp = np.hstack((detection_tmp, denoise_features_fc[i:i+1]))
    detection_train.append(detection_tmp)
    detection_label.append(to_categorical(0, 2))
#%%
detection_train_adv = []
detection_label_adv = []
for i in range(250):
    detection_tmp = np.hstack((adv_core_features_fc[i:i+1], adv_coa_features_fc[i:i+1]))
    detection_tmp = np.hstack((detection_tmp, adv_denoise_features_fc[i:i+1]))
    detection_train_adv.append(detection_tmp)
    detection_label_adv.append(to_categorical(1, 2))
#%%
detection_train = np.array(detection_train).reshape(300, -1)
detection_label = np.array(detection_label).reshape(300, -1)
detection_train_adv = np.array(detection_train_adv).reshape(250, -1)
detection_label_adv = np.array(detection_label_adv).reshape(250, -1)
#%%
x_train = np.vstack((detection_train[:200], detection_train_adv[:100]))
y_label = np.vstack((detection_label[:200], detection_label_adv[:100]))
x_test = np.vstack((detection_train[200:300], detection_train_adv[100:250]))
y_test = np.vstack((detection_label[200:300], detection_label_adv[100:250]))
#%%
model_detection = detection_model(input_shape=Input(shape=x_train.shape[1:]))
model_detection.fit(x_train, y_label, batch_size=32, epochs=30)
model_detection.save('/public/zly/DGAN/MFR_jin/CUB_models/vgg19_detection.h5')
scores = model_detection.evaluate(detection_train_adv[:150], detection_label_adv[:150])