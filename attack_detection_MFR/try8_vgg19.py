'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import foolbox
import argparse
from keras.initializers import he_normal
from keras import optimizers
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation,MaxPooling2D,Dropout,Flatten
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import os
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

keras.backend.set_learning_phase(0)
parser=argparse.ArgumentParser()

parser.add_argument('-am', '--attack_method', default=foolbox.attacks.DeepFoolAttack)  # 选择攻击方式，比如FGSM
parser.add_argument('-dp', '--data_path', default='imagenet')
parser.add_argument('-od', '--output_dir', default='samples')  # 选择攻击方式，比如FGSM
args = parser.parse_args()

kt=1
slim=tf.contrib.slim
attack_dir = {
foolbox.attacks.DeepFoolAttack:'deepfool',
foolbox.attacks.FGSM:'fgsm',
foolbox.attacks.LBFGSAttack:'lbfgsa',
foolbox.attacks.SaliencyMapAttack:'saliencymap',
foolbox.attacks.IterativeGradientAttack:'iterativegrad',
foolbox.attacks.MomentumIterativeAttack:'mi-fgsm',
foolbox.attacks.ProjectedGradientDescentAttack:'pgd',
foolbox.attacks.CarliniWagnerL2Attack:'CW2'
}
attack_list =(
foolbox.attacks.DeepFoolAttack,
foolbox.attacks.FGSM,
foolbox.attacks.LBFGSAttack,
foolbox.attacks.SaliencyMapAttack,
foolbox.attacks.IterativeGradientAttack,
foolbox.attacks.MomentumIterativeAttack,
foolbox.attacks.ProjectedGradientDescentAttack,
foolbox.attacks.CarliniWagnerL2Attack
)
# 打开ROOT_DIR的完整路径
ROOT_DIR = os.path.abspath('')
#打开绝对路径os.path.dirname(__file__)
def main(slef):
    num_classes = 10
    batch_size = 128
    epochs = 250
    iterations = 391
    dropout = 0.5
    weight_decay = 0.0001
    # log_filepath = r'./vgg19_retrain_logs/'
    subtract_pixel_mean = False

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize
    if subtract_pixel_mean:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = np.mean(x_train, axis=(0, 1, 2, 3))
        std = np.std(x_train, axis=(0, 1, 2, 3))
        x_train = (x_train - mean) / (std + 1e-7)
        x_test = (x_test - mean) / (std + 1e-7)

    else:
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    input_shape = x_train.shape[1:]

    def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    # WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    # filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')

    # build model
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block1_conv1', input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # model modification for cifar-10
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                    kernel_initializer=he_normal(), name='fc_cifa10'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay),
                    kernel_initializer=he_normal(), name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, kernel_regularizer=keras.regularizers.l2(weight_decay),
                    kernel_initializer=he_normal(), name='predictions_cifa10'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    # load pretrained weight from VGG19 by name
    # model.load_weights(filepath, by_name=True)

    # -------- optimizer setting -------- #
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    save_dir = os.path.join(os.getcwd(), '1_vgg19_saved_models')
    model_name = 'Vgg19_cifar10_trained_model.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)

    datagen.fit(x_train)


    model.load_weights('/home/Bear/attack_detection/1_vgg19_saved_models/Vgg19_cifar10_trained_model.h5')  # 模型文件路径

    # Convert class vectors to binary class matrices.
    x_test0=np.load('/home/Bear/attack_detection/New_cifar10_x(0,1),y1/x_test.npy')
    y_test0=np.load('/home/Bear/attack_detection/New_cifar10_x(0,1),y1/y_test.npy')
    y_test0 = keras.utils.to_categorical(y_test0, 10)
    print(x_test0.shape)
    print(y_test0.shape)

    attack_name = attack_dir[args.attack_method]
    cl = []
    x_testl = []
    y_testl = []
    x_adtest = []
    # 攻击后的类标
    y_adtest = []
    # 正常预测类标
    y_pred = []
    y_confidence = []
    y_adconfidence = []
    with foolbox.models.KerasModel(model, bounds=(0, 1), channel_axis=3) as model1:
        attack = args.attack_method(model1)
        for idx, image in enumerate(x_test0):
            pred_label = np.argmax(model1.predictions(image))
            confidence = model1.predictions(image)
            adversarial = attack(image, pred_label)
            if adversarial is None:
                print('adversarial is None')
                continue
            else:
                image_name = str(idx)
                adv_label = np.argmax(model1.predictions(adversarial))
                adconfidence = model1.predictions(adversarial)

                x_testl.append(image)
                y_testl.append(np.argmax(y_test0[idx]))
                y_pred.append(pred_label)
                y_confidence.append(confidence)

                x_adtest.append(adversarial)
                y_adtest.append(adv_label)
                y_adconfidence.append(adconfidence)
                cl.append(idx)
                print("name:{},reallabel:{},source label:{},adv label:{} ".format(image_name, np.argmax(y_test0[idx]),
                                                                                  pred_label, adv_label))

    x_testl = np.array(x_testl)
    y_testl = np.array(y_testl)
    x_adtest = np.array(x_adtest)
    y_adtest = np.array(y_adtest)
    y_pred = np.array(y_pred)
    y_confidence = np.array(y_confidence)
    y_adconfidence = np.array(y_adconfidence)
    cl=np.array(cl)

    print('x_test', x_testl.shape)
    print('y_test', y_testl.shape)
    print('x_adtest', x_adtest.shape)
    print('y_adtest', y_adtest.shape)
    print('y_pred', y_pred.shape)
    print('y_confidence', y_confidence.shape)
    print('y_adconfidence', y_adconfidence.shape)
    print('cl',cl.shape)


    np.save('/home/Bear/attack_detection/deepfool_vgg19_cifar10/orin/x_test',x_testl)
    np.save('/home/Bear/attack_detection/deepfool_vgg19_cifar10/orin/y_test', y_testl)
    np.save('/home/Bear/attack_detection/deepfool_vgg19_cifar10/orin/x_adtest',x_adtest)
    np.save('/home/Bear/attack_detection/deepfool_vgg19_cifar10/orin/y_adtest', y_adtest)
    np.save('/home/Bear/attack_detection/deepfool_vgg19_cifar10/orin/y_pred', y_pred)
    np.save('/home/Bear/attack_detection/deepfool_vgg19_cifar10/orin/y_confidence', y_confidence)
    np.save('/home/Bear/attack_detection/deepfool_vgg19_cifar10/orin/y_adconfidence', y_adconfidence)
    np.save('/home/Bear/attack_detection/deepfool_vgg19_cifar10/orin/cl',cl)


if __name__ == '__main__':
    # parser.add_argument('-mp', '--model_path', default='model/inception_v3.ckpt')  # 选择模型

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 不全部占满显存, 按需分配//
    sess = tf.Session(config=config)# 设置session
    KTF.set_session(sess)

    tf.app.run()