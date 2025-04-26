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
print('0000000000000000df_vgg16_vgg19000000000000000000000000')
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

    scoressss = []
    # for dududu in ['00_-50','00_-25','00_25','00_50']:
    # for dududu in [ 's1_-50', 's1_-25','s1', 's1_25', 's1_50']:
    # for dududu in [ 's2_-50', 's2_-25','s2', 's2_25', 's2_50']:
    # for dududu in [ 'x1_-50', 'x1_-25','x1', 'x1_25', 'x1_50']:
    # for dududu in [ 'x2_-50', 'x2_-25','x2', 'x2_25', 'x2_50']:
    # for dududu in [ 'y1_-50', 'y1_-25','y1', 'y1_25', 'y1_50']:
    # for dududu in [ 'y2_-50', 'y2_-25','y2', 'y2_25', 'y2_50']:
    # for dududu in [ 'z1_-50', 'z1_-25','z1', 'z1_25', 'z1_50']:
    # for dududu in [ 'z2_-50', 'z2_-25','z2', 'z2_25', 'z2_50']:
    # for dududu in ['rs20','rs24','rs28','rs36','rs40','rs44']:
    # for dududu in ['gv001m0','gv002m0','gv003m0','gv004m0','gv005m0']:
    # for dududu in ['s2_-50', 's2_-25','s2', 's2_25', 's2_50','x1_-50', 'x1_-25','x1', 'x1_25', 'x1_50',
    #                'x2_-50', 'x2_-25', 'x2', 'x2_25', 'x2_50','y1_-50', 'y1_-25','y1', 'y1_25', 'y1_50',
    #                'y2_-50', 'y2_-25', 'y2', 'y2_25', 'y2_50','z1_-50', 'z1_-25','z1', 'z1_25', 'z1_50',
    #                'z2_-50', 'z2_-25', 'z2', 'z2_25', 'z2_50','rs20','rs24','rs28','rs36','rs40','rs44',
    #                'gv001m0', 'gv002m0', 'gv003m0', 'gv004m0', 'gv005m0']:
    # for dududu in ['00','00_-50','00_-25','00_25','00_50',
    #     #                's1_-50', 's1_-25','s1', 's1_25', 's1_50','s2_-50', 's2_-25','s2', 's2_25', 's2_50',
    #     #                'x1_-50', 'x1_-25','x1', 'x1_25', 'x1_50','x2_-50', 'x2_-25', 'x2', 'x2_25', 'x2_50',
    #     #                'y1_-50', 'y1_-25','y1', 'y1_25', 'y1_50','y2_-50', 'y2_-25', 'y2', 'y2_25', 'y2_50',
    #     #                'z1_-50', 'z1_-25','z1', 'z1_25', 'z1_50','z2_-50', 'z2_-25', 'z2', 'z2_25', 'z2_50',
    #     #                'rs16','rs20','rs24','rs28','rs36','rs40','rs44','rs48',
    #     #                'gv001m0', 'gv002m0', 'gv003m0', 'gv004m0', 'gv005m0']:
    for dududu in ['00']:
        # Convert class vectors to binary class matrices.
        x_test0 = np.load('/home/Bear/attack_detection/deepfool_vgg16_cifar10/cifar10_all/x_test{}'.format(dududu) + '.npy')
        x_adtest0 = np.load('/home/Bear/attack_detection/deepfool_vgg16_cifar10/cifar10_all/x_adtest{}'.format(dududu) + '.npy')
        y_test0 = np.load('/home/Bear/attack_detection/deepfool_vgg16_cifar10/cifar10_all/y_test00.npy')
        length = len(x_test0)
        print(dududu, length)
        print(x_test0.shape)
        print(x_adtest0.shape)
        print(y_test0.shape)
        y_test01 = keras.utils.to_categorical(y_test0, num_classes)
        scores1 = model.evaluate(x_test0, y_test01, verbose=1)
        print('Test loss:', scores1[0])
        print('Test accuracy:', scores1[1])
        scores2 = model.evaluate(x_adtest0, y_test01, verbose=1)
        print('adTest loss:', scores2[0])
        print('adTest accuracy:', scores2[1])
        scoressss.append(scores1[1])
        scoressss.append(scores2[1])
        attack_name = attack_dir[args.attack_method]
        cl = []
        # 攻击后的类标
        y_adtest = []
        # 正常预测类标
        y_pred = []
        y_confidence = []
        y_adconfidence = []
        with foolbox.models.KerasModel(model, bounds=(0, 1), channel_axis=3) as model1:
            attack = args.attack_method(model1)
            for idx in range(0, length):
                pred_label = np.argmax(model1.predictions(x_test0[idx]))
                confidence = model1.predictions(x_test0[idx])

                image_name = str(idx)
                adv_label = np.argmax(model1.predictions(x_adtest0[idx]))
                adconfidence = model1.predictions(x_adtest0[idx])

                y_pred.append(pred_label)
                y_confidence.append(confidence)

                y_adtest.append(adv_label)
                y_adconfidence.append(adconfidence)

                print("name:{},reallabel:{},source label:{},adv label:{} ".format(image_name, y_test0[idx],
                                                                                  pred_label, adv_label))

        y_adtest = np.array(y_adtest)
        y_pred = np.array(y_pred)
        y_confidence = np.array(y_confidence)
        y_adconfidence = np.array(y_adconfidence)

        print('y_adtest', y_adtest.shape)
        print('y_pred', y_pred.shape)
        print('y_confidence', y_confidence.shape)
        print('y_adconfidence', y_adconfidence.shape)

        # np.save('/tmp/df_vgg16_vgg19/y_adtest{}'.format(dududu), y_adtest)
        # np.save('/tmp/df_vgg16_vgg19/y_pred{}'.format(dududu), y_pred)
        np.save('/tmp/df_vgg16_vgg19/y_confidence{}'.format(dududu),y_confidence)
        np.save('/tmp/df_vgg16_vgg19/y_adconfidence{}'.format(dududu),y_adconfidence)

    print('scoressss', scoressss)


if __name__ == '__main__':
    # parser.add_argument('-mp', '--model_path', default='model/inception_v3.ckpt')  # 选择模型

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # 不全部占满显存, 按需分配//
    config.gpu_options.per_process_gpu_memory_fraction = 0.05
    sess = tf.Session(config=config)# 设置session
    KTF.set_session(sess)

    tf.app.run()