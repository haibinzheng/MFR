'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import os
import keras
import foolbox
import scipy.misc
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import numpy as np
import argparse

keras.backend.set_learning_phase(0)
parser=argparse.ArgumentParser()

parser.add_argument('-am', '--attack_method', default=foolbox.attacks.DeepFoolAttack)  # 选择攻击方式，比如FGSM
parser.add_argument('-dp', '--data_path', default='imagenet')
parser.add_argument('-od', '--output_dir', default='samples')  # 选择攻击方式，比如FGSM
args = parser.parse_args()
print('000000000000df_vgg16_cnn0000000000000000000')
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

    batch_size = 32
    num_classes = 10
    epochs = 100
    data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), '1_cnn_saved_models')
    model_name = 'keras_cifar10_trained_model.h5'

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)


    model.load_weights('/home/Bear/attack_detection/1_cnn_saved_models/keras_cifar10_trained_model.h5')  # 模型文件路径

    scoressss = []
    # for dududu in ['00_-50','00_-25','00_25','00_50','gv001m0','gv002m0','gv003m0','gv004m0','gv005m0']:
    # for dududu in [ 's1_-50', 's1_-25','s1', 's1_25', 's1_50']:
    # for dududu in [ 's2_-50', 's2_-25','s2', 's2_25', 's2_50']:
    # for dududu in [ 'x1_-50', 'x1_-25','x1', 'x1_25', 'x1_50']:
    # for dududu in [ 'x2_-50', 'x2_-25','x2', 'x2_25', 'x2_50']:
    # for dududu in [ 'y1_-50', 'y1_-25','y1', 'y1_25', 'y1_50']:
    # for dududu in [ 'y2_-50', 'y2_-25','y2', 'y2_25', 'y2_50']:
    # for dududu in [ 'z1_-50', 'z1_-25','z1', 'z1_25', 'z1_50']:
    # for dududu in ['z2_-50', 'z2_-25', 'z2', 'z2_25', 'z2_50']:
    # for dududu in ['rs16','rs20','rs24','rs28','rs36','rs40','rs44','rs48']:
    # for dududu in ['gv001m0','gv002m0','gv003m0','gv004m0','gv005m0']:
    # for dududu in ['00_-50','00_-25','00_25','00_50',
    #                's1_-50', 's1_-25','s1', 's1_25', 's1_50','s2_-50', 's2_-25','s2', 's2_25', 's2_50',
    #                'x1_-50', 'x1_-25','x1', 'x1_25', 'x1_50','x2_-50', 'x2_-25', 'x2', 'x2_25', 'x2_50',
    #                'y1_-50', 'y1_-25','y1', 'y1_25', 'y1_50','y2_-50', 'y2_-25', 'y2', 'y2_25', 'y2_50',
    #                'z1_-50', 'z1_-25','z1', 'z1_25', 'z1_50','z2_-50', 'z2_-25', 'z2', 'z2_25', 'z2_50',
    #                'rs16','rs20','rs24','rs28','rs36','rs40','rs44','rs48',
    #                'gv001m0', 'gv002m0', 'gv003m0', 'gv004m0', 'gv005m0']:        # Convert class vectors to binary class matrices.
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

                # print("name:{},reallabel:{},source label:{},adv label:{} ".format(image_name, y_test0[idx],
                #                                                                   pred_label, adv_label))

        y_adtest = np.array(y_adtest)
        y_pred = np.array(y_pred)
        y_confidence = np.array(y_confidence)
        y_adconfidence = np.array(y_adconfidence)

        print('y_adtest', y_adtest.shape)
        print('y_pred', y_pred.shape)
        print('y_confidence', y_confidence.shape)
        print('y_adconfidence', y_adconfidence.shape)

        # np.save('/home/Bear/attack_detection/cw_resnet_cnn/cifar10_all/y_adtest{}'.format(dududu), y_adtest)
        # np.save('/home/Bear/attack_detection/cw_resnet_cnn/cifar10_all/y_pred{}'.format(dududu), y_pred)
        np.save('/tmp/df_vgg16_cnn/y_confidence{}'.format(dududu),y_confidence)
        np.save('/tmp/df_vgg16_cnn/y_adconfidence{}'.format(dududu),y_adconfidence)

    print('scoressss', scoressss)

if __name__ == '__main__':
    # parser.add_argument('-mp', '--model_path', default='model/inception_v3.ckpt')  # 选择模型

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # 不全部占满显存, 按需分配//
    config.gpu_options.per_process_gpu_memory_fraction = 0.05
    #指定分配30%空间
    sess = tf.Session(config=config)# 设置session
    KTF.set_session(sess)

    tf.app.run()