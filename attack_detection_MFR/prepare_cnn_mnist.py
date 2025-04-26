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
from keras.datasets import mnist


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
    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if KTF.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print(np.min(x_test))
    print(np.max(x_test))
    print(np.min(x_train))
    print(np.max(x_train))

    print(y_train.shape)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.load_weights('/home/Bear/attack_detection/1_mnist_cnn_model/CNN_mnist_trained_model')  # 模型文件路径

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
    print('deepfool_cnn_mnist')
    for dududu in ['00_-50', '00_-25', '00_25', '00_50', 's1_-50', 's1_-25', 's1', 's1_25', 's1_50',
                   's2_-50', 's2_-25', 's2', 's2_25', 's2_50', 'x1_-50', 'x1_-25', 'x1', 'x1_25', 'x1_50',
                   'x2_-50', 'x2_-25', 'x2', 'x2_25', 'x2_50', 'y1_-50', 'y1_-25', 'y1', 'y1_25', 'y1_50',
                   'y2_-50', 'y2_-25', 'y2', 'y2_25', 'y2_50', 'z1_-50', 'z1_-25', 'z1', 'z1_25', 'z1_50',
                   'z2_-50', 'z2_-25', 'z2', 'z2_25', 'z2_50', 'rs16','rs20', 'rs24', 'rs32', 'rs36', 'rs40', 'rs44',
                   'gv001m0', 'gv002m0', 'gv003m0', 'gv004m0', 'gv005m0']:

        # Convert class vectors to binary class matrices.
        x_test0 = np.load('/home/Bear/attack_detection/deepfool_cnn_mnist/mnist_all/x_test{}'.format(dududu) + '.npy')
        x_adtest0 = np.load('/home/Bear/attack_detection/deepfool_cnn_mnist/mnist_all/x_adtest{}'.format(dududu) + '.npy')
        y_test0 = np.load('/home/Bear/attack_detection/deepfool_cnn_mnist/mnist_all/y_test00.npy')
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
        with foolbox.models.KerasModel(model, bounds=(0, 1), channel_axis=1) as model1:
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

        # np.save('/home/Bear/attack_detection/deepfool_cnn_mnist/mnist_all/y_adtest{}'.format(dududu), y_adtest)
        # np.save('/home/Bear/attack_detection/deepfool_cnn_mnist/mnist_all/y_pred{}'.format(dududu), y_pred)
        # np.save('/home/Bear/attack_detection/deepfool_cnn_mnist/mnist_all/y_confidence{}'.format(dududu),y_confidence)
        # np.save('/home/Bear/attack_detection/deepfool_cnn_mnist/mnist_all/y_adconfidence{}'.format(dududu),y_adconfidence)

    print('scoressss', scoressss)

if __name__ == '__main__':
    # parser.add_argument('-mp', '--model_path', default='model/inception_v3.ckpt')  # 选择模型

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 不全部占满显存, 按需分配//
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    #指定分配30%空间
    sess = tf.Session(config=config)# 设置session
    KTF.set_session(sess)

    tf.app.run()