from __future__ import print_function
import tensorflow as tf
import foolbox
import argparse
import numpy as np
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation,MaxPooling2D,Dropout,Flatten
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import os
from keras.datasets import cifar10
from keras.models import Sequential
import keras.backend.tensorflow_backend as KTF
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

keras.backend.set_learning_phase(0)
parser=argparse.ArgumentParser()

parser.add_argument('-am', '--attack_method', default=foolbox.attacks.FGSM)  # 选择攻击方式，比如FGSM
parser.add_argument('-dp', '--data_path', default='imagenet')
parser.add_argument('-od', '--output_dir', default='samples')  # 选择攻击方式，比如FGSM
args = parser.parse_args()

print('000000000000000000000000000000000000000000000000000000000000')
print('FGSM')

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
foolbox.attacks.CarliniWagnerL2Attack:'CW2',
foolbox.attacks.GaussianBlurAttack:'GBA',
foolbox.attacks.BoundaryAttack:'boundary',
foolbox.attacks.LocalSearchAttack:'LSA',
foolbox.attacks.PointwiseAttack:'PWA',
foolbox.attacks.ContrastReductionAttack:'CRA'

}
attack_list ={
foolbox.attacks.DeepFoolAttack,
foolbox.attacks.FGSM,
foolbox.attacks.LBFGSAttack,
foolbox.attacks.SaliencyMapAttack,
foolbox.attacks.IterativeGradientAttack,
foolbox.attacks.MomentumIterativeAttack,
foolbox.attacks.ProjectedGradientDescentAttack,
foolbox.attacks.CarliniWagnerL2Attack,
foolbox.attacks.GaussianBlurAttack,
foolbox.attacks.BoundaryAttack,
foolbox.attacks.LocalSearchAttack,
foolbox.attacks.PointwiseAttack,
foolbox.attacks.ContrastReductionAttack

}

# 打开ROOT_DIR的完整路径
ROOT_DIR = os.path.abspath('')
#打开绝对路径os.path.dirname(__file__)
def main(slef):
    batch_size = 128
    num_classes = 10
    epochs = 50

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    scoressss = []

    model.load_weights('/home/Bear/attack_detection/1_mnist_mlp_models/mnist_mlp_model.h5')  # 模型文件路径

    # for dududu in ['00_-50','00_-25','00_25','00_50','gv001m0','gv002m0','gv003m0','gv004m0','gv005m0']:
    # for dududu in [ 's1_-50', 's1_-25','s1', 's1_25', 's1_50','s2_-50', 's2_-25','s2', 's2_25', 's2_50']:
    # for dududu in [ 'x1_-50', 'x1_-25','x1', 'x1_25', 'x1_50','x2_-50', 'x2_-25','x2', 'x2_25', 'x2_50']:
    # for dududu in ['z1_-50', 'z1_-25', 'z1', 'z1_25', 'z1_50', 'z2_-50', 'z2_-25', 'z2', 'z2_25', 'z2_50']:
    # for dududu in [ 'y1_-50', 'y1_-25','y1', 'y1_25', 'y1_50','y2_-50', 'y2_-25','y2', 'y2_25', 'y2_50']:
    # for dududu in ['rs16','rs20','rs24','rs28','rs36','rs40','rs44']:
    for dududu in ['00_-50','00_-25','00_25','00_50',
                   's1_-50', 's1_-25','s1', 's1_25', 's1_50','s2_-50', 's2_-25','s2', 's2_25', 's2_50',
                   'x1_-50', 'x1_-25','x1', 'x1_25', 'x1_50','x2_-50', 'x2_-25', 'x2', 'x2_25', 'x2_50',
                   'y1_-50', 'y1_-25','y1', 'y1_25', 'y1_50','y2_-50', 'y2_-25', 'y2', 'y2_25', 'y2_50',
                   'z1_-50', 'z1_-25','z1', 'z1_25', 'z1_50','z2_-50', 'z2_-25', 'z2', 'z2_25', 'z2_50',
                   'rs16','rs20','rs24','rs32','rs36','rs40','rs44','rs48',
                   'gv001m0', 'gv002m0', 'gv003m0', 'gv004m0', 'gv005m0']:
    # for dududu in ['s1_9','s1_10',
    #                's1_15','s1_20','s1_25','s1_30','s1_40','s1_45','s1_50',
    #                's1_-50', 's1_-25', 's1_-10', 's1_-1', 's1', 's1_1', 's1_3', 's1_5', 's1_7',]:
    #     # Convert class vectors to binary class matrices.
        x_test0 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/x_test{}'.format(dududu) + '.npy')
        x_adtest0 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/x_adtest{}'.format(dududu) + '.npy')
        y_test0 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/orin/y_test00.npy')
        x_test0=np.reshape(x_test0,(len(x_test0),784))
        x_adtest0 = np.reshape(x_adtest0, (len(x_adtest0), 784))

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


        # np.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adtest{}'.format(dududu), y_adtest)
        # np.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_pred{}'.format(dududu), y_pred)
        np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidence{}'.format(dududu),y_confidence)
        np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_adconfidence{}'.format(dududu),y_adconfidence)

    print('scoressss', scoressss)



if __name__ == '__main__':
    # parser.add_argument('-mp', '--model_path', default='model/inception_v3.ckpt')  # 选择模型

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    # 不全部占满显存, 按需分配//
    sess = tf.Session(config=config)# 设置session
    KTF.set_session(sess)

    tf.app.run()