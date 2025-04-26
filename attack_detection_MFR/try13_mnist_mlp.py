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
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop



print('FGSM')
keras.backend.set_learning_phase(0)
parser=argparse.ArgumentParser()

parser.add_argument('-am', '--attack_method', default=foolbox.attacks.FGSM)  # 选择攻击方式，比如FGSM
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

    model.load_weights('/home/Bear/attack_detection/1_mnist_mlp_models/mnist_mlp_model.h5')  # 模型文件路径

    # Convert class vectors to binary class matrices.
    x_test0=np.load('/home/Bear/attack_detection/dataset_mnist/x_test.npy')
    x_test0 = x_test0.reshape(10000, 784)
    y_test0=np.load('/home/Bear/attack_detection/dataset_mnist/y_test.npy')
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


    np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/orin/x_test',x_testl)
    np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/orin/y_test', y_testl)
    np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/orin/x_adtest',x_adtest)
    # np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/orin/y_adtest', y_adtest)
    # np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/orin/y_pred', y_pred)
    np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/orin/y_confidence', y_confidence)
    np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/orin/y_adconfidence', y_adconfidence)
    np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/orin/cl',cl)


if __name__ == '__main__':
    # parser.add_argument('-mp', '--model_path', default='model/inception_v3.ckpt')  # 选择模型

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 不全部占满显存, 按需分配//
    sess = tf.Session(config=config)# 设置session
    KTF.set_session(sess)

    tf.app.run()