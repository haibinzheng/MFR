'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

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

keras.backend.set_learning_phase(0)
parser=argparse.ArgumentParser()

parser.add_argument('-am', '--attack_method', default=foolbox.attacks.BoundaryAttack)  # 选择攻击方式，比如FGSM
parser.add_argument('-dp', '--data_path', default='imagenet')
parser.add_argument('-od', '--output_dir', default='samples')  # 选择攻击方式，比如FGSM
args = parser.parse_args()
print('0000000000000000000000000000000000000000000000000000000000')
print('foolbox.attacks.BoundaryAttack')

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
foolbox.attacks.LocalSearchAttack:'LSA',
foolbox.attacks.PointwiseAttack:'PWA',
foolbox.attacks.ContrastReductionAttack:'CRA'

}
attack_list =(
foolbox.attacks.DeepFoolAttack,
foolbox.attacks.FGSM,
foolbox.attacks.LBFGSAttack,
foolbox.attacks.SaliencyMapAttack,
foolbox.attacks.IterativeGradientAttack,
foolbox.attacks.MomentumIterativeAttack,
foolbox.attacks.ProjectedGradientDescentAttack,
foolbox.attacks.CarliniWagnerL2Attack,
foolbox.attacks.GaussianBlurAttack,
foolbox.attacks.LocalSearchAttack,
foolbox.attacks.PointwiseAttack,
foolbox.attacks.ContrastReductionAttack
)
# 打开ROOT_DIR的完整路径
ROOT_DIR = os.path.abspath('')
#打开绝对路径os.path.dirname(__file__)
def main(slef):
    batch_size = 128
    epoches = 250
    learning_rate = 0.1
    lr_decay = 1e-6
    lr_drop = 20
    num_classes = 10
    weight_decay = 0.0005
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

    # lr
    def lr_schedule(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

    lr_scheduler = LearningRateScheduler(lr_schedule)

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=input_shape, kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(512, kernel_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # optimization details
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    save_dir = os.path.join(os.getcwd(), '1_vgg16_saved_models')
    model_name = 'Vgg16_cifar10_trained_model.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)



    model.load_weights('/home/Bear/attack_detection/1_vgg16_saved_models/Vgg16_cifar10_trained_model_1216.h5')  # 模型文件路径

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


    # np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/x_test',x_testl)
    # np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/y_test', y_testl)
    # np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/x_adtest',x_adtest)
    # np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/y_adtest', y_adtest)
    # np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/y_pred', y_pred)
    # np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/y_confidence', y_confidence)
    # np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/y_adconfidence', y_adconfidence)
    # np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/cl', cl)


if __name__ == '__main__':
    # parser.add_argument('-mp', '--model_path', default='model/inception_v3.ckpt')  # 选择模型

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # 不全部占满显存, 按需分配
    config.gpu_options.per_process_gpu_memory_fraction = 0.15
    #指定分配30%空间
    sess = tf.Session(config=config)# 设置session
    KTF.set_session(sess)

    tf.app.run()