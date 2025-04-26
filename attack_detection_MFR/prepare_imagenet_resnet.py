import tensorflow as tf
import foolbox
import argparse
import numpy as np
from scipy.misc import imsave,imresize,imshow
import idx2numpy
import os
from tensorflow.contrib.slim.nets import inception,resnet_v2
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
import matplotlib.pylab as plt
from PIL import Image
import  keras
from sklearn.metrics import accuracy_score
import keras.backend.tensorflow_backend as KTF
parser=argparse.ArgumentParser()
parser.add_argument('-am','--attack_method',default=foolbox.attacks.FGSM)  #选择攻击方式，比如FGSM
parser.add_argument('-dp','--data_path',default='example_299_299.png')
args=parser.parse_args()

print('FGSM' )


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


kt=1


slim=tf.contrib.slim


def main(self):
    num_classes = 1001

    x = tf.placeholder(dtype=tf.float32, shape=[1, 299, 299, 3])
    x_ = tf.image.resize_images(x, size=(299, 299))

    # with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    #     logits, end_points_v3 = inception_v3.inception_v3(
    #         x, num_classes=num_classes, is_training=False)
    # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    #     logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
    #         x, num_classes=num_classes, is_training=False)
    # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    #   logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
    #       x, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')
    # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    #     logits, end_points_resnet = resnet_v2.resnet_v2_152(x, num_classes=1001, is_training=False)
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points_resnet = resnet_v2.resnet_v2_152(x, num_classes=1001, is_training=False)

    saver = tf.train.Saver()
    i=0
    j=0

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    attack_name = attack_dir[args.attack_method]

    with tf.Session(config=config) as sess:
        # saver.restore(sess,r'/home/Bear/attack_detection/models/resnet_v2_101.ckpt')
        # saver.restore(sess, r'/home/Bear/attack_detection/models/inception_resnet_v2_2016_08_30.ckpt')
        saver.restore(sess, r'/home/Bear/attack_detection/models/resnet_v2_152.ckpt')

        '''['00_-50', '00_-25', '00_25', '00_50',
                       's1_-50', 's1_-25', 's1', 's1_25', 's1_50', 's2_-50', 's2_-25', 's2', 's2_25', 's2_50',
                       'x1_-50', 'x1_-25', 'x1', 'x1_25', 'x1_50', 'x2_-50', 'x2_-25', 'x2', 'x2_25', 'x2_50',
                       'y1_-50', 'y1_-25', 'y1', 'y1_25', 'y1_50', 'y2_-50', 'y2_-25', 'y2', 'y2_25', 'y2_50',
                       'z1_-50', 'z1_-25', 'z1', 'z1_25', 'z1_50', 'z2_-50', 'z2_-25', 'z2', 'z2_25', 'z2_50',
                       'rs100', 'rs150', 'rs200', 'rs250', 'rs280', 'rs295','rs303','rs350', 'rs400', 'rs450',
                       'gv001m0', 'gv002m0', 'gv003m0', 'gv004m0', 'gv005m0']:,'rs303','re310','rs350', 'rs400', 'rs450','rs250','rs280','rs290',
                       'gv001m0', 'gv002m0', 'gv003m0', 'gv004m0', 'gv005m0'''
        for dududu in [ 's1']:
            print(dududu)
            x_test0 = np.load('/tmp/fgsm_resnet152_imagenet/image_all/all_2000/x_test{}'.format(dududu) + '.npy')
            x_adtest0 = np.load('/tmp/fgsm_resnet152_imagenet/image_all/all_2000//x_adtest{}'.format(dududu) + '.npy')
            y_test0 = np.load('/tmp/fgsm_resnet152_imagenet/orin1/y_test2000.npy')
            # x_test0 = np.reshape(x_test0, (299, 299, 3))
            # x_adtest0 = np.reshape(x_adtest0, (299, 299, 3))
            # y_test0=keras.utils.to_categorical(y_test0,1001)
            y_confidence = []
            y_adconfidence = []

            for i in range(0,500):
                with foolbox.models.TensorFlowModel(x, tf.reshape(logits, shape=(-1, 1001)), (0, 1)) as model1:
                    confidence = model1.predictions(x_test0[i])
                    image_name = str(i)
                    adconfidence = model1.predictions(x_adtest0[i])
                    y_confidence.append(confidence)
                    y_adconfidence.append(adconfidence)

                    print("name:{},reallabel:{},source label:{},adv label:{} ".format(
                        image_name, y_test0[i],np.argmax(confidence), np.argmax(adconfidence)))

            y_confidence = np.array(y_confidence)
            y_adconfidence = np.array(y_adconfidence)

            print('y_confidence', y_confidence.shape)
            print('y_adconfidence', y_adconfidence.shape)

            np.save('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidence{}'.format(dududu),y_confidence)
            np.save('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_adconfidence{}'.format(dududu),y_adconfidence)

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 不全部占满显存, 按需分配//
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    #指定分配30%空间

    sess = tf.Session(config=config)# 设置session
    KTF.set_session(sess)

    tf.app.run()

