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

parser=argparse.ArgumentParser()
parser.add_argument('-am','--attack_method',default=foolbox.attacks.DeepFoolAttack)  #选择攻击方式，比如FGSM
parser.add_argument('-dp','--data_path',default='example_299_299.png')
args=parser.parse_args()

print('resnet_DF')


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

x_test0=np.load('/tmp/orin_imagenet/orin_imagenet.npy')
# y_test0=np.load('/tmp/imagenet_orin/label_orin.npy')
# y_test0 = keras.utils.to_categorical(y_test0, 1001)
x_test0=x_test0/255
slim=tf.contrib.slim
print(np.min(x_test0))
print(np.max(x_test0))

def main(self):
    num_classes = 1001

    x = tf.placeholder(dtype=tf.float32, shape=[1, 299, 299, 3])
    x_ = tf.image.resize_images(x, size=(299, 299))

    # with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    #     logits, end_points_v3 = inception_v3.inception_v3(x, num_classes=num_classes, is_training=False)
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

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    print(x_test0.shape)
    # print(y_test0.shape)

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

    x_adtest=[]
    x_test=[]
    name_cw=[]
    with tf.Session(config=config) as sess:
        # saver.restore(sess,r'/home/Bear/attack_detection/models/resnet_v2_101.ckpt')
        # saver.restore(sess, r'/home/Bear/attack_detection/models/inception_resnet_v2_2016_08_30.ckpt')
        saver.restore(sess, r'/home/Bear/attack_detection/models/resnet_v2_152.ckpt')
        # saver.restore(sess,r'/home/Bear/attack_detection/models/inception_v3.ckpt')
        with foolbox.models.TensorFlowModel(x, tf.reshape(logits, shape=(-1, 1001)), (0, 1)) as model1:
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
                    # y_testl.append(np.argmax(y_test0[idx]))
                    y_pred.append(pred_label)
                    y_testl.append(pred_label)
                    y_confidence.append(confidence)

                    x_adtest.append(adversarial)
                    y_adtest.append(adv_label)
                    y_adconfidence.append(adconfidence)
                    cl.append(idx)
                    print(
                        "name:{},reallabel:{},source label:{},adv label:{} ".format(image_name, pred_label,
                                                                                    pred_label, adv_label))

        x_testl = np.array(x_testl)
        y_testl = np.array(y_testl)
        x_adtest = np.array(x_adtest)
        y_adtest = np.array(y_adtest)
        y_pred = np.array(y_pred)
        y_confidence = np.array(y_confidence)
        y_adconfidence = np.array(y_adconfidence)
        cl = np.array(cl)

        print('x_test', x_testl.shape)
        print('y_test', y_testl.shape)
        print('x_adtest', x_adtest.shape)
        print('y_adtest', y_adtest.shape)
        print('y_pred', y_pred.shape)
        print('y_confidence', y_confidence.shape)
        print('y_adconfidence', y_adconfidence.shape)
        print('cl', cl.shape)

        np.save('/tmp/deepfool_resnet152_imagenet/orin/x_test',x_testl)
        np.save('/tmp/deepfool_resnet152_imagenet/orin/y_test', y_testl)
        np.save('/tmp/deepfool_resnet152_imagenet/orin/x_adtest',x_adtest)
        np.save('/tmp/deepfool_resnet152_imagenet/orin/y_adtest', y_adtest)
        # np.save('/tmp/imagenet_cw/orin/y_pred', y_pred)
        np.save('/tmp/deepfool_resnet152_imagenet/orin/y_confidence', y_confidence)
        np.save('/tmp/deepfool_resnet152_imagenet/orin/y_adconfidence', y_adconfidence)
        np.save('/tmp/deepfool_resnet152_imagenet/orin/cl', cl)
        # np.save('/home/Bear/attack_detection/imagenet_fgsm/x_fgsm',x_adtest)
        # np.save('/home/Bear/attack_detection/imagenet_fgsm/cl_fgsm', name_cw)

if __name__ == '__main__':
    tf.app.run()




    # with tf.Graph().as_default():
    #     inference(args)