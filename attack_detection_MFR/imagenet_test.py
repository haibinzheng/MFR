import tensorflow as tf
import foolbox
import argparse
import numpy as np
from scipy.misc import imsave,imresize,imshow
import idx2numpy
import os
import cv2
from tensorflow.contrib.slim.nets import inception,resnet_v2
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
import matplotlib.pylab as plt
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('-am','--attack_method',default=foolbox.attacks.FGSM)  #选择攻击方式，比如FGSM
parser.add_argument('-dp','--data_path',default='example_299_299.png')
args=parser.parse_args()

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

def load_image(path):
    # print('load image from {}'.format(path))
    image=plt.imread(path)
    # image=image[:,:,:3]
    image=plt.resize(image,new_shape=(299,299,3))
    # image=np.expand_dims(image,axis=0)
    return image/255.

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5 * 255, format='png')


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
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points_resnet = resnet_v2.resnet_v2_152(x, num_classes=1001, is_training=False)

    saver = tf.train.Saver()
    i=0
    j=0
    y_test0 = np.load('/tmp/imagenet_orin/label_orin.npy')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    x_adtest=[]
    x_test=[]
    name_cw=[]
    ii=0
    with tf.Session(config=config) as sess:
        # saver.restore(sess,r'/home/zhb/Documents/crx/models/resnet_v2_101.ckpt')
        # saver.restore(sess, r'/home/zhb/Documents/crx/models/inception_resnet_v2_2016_08_30.ckpt')
        saver.restore(sess, r'/home/Bear/attack_detection/models/resnet_v2_152.ckpt')

        for i in range(0, 5000):
            try:
                cl = "%08d" % (i+1)
                src_image = plt.imread('/home/Bear/crx/ILSVRC2012_img_val/imagenet5000/ILSVRC2012_val_{}'.format(cl) + '.JPEG')
                src_image = cv2.resize(src_image, (299, 299), interpolation=cv2.INTER_CUBIC)
                with foolbox.models.TensorFlowModel(x,tf.reshape(logits,shape=(-1,1001)),(0,255)) as model:
                    label = np.argmax(model.predictions(src_image))
                    attack = args.attack_method(model)
                    perturbed_images = attack(src_image,label)
                    pred_label_perturbed=np.argmax(model.predictions(perturbed_images))
                    print(
                        "name:{},reallabel:{},pred_label:{},adv label:{} ".format(i, y_test0[i],label,
                                                                                    pred_label_perturbed))
                if label!=pred_label_perturbed:
                    i=i+1
                # imsave('/home/zhb/Documents/crx/AttentionDefense/adv/inceptionv3/momentum/{}.jpg'.format(filename), perturbed_images)
                #     np.save('/home/Bear/attack_detection/imagenet_cw/{}'.format(filename),perturbed_images)
                    x_adtest.append(perturbed_images)
                    # name_cw.append(filename)
                    # print('save:',filename)
                #     if pred_label_perturbed != label:  #攻击成功率
                #         i = i + 1
                # print(i)
                # print(i/500)
                    # imsave('giraffe/adv_example_{0}.png'.format(attack_dir[args.attack_method]),adversarial)
            except:
                # print('src image error',filename)
                j=j+1
        print('err num:',j)
        print('success num',i)
        x_adtest=np.array(x_adtest)
        print(x_adtest.shape)
        # np.save('/home/Bear/attack_detection/imagenet_fgsm/x_fgsm',x_adtest)
        # np.save('/home/Bear/attack_detection/imagenet_fgsm/cl_fgsm', name_cw)

if __name__ == '__main__':
    tf.app.run()




    # with tf.Graph().as_default():
    #     inference(args)