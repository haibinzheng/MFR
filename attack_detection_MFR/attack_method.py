import tensorflow as tf
import foolbox
import argparse
import numpy as np
from scipy.misc import imsave,imread,imresize,imshow
import os
attack_list =(
foolbox.attacks.DeepFoolAttack,
foolbox.attacks.FGSM,
foolbox.attacks.LBFGSAttack,
foolbox.attacks.SaliencyMapAttack,
foolbox.attacks.IterativeGradientAttack,
foolbox.attacks.MomentumIterativeAttack,
foolbox.attacks.CarliniWagnerL2Attack
)
parser=argparse.ArgumentParser()
parser.add_argument('-am','--attack_method',default=attack_list[1])  #选择攻击方式，比如FGSM
parser.add_argument('-dp','--data_path',default='example_299_299.png')
args=parser.parse_args()
attack_dir = {
foolbox.attacks.DeepFoolAttack:'deepfool',
foolbox.attacks.FGSM:'fgsm',
foolbox.attacks.LBFGSAttack:'lbfgsa',
foolbox.attacks.SaliencyMapAttack:'saliencymap',
foolbox.attacks.IterativeGradientAttack:'iterativegrad',
foolbox.attacks.MomentumIterativeAttack:'momentumiter',
foolbox.attacks.CarliniWagnerL2Attack:'cw2'
}

kt=1


slim=tf.contrib.slim

def haha(inputs):
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    logits = tf.layers.dense(inputs=pool2_flat, units=10)
    return logits

def main(self):

    mnist_data_dir = '.'
    mnist_images = idx2numpy.convert_from_file(os.path.join(mnist_data_dir, 't10k-images.idx3-ubyte'))
    mnist_labels = idx2numpy.convert_from_file(os.path.join(mnist_data_dir, 't10k-labels.idx1-ubyte'))
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='images')
    logits = haha(x)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    i=0
    with tf.Session() as sess:
        saver.restore(sess, os.path.join('saved_models', 'keras_cifar10_trained_model.h5t'))
        # saver.restore(sess,'/home/Bear/models/inception_v3.ckpt')  #模型文件路径
        # saver.restore(sess,'/home/Bear/models/resnet_v2_101.ckpt')  #模型文件路径

        for j in range (0,499):
            i_random_image = j
            src_image = mnist_images[i_random_image]
            src_image = np.reshape(src_image, newshape=[28, 28, 1])
            test_label = mnist_labels[i_random_image]
            # test_label = np.expand_dims(test_label, 0)
            # test_label=np.reshape(test_label,[-1])



            with foolbox.models.TensorFlowModel(x,tf.reshape(logits,shape=(-1,10)),(0,255)) as model:
                attack=args.attack_method(model)
                perturbed_images = attack(src_image, test_label)

            pred_label_perturbed=np.argmax(model.predictions(perturbed_images))
            print(j)
            print("Ground truth label:", test_label)
            print("Predicted label after perturbation:", pred_label_perturbed)
            attack=args.attack_method(model)
            perturbed_images=np.tile(perturbed_images,(1,1,3))
            imsave('/home/Bear/stAdv-master/demo/fgsm_output/{}.png'.format(j), perturbed_images)
            if pred_label_perturbed != test_label:
                i = i + 1
        print(i)
        print(i/500)
if __name__ == '__main__':
    tf.app.run()




    # with tf.Graph().as_default():
    #     inference(args)