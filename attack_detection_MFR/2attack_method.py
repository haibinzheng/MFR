import tensorflow as tf
import foolbox
import argparse
import numpy as np
from matplotlib.pyplot import imread,imshow
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

import glob,scipy,os,sys
from scipy.misc import imread
# 打开ROOT_DIR的完整路径
ROOT_DIR = os.path.abspath('')
#打开绝对路径os.path.dirname(__file__)
def load_image(dataset_name, size=(224,224)) :
    if os.path.isfile(dataset_name):
        x=[imread(dataset_name,mode='RGB')[:,:,:3]]
    else:
        x = glob.glob(os.path.join("./dataset", dataset_name, '*.*'))
        print(x)
        x = [scipy.misc.imread(i,mode='RGB')[:,:,:3] for i in x]
        x = [scipy.misc.imresize(i,size=size)   for i in x]
    return np.array(x)/255.0

def inference(args,num_classes=1001):
    from scipy.misc import imsave
    path=args.data_path
    src_image=load_image(path,size=(299,299))
    from tensorflow.contrib.slim.nets import inception,resnet_v2

    x_=tf.placeholder(dtype=tf.float32,shape=[None,299,299,3])
    # x=tf.image.resize_images(x_,size=(224,224))


    # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    #     logits, end_points_resnet = resnet_v2.resnet_v2_101(
    #         x_, num_classes=num_classes, is_training=False)
    #
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(
            x_, num_classes=1001, is_training=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess,args.model_path)  #模型文件路径
        attack_name = attack_dir[args.attack_method]
        # saver.restore(sess,os.path.join(ROOT_DIR,args.model_path))  #模型文件路径
        with foolbox.models.TensorFlowModel(x_,tf.reshape(logits,shape=(-1,num_classes)),(0,1)) as model:
            attack = args.attack_method(model)
            # print(model.predictions(src_image))
            for idx,image in enumerate(src_image):
                image_name = str(idx)
                label=np.argmax(model.predictions(image))
                ##test
                adversarial=attack(image,label)
                print("name:{}, source label:{},  adv label:{} ".format(image_name,label,
                                                                        np.argmax(model.predictions(adversarial))))
                # if
                # print('')

                haha=np.abs(adversarial-image)
                imsave(os.path.join(args.output_dir,attack_name,image_name)+'.npy',adversarial)
                imsave(os.path.join(args.output_dir,attack_name,'perturbation',image_name)+'.png',haha)
                if idx > 200:
                    break
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-am','--attack_method',default= foolbox.attacks.FGSM)  #选择攻击方式，比如FGSM
    parser.add_argument('-dp','--data_path',default='imagenet')
    parser.add_argument('-od', '--output_dir', default='baseline')  # 选择攻击方式，比如FGSM
    parser.add_argument('-mp', '--model_path', default='model/inception_v3.ckpt')  # 选择攻击方式，比如FGSM
    args=parser.parse_args()

    # with tf.Graph().as_default():
    inference(args)