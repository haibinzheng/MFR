from __future__ import print_function
import scipy.misc
import numpy as np
from PIL import Image

# The data, split between train and test sets:
for dududu in ['s1']:
    print(dududu)
    x_test=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_test{}'.format(dududu)+'.npy')
    x_adtest=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_adtest{}'.format(dududu)+'.npy')

    length=len(x_test)
    x_test=x_test*255
    x_adtest=x_adtest*255

    print(x_test.shape)

    print(np.max(x_test))
    print(np.min(x_test))
    print(np.max(x_adtest))
    print(np.min(x_adtest))

    print('x_adtest',x_adtest.shape)
    print('x_test',x_test.shape)

    for j in [-50,-25,-10,-1,1,3,5,7,9,10,15,20,25,30,40,45,50]:
        x_rotest = []
        print('angle=',j)
        for i in range(0,length):
            x_rotest.append(scipy.misc.imrotate(x_test[i],j,interp='cubic'))
        x_rotest=np.array(x_rotest)
        print(x_rotest.shape)
        x_rotest = x_rotest / 255
        print(np.max(x_rotest))
        print(np.min(x_rotest))
        print(np.max(x_rotest))
        print(np.min(x_rotest))
        name=str(j)
        np.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_test{}'.format(dududu)+'_{}'.format(name), x_rotest)


        x_adrotest = []
        print('angle=',j)
        for i in range(0,length):
            x_adrotest.append(scipy.misc.imrotate(x_adtest[i],j,interp='cubic'))
        x_adrotest=np.array(x_adrotest)
        print(x_adrotest.shape)
        x_adrotest = x_adrotest / 255
        print(np.max(x_adrotest))
        print(np.min(x_adrotest))
        print(np.max(x_adrotest))
        print(np.min(x_adrotest))
        name=str(j)
        np.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_adtest{}'.format(dududu)+'_{}'.format(name), x_adrotest)
