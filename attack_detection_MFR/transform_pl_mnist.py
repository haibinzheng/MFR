from __future__ import print_function
import scipy.misc
import numpy as np
from PIL import Image

# The data, split between train and test sets:
for dududu in ['00','s1','s2','x1','x2','z1','z2','y1','y2']:
    print(dududu)
    x_test=np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/x_test{}'.format(dududu)+'.npy')
    x_adtest=np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/x_adtest{}'.format(dududu)+'.npy')
    print(x_test.shape)
    print(x_adtest.shape)

    x_test=np.reshape(x_test,(len(x_test),28,28))
    x_adtest=np.reshape(x_adtest,(len(x_adtest),28,28))
    print(x_test.shape)
    print(x_adtest.shape)


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

    for j in [-2,-1,1,2]:
        x_rotest = []
        print('angle=',j*25)
        for i in range(0,length):
            x_rotest.append(scipy.misc.imrotate(x_test[i],25*j,interp='cubic'))
        x_rotest=np.array(x_rotest)
        print(x_rotest.shape)
        x_rotest = x_rotest / 255
        print(np.max(x_rotest))
        print(np.min(x_rotest))
        print(np.max(x_rotest))
        print(np.min(x_rotest))
        name=str(25*j)
        x_rotest=np.reshape(x_rotest,(len(x_rotest),28,28,1))
        print(x_rotest.shape)
        np.save('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/x_test{}'.format(dududu)+'_{}'.format(name), x_rotest)


        x_adrotest = []
        print('angle=',j*25)
        for i in range(0,length):
            x_adrotest.append(scipy.misc.imrotate(x_adtest[i],25*j,interp='cubic'))
        x_adrotest=np.array(x_adrotest)
        print(x_adrotest.shape)
        x_adrotest = x_adrotest / 255
        print(np.max(x_adrotest))
        print(np.min(x_adrotest))
        print(np.max(x_adrotest))
        print(np.min(x_adrotest))
        name=str(25*j)
        x_adrotest=np.reshape(x_adrotest,(len(x_adrotest),28,28,1))
        print(x_adrotest.shape)
        np.save('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/x_adtest{}'.format(dududu)+'_{}'.format(name), x_adrotest)


    x_test = np.reshape(x_test, (len(x_test), 28, 28,1))
    x_adtest = np.reshape(x_adtest, (len(x_adtest), 28, 28,1))
    print(x_test.shape)
    print(x_adtest.shape)
  # if i==550:
        #     img1=Image.fromarray(x_rotest[i])
        #     img1.show()
        #     img1.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_adtest00_{}'.format(str(j))+'.jpg')
            # 'img1.jpg'

# if i==550:
#     img1=Image.fromarray(x_rotest[i])
#     img1.show()
#     img1.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_adtest00_{}'.format(str(j))+'.jpg')
# 'img1.jpg'