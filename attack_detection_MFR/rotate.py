from __future__ import print_function
import scipy.misc
import numpy as np
from PIL import Image


# The data, split between train and test sets:
x_test=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_00/x_test.npy')
x_adtest=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_00/x_adtest.npy')

x_test=x_test*255
x_adtest=x_adtest*255

print(np.max(x_test))
print(np.min(x_test))
print(np.max(x_adtest))
print(np.min(x_adtest))

print('x_adtest',x_adtest.shape)
print('x_test',x_test.shape)
x_rotest=[]
for i in range(0,7390):
    x_rotest.append(scipy.misc.imrotate(x_test[i],1,interp='cubic'))
x_rotest=np.array(x_rotest)
print(x_rotest.shape)
print(x_rotest[0])
# np.save('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_25/x_test25',x_rotest)

x_roadtest=[]
for i in range(0,7390):
    x_roadtest.append(scipy.misc.imrotate(x_adtest[i],1,interp='cubic'))
x_roadtest=np.array(x_roadtest)
print(x_roadtest.shape)
print(x_roadtest[0])
# np.save('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_25/x_adtest25',x_roadtest)

print(np.max(x_rotest-x_test))
print(np.min(x_rotest-x_test))
print(np.max(x_roadtest-x_adtest))
print(np.min(x_roadtest-x_adtest))

x_rotest=x_rotest/255
x_roadtest=x_roadtest/255
print(np.max(x_rotest))
print(np.min(x_rotest))
print(np.max(x_roadtest))
print(np.min(x_roadtest))

np.save('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_01/x_test01',x_rotest)
np.save('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_01/x_adtest01',x_roadtest)