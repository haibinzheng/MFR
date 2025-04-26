from __future__ import print_function
import scipy.misc
import numpy as np
from PIL import Image

# The data, split between train and test sets:
x_test=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_all/x_test00.npy')
x_adtest=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_all/x_adtest00.npy')

x_test=x_test*255
x_adtest=x_adtest*255

print(x_test.shape)

print(np.max(x_test))
print(np.min(x_test))
print(np.max(x_adtest))
print(np.min(x_adtest))

print('x_adtest',x_adtest.shape)
print('x_test',x_test.shape)
# 50
x_rotest50=[]
for i in range(0,len(x_test)):
    x_rotest50.append(scipy.misc.imrotate(x_test[i],-25,interp='cubic'))
    # if i==550:
    #     img1=Image.fromarray(x_rotest50[i])
    #     img1.show()
    #     img1.save('img1.jpg')
x_rotest50=np.array(x_rotest50)
print(x_rotest50.shape)

x_roadtest50=[]
for i in range(0,len(x_adtest)):
    x_roadtest50.append(scipy.misc.imrotate(x_adtest[i],-25,interp='cubic'))
    # if i==550:
    #     img1=Image.fromarray(x_roadtest50[i])
    #     img1.show()
    #     img1.save('img2.jpg')
x_roadtest50=np.array(x_roadtest50)
print(x_roadtest50.shape)


x_rotest50=x_rotest50/255
x_roadtest50=x_roadtest50/255

print(np.max(x_rotest50))
print(np.min(x_rotest50))
print(np.max(x_roadtest50))
print(np.min(x_roadtest50))

np.save('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_all/x_test00_-25',x_rotest50)
np.save('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_all/x_adtest00_-25',x_roadtest50)




# if i==550:
#     img1=Image.fromarray(x_rotest[i])
#     img1.show()
#     img1.save('img888.jpg')