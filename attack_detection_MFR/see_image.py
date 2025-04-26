from __future__ import print_function
import scipy.misc
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import cv2
import os
label=np.load('/home/Bear/attack_detection/imag_val_lable.npy')
cls=[]
# for i in range(1,5001):
#     cls.append("%08d" % i)
# print(cls)
m=0
x_orin=[]
label_select=[]
for i in range(1,5001):
    cl="%08d" % i
    print(cl)
    image=plt.imread('/home/Bear/crx/ILSVRC2012_img_val/imagenet5000/ILSVRC2012_val_{}'.format(cl)+'.JPEG')
    print(image.shape)
    # plt.imsave('img0857.png', image)
    image=cv2.resize(image, (299,299), interpolation = cv2.INTER_CUBIC)
    # print(image)
    asize=image.shape
    print('asize',asize)
    print(image.shape)
    m=m+1
    if asize==(299,299,3):
        print(m)
        print(asize[2])
        x_orin.append(image)
        label_select.append(label[i-1])
        cls.append(i-1)
    if i==16:
        plt.imsave('img0016.png', image)

# print(x_orin.shape)
print(m)
x_orin=np.array(x_orin)
label_select=np.array(label_select)
cls=np.array(cls)
print(x_orin.shape)
print(label_select.shape)
print(cls.shape)
# np.save('/tmp/imagenet_orin/label_orin',label_select)
# np.save('/tmp/imagenet_orin/cl4901_orin',cls)
np.save('/tmp/orin_imagenet/orin_imagenet',x_orin)
# plt.imsave('img0856.png', image)



# for filename in os.listdir('/home/Bear/crx/ILSVRC2012_img_val/imagenet5000'):
#     src_image=load_image(os.path.join('/home/Bear/crx/ILSVRC2012_img_val/imagenet5000',filename))
#     # x_1=scipy.misc.imread(src_image)
#     x_test.append(src_image)
# x_test=np.array(x_test)
# print(x_test.shape)

# x=np.load('/tmp/imagenet_orin/imagenet_orin.npy')
# print(x.shape)
# # c=scipy.misc.imread(x[666])
# scipy.misc.imsave('0399.png',x[666])