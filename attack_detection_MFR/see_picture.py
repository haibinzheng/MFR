from __future__ import print_function
import scipy.misc
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import os

def load_image(path):
    # print('load image from {}'.format(path))
    image=plt.imread(path)
    # image=image[:,:,:3]
    image=plt.resize(image,new_shape=(299,299,3))
    # image=np.expand_dims(image,axis=0)
    return image/255.

n_0101=np.load('/tmp/imagenet_orin/imagenet_orin.npy')
# n_0102=np.load('/home/Bear/attack_detection/2108.JPEG.npy')
# print(n_0101.shape)
# print(n_0102.shape)
# # n_0101=np.reshape(n_0101,(1,299,299,3))
# # n_0102=np.reshape(n_0102,(1,299,299,3))
x1=n_0101[111]
# x2=n_0102
# x1=np.reshape(x1[11],(299,299,3))
# x2=np.reshape(x2,(299,299,3))
# np.save('img1845.jpg',x1)
# np.save('img1846.jpg',x2)

# scipy.misc.imsave('img2012.png', x2)
# # img1=Image.fromarray(x1)
# # img1.show()
# # img1.save('img1848.jpg')
# #
# # img2=Image.fromarray(x2)
# # img2.show()
# # img2.save('img1849.jpg')




x_test=[]
for filename in os.listdir('/home/Bear/crx/ILSVRC2012_img_val/imagenet5000'):
    src_image=load_image(os.path.join('/home/Bear/crx/ILSVRC2012_img_val/imagenet5000',filename))
    # x_1=scipy.misc.imread(src_image)
    x_test.append(src_image)
x_test=np.array(x_test)
print(x_test.shape)
#
# np.save('/tmp/imagenet_orin/imagenet_orin',x_test)