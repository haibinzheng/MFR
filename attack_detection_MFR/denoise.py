import numpy as np
import skimage

x_test0 = np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_00/x_test.npy')
x_adtest0 = np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_00/x_adtest.npy')
x_test0=x_test0*255
x_adtest0=x_adtest0*255
y_test0 = np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_00/y_test.npy')


x_noise=skimage.util.random_noise(X,mode='gaussian',var=0.01,mean=0, seed=None, clip=False)
noise=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/noise_gv001m0.npy')
print(noise[0])
x_testn001m0=x_test0+noise
x_adtestn001m0=x_adtest0+noise
print(x_adtestn001m0[500])
np.save('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_gv001m0/x_testn001m0',x_testn001m0)
np.save('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_gv001m0/x_adtestn001m0',x_adtestn001m0)


