import numpy as np
import skimage

x_test0 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_test00.npy')
x_adtest0 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_adtest00.npy')
y_test0 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_test00.npy')
noise=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/noise/noise_gv001m0.npy')
print(noise.shape)

x_testnoise=x_test0+noise
x_adtestnoise=x_adtest0+noise

print(np.max(x_testnoise))
print(np.min(x_testnoise))
print(np.max(x_adtestnoise))
print(np.min(x_adtestnoise))
x_testnoise=np.clip(x_testnoise,0,1)
x_adtestnoise=np.clip(x_adtestnoise,0,1)
print(np.max(x_testnoise))
print(np.min(x_testnoise))
print(np.max(x_adtestnoise))
print(np.min(x_adtestnoise))

print(x_testnoise.shape)
print(x_adtestnoise.shape)
np.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_testgv001m0',x_testnoise)
np.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_adtestgv001m0',x_adtestnoise)

# x_adtestnoise002=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_gv003m0/x_adtestn003m0.npy')
# print(x_adtestnoise002.shape)
