import numpy as np
import scipy.misc
x_test = np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_all/x_test00.npy')
x_adtest = np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_all/x_adtest00.npy')
y_test = np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_all/y_test00.npy')
x_adtest=x_adtest*255
x_test=x_test*255

print(np.max(x_test))
print(np.min(x_test))
print(np.max(x_adtest))
print(np.min(x_adtest))

print(x_test.shape)
print(x_adtest.shape)

x_testrs1=[]
x_adtestrs1=[]
for i in range(0,7390):
    x_testrs1.append(scipy.misc.imresize(x_test[i],(48,48), interp='bilinear'))
x_testreshape1 = np.array(x_testrs1)

for i in range(0,7390):
    x_adtestrs1.append(scipy.misc.imresize(x_adtest[i],(48,48), interp='bilinear'))
x_adtestreshape1 = np.array(x_adtestrs1)

print(x_testreshape1.shape)
print(x_adtestreshape1.shape)

x_testrs2=[]
x_adtestrs2=[]
for i in range(0,7390):
    x_testrs2.append(scipy.misc.imresize(x_testreshape1[i],(32,32), interp='bilinear'))
x_testreshape2 = np.array(x_testrs2)

for i in range(0,7390):
    x_adtestrs2.append(scipy.misc.imresize(x_adtestreshape1[i],(32,32), interp='bilinear'))
x_adtestreshape2 = np.array(x_adtestrs2)

print(x_testreshape2.shape)
print(x_adtestreshape2.shape)

x_adtestreshape2=x_adtestreshape2/255
x_testreshape2=x_testreshape2/255

print(np.max(x_testreshape2))
print(np.min(x_testreshape2))
print(np.max(x_adtestreshape2))
print(np.min(x_adtestreshape2))
np.save('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_all/x_testrs48',x_testreshape2)
np.save('/home/Bear/attack_detection/fgsm_cnn_cifar10/cifar10_all/x_adtestrs48',x_adtestreshape2)