import numpy as np
import os

cls=np.load('/home/Bear/attack_detection/cw_resnet_cifar10/orin/cl.npy')

x_adtest= []
x_test= []
y_adconfidence =[]
y_adtest=[]
y_confidence=[]
y_pred=[]
y_test=[]
for cl in cls:     #namelist
    print(cl)
    x_adtest0=np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/x_adtest/{}'.format(str(cl))+'.npy')
    x_test0 = np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/x_test/{}'.format(str(cl)) + '.npy')
    y_adconfidence0 = np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_adconfidence/{}'.format(str(cl)) + '.npy')
    y_adtest0 = np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_adtest/{}'.format(str(cl)) + '.npy')
    y_confidence0 = np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_confidence/{}'.format(str(cl)) + '.npy')
    y_pred0 = np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_pred/{}'.format(str(cl)) + '.npy')
    y_test0 = np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_test/{}'.format(str(cl)) + '.npy')

    x_test.append(x_test0)
    y_test.append(y_test0)
    x_adtest.append(x_adtest0)
    y_adtest.append(y_adtest0)
    y_pred.append(y_pred0)
    y_confidence.append(y_confidence0)
    y_adconfidence.append(y_adconfidence0)

x_test = np.array(x_test)
y_test = np.array(y_test)
x_adtest = np.array(x_adtest)
y_adtest = np.array(y_adtest)
y_pred = np.array(y_pred)
y_confidence = np.array(y_confidence)
y_adconfidence = np.array(y_adconfidence)

print('x_test', x_test.shape)
print('y_test', y_test.shape)
print('x_adtest', x_adtest.shape)
print('y_adtest', y_adtest.shape)
print('y_pred', y_pred.shape)
print('y_confidence', y_confidence.shape)
print('y_adconfidence', y_adconfidence.shape)

np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/x_test', x_test)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_test10000', y_test)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/x_adtest', x_adtest)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_adtest', y_adtest)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_pred', y_pred)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_confidence', y_confidence)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_adconfidence', y_adconfidence)

