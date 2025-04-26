import numpy as np
from sklearn.model_selection import train_test_split

ax_black1=np.load('/home/Bear/attack_detection/GBA_vgg16_cifar10/cifar10_all/18/x_adconfidence18.npy')
ax_black2=np.load('/home/Bear/attack_detection/LSA_vgg16_cifar10/cifar10_all/18/x_adconfidence18.npy')
ax_black3=np.load('/home/Bear/attack_detection/PWA_vgg16_cifar10/cifar10_all/18/x_adconfidence18.npy')
ax_black4=np.load('/home/Bear/attack_detection/CRA_vgg16_cifar10/cifar10_all/18/x_adconfidence18.npy')
ax_white1=np.load('/home/Bear/attack_detection/cw_vgg16_cifar10/cifar10_all/18/x_adconfidence18.npy')
ax_white2=np.load('/home/Bear/attack_detection/deepfool_vgg16_cifar10/cifar10_all/18/x_adconfidence18.npy')
ax_white3=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/18/x_adconfidence18.npy')

x_black1=np.load('/home/Bear/attack_detection/GBA_vgg16_cifar10/cifar10_all/18/x_confidence18.npy')
x_black2=np.load('/home/Bear/attack_detection/LSA_vgg16_cifar10/cifar10_all/18/x_confidence18.npy')
x_black3=np.load('/home/Bear/attack_detection/PWA_vgg16_cifar10/cifar10_all/18/x_confidence18.npy')
x_black4=np.load('/home/Bear/attack_detection/CRA_vgg16_cifar10/cifar10_all/18/x_confidence18.npy')
x_white1=np.load('/home/Bear/attack_detection/cw_vgg16_cifar10/cifar10_all/18/x_confidence18.npy')
x_white2=np.load('/home/Bear/attack_detection/deepfool_vgg16_cifar10/cifar10_all/18/x_confidence18.npy')
x_white3=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/18/x_confidence18.npy')

x_all=np.concatenate((x_black1,x_black2,x_black3,x_black4,x_white1,x_white2,x_white3),axis=0)
ax_all=np.concatenate((ax_black1,ax_black2,ax_black3,ax_black4,ax_white1,ax_white2,ax_white3),axis=0)
print(x_all.shape)
print(ax_all.shape)
np.save('/tmp/data_wb/x',x_all)
np.save('/tmp/data_wb/ax',ax_all)
y_x=np.zeros((len(x_all)))
y_ax=np.zeros((len(ax_all)))+1
print(y_ax.shape)
print(y_x.shape)

print(y_x)
print(np.max(y_x))
print(np.min(y_x))

print(y_ax)
print(np.max(y_ax))
print(np.min(y_ax))

np.save('/tmp/data_wb/y',y_x)
np.save('/tmp/data_wb/ay',y_ax)

X = np.concatenate((x_all,ax_all), axis=0)
Y=  np.concatenate((y_x,y_ax), axis=0)
print(X.shape)
print(np.max(X))
print(np.min(X))
print(Y.shape)
np.save('/tmp/data_wb/Y',Y)
np.save('/tmp/data_wb/X',X)