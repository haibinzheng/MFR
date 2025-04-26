import numpy as np
from sklearn.model_selection import train_test_split

x_black1=np.load('/home/Bear/attack_detection/GBA_vgg16_cifar10/cifar10_all/56/x_adconfidence56.npy')
x_black2=np.load('/home/Bear/attack_detection/LSA_vgg16_cifar10/cifar10_all/56/x_adconfidence56.npy')
x_black3=np.load('/home/Bear/attack_detection/PWA_vgg16_cifar10/cifar10_all/56/x_adconfidence56.npy')
x_black4=np.load('/home/Bear/attack_detection/CRA_vgg16_cifar10/cifar10_all/56/x_adconfidence56.npy')
# x_black5=np.load('/home/Bear/attack_detection/LSA_vgg16_cifar10/cifar10_all/5/x_adconfidence5.npy')
# x_black6=np.load('/home/Bear/attack_detection/LSA_vgg16_cifar10/cifar10_all/5/x_adconfidence5.npy')
# x_black7=np.load('/home/Bear/attack_detection/LSA_vgg16_cifar10/cifar10_all/5/x_adconfidence5.npy')
x_white1=np.load('/home/Bear/attack_detection/cw_vgg16_cifar10/cifar10_all/56/x_adconfidence56.npy')
x_white2=np.load('/home/Bear/attack_detection/deepfool_vgg16_cifar10/cifar10_all/56/x_adconfidence56.npy')
x_white3=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/56/x_adconfidence56.npy')
# x_white4=np.load('/home/Bear/attack_detection/fgsm_resnet_cifar10/cifar10_all/05/x_adconfidence05.npy')
# x_white5=np.load('/home/Bear/attack_detection/fgsm_resnet_cifar10/cifar10_all/05/x_adconfidence05.npy')
# x_white6=np.load('/home/Bear/attack_detection/fgsm_resnet_cifar10/cifar10_all/05/x_adconfidence05.npy')
# x_white7=np.load('/home/Bear/attack_detection/fgsm_resnet_cifar10/cifar10_all/05/x_adconfidence05.npy')

x_black=np.concatenate((x_black1,x_black2,x_black3,x_black4),axis=0)
x_white=np.concatenate((x_white1,x_white2,x_white3),axis=0)


print(x_black.shape)
print(x_white.shape)
y_white=np.zeros((len(x_white)))
y_black=np.zeros((len(x_black)))+1

print(y_white)
print(np.max(y_white))
print(np.min(y_white))

print(y_black)
print(np.max(y_black))
print(np.min(y_black))
# np.save('/home/Bear/attack_detection/w_b_fgsm_GBA_resnet/x_white',x_white)
# np.save('/home/Bear/attack_detection/w_b_fgsm_GBA_resnet/y_white',y_white)
# np.save('/home/Bear/attack_detection/w_b_fgsm_GBA_resnet/x_black',x_black)
# np.save('/home/Bear/attack_detection/w_b_fgsm_GBA_resnet/y_black',y_black)
#
#
X = np.concatenate((x_white,x_black), axis=0)
print(X.shape)
print(np.max(X))
print(np.min(X))
Y = np.concatenate((y_white,y_black), axis=0)
print(Y.shape)
#
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
print(np.max(x_test))
print(np.min(x_test))
print(np.max(y_test))
print(np.min(y_test))
print(np.max(y_train))
print(np.min(y_train))
# np.save('/home/Bear/attack_detection/WorB/x_train56',x_train)
# np.save('/home/Bear/attack_detection/WorB/y_train56',y_train)
# np.save('/home/Bear/attack_detection/WorB/x_test56',x_test)
# np.save('/home/Bear/attack_detection/WorB/y_test56',y_test)
np.save('/home/Bear/attack_detection/WorB/X56_ad',X)
np.save('/home/Bear/attack_detection/WorB/Y56_ad',Y)
print(X.shape)
print(Y.shape)
