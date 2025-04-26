import numpy as np
import keras
from sklearn.model_selection import train_test_split

cw_vgg16=np.load('/home/Bear/attack_detection/cw_vgg16_cifar10/cifar10_all/5/x_adconfidence5.npy')
fgsm_vgg16=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/5/x_adconfidence5.npy')
df_vgg16=np.load('/home/Bear/attack_detection/deepfool_vgg16_cifar10/cifar10_all/5/x_adconfidence5.npy')
cwfgsmdf_vgg16=np.concatenate((cw_vgg16,fgsm_vgg16,df_vgg16),axis=0)
print(cw_vgg16.shape)
print(fgsm_vgg16.shape)
print(df_vgg16.shape)
print(cwfgsmdf_vgg16.shape)
y1=np.zeros((len(cw_vgg16)))
y2=np.zeros((len(fgsm_vgg16)))+1
y3=np.zeros((len(df_vgg16)))+2
y=np.concatenate((y1,y2,y3),axis=0)
print(y1.shape)
print(y2.shape)
print(y3.shape)
print(y.shape)
np.save('/home/Bear/attack_detection/classify/cwfgsmdf_vgg16/X',cwfgsmdf_vgg16)
np.save('/home/Bear/attack_detection/classify/cwfgsmdf_vgg16/Y',y)
x_train, x_test, y_train, y_test = train_test_split(cwfgsmdf_vgg16, y, test_size=0.3)
print(np.max(x_test))
print(np.min(x_test))
print(np.max(y_test))
print(np.min(y_test))
print(np.max(y_train))
print(np.min(y_train))
np.save('/home/Bear/attack_detection/classify/cwfgsmdf_vgg16/x_train',x_train)
np.save('/home/Bear/attack_detection/classify/cwfgsmdf_vgg16/y_train',y_train)
np.save('/home/Bear/attack_detection/classify/cwfgsmdf_vgg16/x_test',x_test)
np.save('/home/Bear/attack_detection/classify/cwfgsmdf_vgg16/y_test',y_test)
