import tensorflow as tf
import foolbox
import argparse
import numpy as np
from matplotlib.pyplot import imread,imshow
import keras

# adversary samples

x_adtest=np.load('/home/Bear/attack_detection/cifar10_0/x_adtest.npy')
y_adtest=np.load('/home/Bear/attack_detection/cifar10_0/y_adtest.npy')
x_test=np.load('/home/Bear/attack_detection/cifar10_0/x_test.npy')
y_test=np.load('/home/Bear/attack_detection/cifar10_0/y_test.npy')
y_pred=np.load('/home/Bear/attack_detection/cifar10_0/y_pred.npy')
y_adconfidence=np.load('/home/Bear/attack_detection/fgsm_cifar10/00/y_adconfidence.npy')
print(x_adtest.shape)
print(y_adtest.shape)
print(x_test.shape)
print(y_test.shape)
print(y_pred.shape)
# 去掉攻击不成功的图片
x_test=np.delete(x_test,[3394,3701,4136,4292,6104,7282],axis=0)
y_test=np.delete(y_test,[3394,3701,4136,4292,6104,7282],axis=0)
# y_pred=np.delete(y_pred,[3394,3701,4136,4292,6104,7282],axis=0)
print(x_test.shape)
print(y_test.shape)
print(y_pred.shape)

# 筛选出预测正确的图片
# 筛选出攻击成功的图片
print('筛选出攻击成功的图片')
column2=[]
for i in range(0,8796):
    if y_adtest[i] == y_test[i]:
        column2.append(i)
print(column2)
column2=np.array(column2)
print(column2.shape)
x_test=np.delete(x_test,column2,axis=0)
y_test=np.delete(y_test,column2,axis=0)
x_adtest=np.delete(x_adtest,column2,axis=0)
y_adtest=np.delete(y_adtest,column2,axis=0)
y_pred=np.delete(y_pred,column2,axis=0)
print(x_adtest.shape)
print(y_adtest.shape)
print(x_test.shape)
print(y_test.shape)
print(y_pred.shape)

print('筛选出预测正确的图片')
column=[]
for i in range(0,9344):
    if not y_test[i] == y_pred[i]:
        column.append(i)
column=np.array(column)
print(column.shape)
x_test=np.delete(x_test,column,axis=0)
y_test=np.delete(y_test,column,axis=0)
x_adtest=np.delete(x_adtest,column,axis=0)
y_adtest=np.delete(y_adtest,column,axis=0)
y_pred=np.delete(y_pred,column,axis=0)
print(x_adtest.shape)
print(y_adtest.shape)
print(x_test.shape)
print(y_test.shape)
print(y_pred.shape)

np.save('/home/Bear/attack_detection/fgsm_cifar10/0/x_adtest', x_adtest)
np.save('/home/Bear/attack_detection/fgsm_cifar10/0/y_adtest', y_adtest)
np.save('/home/Bear/attack_detection/fgsm_cifar10/0/y_pred', y_pred)
np.save('/home/Bear/attack_detection/fgsm_cifar10/0/x_test', x_test)
np.save('/home/Bear/attack_detection/fgsm_cifar10/0/y_test', y_test)
