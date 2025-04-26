from __future__ import print_function
import numpy as np

x_test=np.load('/home/Bear/attack_detection/cnn_cifar_10/x_test10000.npy')
y_test=np.load('/home/Bear/attack_detection/cnn_cifar_10/y_test10000.npy')
y_pred=np.load('/home/Bear/attack_detection/cnn_cifar_10/y_pred10000.npy')
y_test=np.argmax(y_test,axis=1)
y_pred=np.argmax(y_pred,axis=1)
print(x_test.shape)
print(y_test.shape)
print(y_pred.shape)
# y_test=y_test.reshape((10000,))
# print(y_pred.shape)

x_testl=[]
y_testl=[]
y_predl=[]
m=0
for i in range(0,10000):
    if  y_pred[i]==y_test[i]:
        x_testl.append(x_test[i])
        y_testl.append(y_test[i])
        y_predl.append(y_pred[i])
        m=m+1
print(m)
x_testl=np.array(x_testl)
y_testl=np.array(y_testl)
y_predl=np.array(y_predl)
print(x_testl.shape)
print(y_testl.shape)
print(y_predl.shape)
n=0
for id in range(0,7539):
    if y_predl[id]==y_testl[id]:
        print(y_predl[id],y_testl[id])
        n=n+1
print(n)
print(x_testl.shape)
print(y_testl.shape)
print(y_predl.shape)
m=0
for c in range(0,9194):
    if y_predl[c]==y_testl[c]:
        print(c,y_predl[c], y_testl[c])
        m=m+1
    else:
        print("88888888888888888888888888888888888888888888888888888888888888")
print(m)
print(x_testl.shape)
print(y_testl.shape)
print(y_predl.shape)



np.save('/home/Bear/attack_detection/cnn_cifar_10/x_test7539',x_testl)
np.save('/home/Bear/attack_detection/cnn_cifar_10/y_test7539',y_testl)
np.save('/home/Bear/attack_detection/cnn_cifar_10/y_pred7539',y_predl)

