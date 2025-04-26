import numpy as np
x_test=np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/x_test.npy')
y_test=np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_test10000.npy')
y_pred=np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_pred.npy')
x_adtest=np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/x_adtest.npy')
y_adconfidence=np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_adconfidence.npy')
y_confidence=np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_confidence.npy')
y_adtest=np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_adtest.npy')
cl=np.load('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/cl.npy')

print('x_test', x_test.shape)
print('y_test', y_test.shape)
print('x_adtest', x_adtest.shape)
print('y_adtest', y_adtest.shape)
print('y_pred', y_pred.shape)
print('y_confidence', y_confidence.shape)
print('y_adconfidence', y_adconfidence.shape)
print('cl',cl.shape)

x_testl = []
y_testl = []
x_adtestl = []
# 攻击后的类标
y_adtestl = []
# 正常预测类标
y_predl = []
y_confidencel = []
y_adconfidencel = []
cll=[]

n=0
for i in range(0,len(cl)):
    if  y_pred[i]==y_test[i] and (y_test[i]!=y_adtest[i]) :
        x_testl.append(x_test[i])
        y_testl.append(y_test[i])
        y_predl.append(y_pred[i])
        y_confidencel.append(y_confidence[i])

        x_adtestl.append(x_adtest[i])
        y_adtestl.append(y_adtest[i])
        y_adconfidencel.append(y_adconfidence[i])
        cll.append(cl[i])
        n=n+1
print(n)
x_testl = np.array(x_testl)
y_testl = np.array(y_testl)
x_adtestl = np.array(x_adtestl)
y_adtestl = np.array(y_adtestl)
y_predl= np.array(y_predl)
y_confidencel = np.array(y_confidencel)
y_adconfidencel = np.array(y_adconfidencel)
cll=np.array(cll)

print('x_testl', x_testl.shape)
print('y_testl', y_testl.shape)
print('x_adtestl', x_adtestl.shape)
print('y_adtestl', y_adtestl.shape)
print('y_predl', y_predl.shape)
print('y_confidencel', y_confidencel.shape)
print('y_adconfidencel', y_adconfidencel.shape)
print('cl1',cll.shape)

m=0
for c in range(0,n):
    if y_predl[c]==y_testl[c]:
        print(c,y_predl[c], y_testl[c],y_adtest[c])
        m=m+1
    else:
        print("88888888888888888888888888888888888888888888888888888888888888")
print(m)
print('x_test', x_testl.shape)
print('y_test', y_testl.shape)
print('x_adtest', x_adtestl.shape)
print('y_adtest', y_adtestl.shape)
print('y_pred', y_predl.shape)
print('y_confidence', y_confidencel.shape)
print('y_adconfidence', y_adconfidencel.shape)
print('cl',cll)


np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/x_test{}'.format(str(m)),x_testl)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_test{}'.format(str(m)),y_testl)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/x_adtest{}'.format(str(m)),x_adtestl)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_adtest{}'.format(str(m)),y_adtestl)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_pred{}'.format(str(m)),y_predl)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_confidence{}'.format(str(m)),y_confidencel)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/y_adconfidence{}'.format(str(m)),y_adconfidencel)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/cl{}'.format(str(m)),cll)

np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/cifar10_all/x_test00',x_testl)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/cifar10_all/y_test00',y_testl)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/cifar10_all/x_adtest00',x_adtestl)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/cifar10_all/y_adtest00',y_adtestl)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/cifar10_all/y_pred00',y_predl)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/cifar10_all/y_confidence00',y_confidencel)
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/cifar10_all/y_adconfidence00',y_adconfidencel)
