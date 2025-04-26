import numpy as np

x_test=np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/orin/x_test.npy')
y_test=np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/orin/y_test.npy')
# y_pred=np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/orin/y_test.npy')
y_confidence=np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/orin/y_confidence.npy')
y_pred=np.argmax(y_confidence,axis=1)

x_adtest=np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/orin/x_adtest.npy')
# y_adtest=np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/orin/y_adtest.npy')
y_adconfidence=np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/orin/y_adconfidence.npy')
y_adtest=np.argmax(y_adconfidence,axis=1)
cl=np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/orin/cl.npy')

print('x_test', x_test.shape)
print('y_test', y_test.shape)
print('x_adtest', x_adtest.shape)
print('y_adtest', y_adtest.shape)
print('y_pred', y_pred.shape)
print('y_confidence', y_confidence.shape)
print('y_adconfidence', y_adconfidence.shape)
print('cl',cl.shape)

length=len(x_test)
m=0
n=0

x_test1 = []
y_test1 = []
x_adtest1 = []
y_adtest1 = []
y_pred1 = []
y_confidence1 = []
y_adconfidence1 = []
cl1=[]

for i in range(0,length):
    if  y_test[i] == y_pred[i]:
        m=m+1
        x_test1.append(x_test[i])
        y_test1.append(y_test[i])
        y_pred1.append(y_pred[i])
        y_confidence1.append(y_confidence[i])

        x_adtest1.append(x_adtest[i])
        y_adtest1.append(y_adtest[i])
        y_adconfidence1.append(y_adconfidence[i])
        cl1.append(cl[i])
print('m=',m)
x_test1 = np.array(x_test1)
y_test1 = np.array(y_test1)
x_adtest1 = np.array(x_adtest1)
y_adtest1 = np.array(y_adtest1)
y_pred1 = np.array(y_pred1)
y_confidence1 = np.array(y_confidence1)
y_adconfidence1 = np.array(y_adconfidence1)
cl1=np.array(cl1)

print('x_testl', x_test1.shape)
print('y_testl', y_test1.shape)
print('x_adtestl', x_adtest1.shape)
print('y_adtestl', y_adtest1.shape)
print('y_predl', y_pred1.shape)
print('y_confidencel', y_confidence1.shape)
print('y_adconfidencel', y_adconfidence1.shape)
print('cl1',cl1.shape)

x_test2 = []
y_test2 = []
x_adtest2 = []
y_adtest2 = []
y_pred2 = []
y_confidence2 = []
y_adconfidence2 = []
cl2=[]
for i in range(0,m):
    if  y_test1[i]!=y_adtest1[i]:
        x_test2.append(x_test1[i])
        y_test2.append(y_test1[i])
        y_pred2.append(y_pred1[i])
        y_confidence2.append(y_confidence1[i])

        x_adtest2.append(x_adtest1[i])
        y_adtest2.append(y_adtest1[i])
        y_adconfidence2.append(y_adconfidence1[i])
        cl2.append(cl1[i])
        n=n+1
        # print(y_test2[i],y_pred2[i])

print('n=',n)
# print(x_test2.shape)
x_test2 = np.array(x_test2)
y_test2 = np.array(y_test2)
x_adtest2 = np.array(x_adtest2)
y_adtest2 = np.array(y_adtest2)
y_pred2 = np.array(y_pred2)
y_confidence2 = np.array(y_confidence2)
y_adconfidence2 = np.array(y_adconfidence2)
cl2=np.array(cl2)

print('x_test2', x_test2.shape)
print('y_test2l', y_test2.shape)
print('x_adtest2l', x_adtest2.shape)
print('y_adtest2l', y_adtest2.shape)
print('y_pred2l', y_pred2.shape)
print('y_confidence2l', y_confidence2.shape)
print('y_adconfidence2l', y_adconfidence2.shape)
print('cl21',cl2.shape)

# np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/x_test{}'.format(str(n)), x_test2)
# np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/y_test{}'.format(str(n)), y_test2)
# np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/x_adtest{}'.format(str(n)), x_adtest2)
# np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/y_adtest{}'.format(str(n)), y_adtest2)
# np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/y_pred{}'.format(str(n)), y_pred2)
# np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/y_confidence{}'.format(str(n)), y_confidence2)
# np.save('/home/Bear/attack_detection/CRA_vgg16_cifar10/orin/y_adconfidence{}'.format(str(n)), y_adconfidence2)
np.save('/home/Bear/attack_detection/deepfool_mlp_mnist/orin/cl{}'.format(str(n)), cl2)
#
np.save('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/x_test00', x_test2)
np.save('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/y_test00', y_test2)
np.save('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/x_adtest00', x_adtest2)
np.save('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/y_adtest00', y_adtest2)
np.save('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/y_pred00', y_pred2)
np.save('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/y_confidence00', y_confidence2)
np.save('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/y_adconfidence00', y_adconfidence2)



# for i in range(0,7390):
#     print(y_test2[i],y_pred2[i], y_adtest2[i])

print(np.max(x_test2))
print(np.min(x_test2))
print(np.max(x_adtest2))
print(np.min(x_adtest2))
print(cl2)



