import numpy as np
import scipy.misc
# import imagenet_utils as iu

# x_adtest=iu.get_val_dataflow( datadir='/tmp/fgsm_resnet152_imagenet/orin/x_adtest.npy', batch_size=128)

x_test=np.load('/tmp/fgsm_resnet152_imagenet/orin1/x_test2000.npy')
x_adtest=np.load('/tmp/fgsm_resnet152_imagenet/orin1/x_adtest2000.npy')
print(np.max(x_test))
print(np.min(x_test))
print(np.max(x_adtest))
print(np.min(x_adtest))

length=len(x_test)
x_adtest=x_adtest*255
x_test=x_test*255

print(np.max(x_test))
print(np.min(x_test))
print(np.max(x_adtest))
print(np.min(x_adtest))

# x_test=np.reshape(x_test,(len(x_test),28,28))
# x_adtest=np.reshape(x_adtest,(len(x_adtest),28,28))
#
# print(x_test.shape)
# print(x_adtest.shape)

for j in [303,310,350,400,450,250,280,290,295]:
    print(x_test.shape)
    print(x_adtest.shape)
    print('size=',j)
    x_testrs1=[]
    x_adtestrs1=[]
    for i in range(0,length):
        x_testrs1.append(scipy.misc.imresize(x_test[i],(j,j), interp='bilinear'))
    x_testreshape1 = np.array(x_testrs1)

    for i in range(0,length):
        x_adtestrs1.append(scipy.misc.imresize(x_adtest[i],(j,j), interp='bilinear'))
    x_adtestreshape1 = np.array(x_adtestrs1)

    print(x_testreshape1.shape)
    print(x_adtestreshape1.shape)

    x_testrs2=[]
    x_adtestrs2=[]
    for i in range(0,length):
        x_testrs2.append(scipy.misc.imresize(x_testreshape1[i],(299,299), interp='bilinear'))
    x_testreshape2 = np.array(x_testrs2)

    for i in range(0,length):
        x_adtestrs2.append(scipy.misc.imresize(x_adtestreshape1[i],(299,299), interp='bilinear'))
    x_adtestreshape2 = np.array(x_adtestrs2)

    print(x_testreshape2.shape)
    print(x_adtestreshape2.shape)

    x_adtestreshape2=x_adtestreshape2/255
    x_testreshape2=x_testreshape2/255

    print(np.max(x_testreshape2))
    print(np.min(x_testreshape2))
    print(np.max(x_adtestreshape2))
    print(np.min(x_adtestreshape2))

    np.save('/tmp/fgsm_resnet152_imagenet/image_all/all_2000/x_testrs{}'.format(str(j)),x_testreshape2)
    np.save('/tmp/fgsm_resnet152_imagenet/image_all/all_2000/x_adtestrs{}'.format(str(j)),x_adtestreshape2)