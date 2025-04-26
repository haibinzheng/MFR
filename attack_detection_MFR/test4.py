import numpy as np
import skimage
import sklearn.preprocessing
# y_pred=np.load('/home/Bear/attack_detection/cw2_cifar10/cw2_con0/cw2_con0_all/y_confidence45.npy')
# print(y_pred[0,0].shape)
# y=np.load('/home/Bear/attack_detection/cw2_cifar10/cw2_con0/cw2_con0_00/y_test7539.npy')
# print(y[6666])
# def softmax(x):
#     return np.exp(x)/np.sum(np.exp(x),axis=0)
# print(softmax(y_pred[6666,0]))
# print(np.argmax(softmax(y_pred[6666,0])))
# a=softmax(y_pred[6666,0])
# m=0
# for i in range(0,10):
#     m=m+a[i]
# print(m)

# 分散数据集
# for i in range(0,len(x_test)):
#     x_test11 = []
#     x_adtest11 = []
#     x_test11.append(x_test[i])
#     x_adtest11.append(x_adtest[i])
#     x_test11=np.array(x_test11)
#     x_adtest11=np.array(x_adtest11)
#     print(x_test11.shape)
#     print(x_adtest11.shape)
#     np.save('/tmp/fgsm_resnet152_imagenet/orin/x_testaa{}'.format(str(i)),x_test11)
#     np.save('/tmp/fgsm_resnet152_imagenet/orin/x_adtestaa{}'.format(str(i)),x_adtest11)
# 生成噪声
x_test0=np.load('/tmp/fgsm_resnet152_imagenet/orin/x_test.npy')
# x_adtest0=np.load('/tmp/fgsm_resnet152_imagenet/orin/x_adtest.npy')
for i in range(5,6):
    x_testnoise=[]
    x_adtestnoise=[]
    print('noise=',i*0.01)
    x_noise=skimage.util.random_noise(x_test0,mode='gaussian',var=0.01*i,mean=0, seed=None, clip=False)
    # print(np.max(x_noise))
    # print(np.min(x_noise))
    noise=x_noise-x_test0
    print(np.max(noise))
    print(np.min(noise))
    np.save('/tmp/fgsm_resnet152_imagenet/noise/noise_gv00{}'.format(str(i))+'m0.npy', noise)





