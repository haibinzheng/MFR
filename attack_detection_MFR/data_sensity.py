import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

# ad
# y_adtest01 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1.npy')
# y_adtest02 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_1.npy')
# y_adtest03 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_3.npy')
# y_adtest04 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_5.npy')
# y_adtest05 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_7.npy')
# y_adtest06 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_9.npy')
# y_adtest07 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_10.npy')
# y_adtest08 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_15.npy')
# y_adtest09 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_20.npy')
# y_adtest10 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_25.npy')
# y_adtest11 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_30.npy')
# y_adtest12 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_40.npy')
# y_adtest13 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_45.npy')
# y_adtest14 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_50.npy')
# y_adtest15 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_-50.npy')
# y_adtest16 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_-25.npy')
# y_adtest17 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_-10.npy')
# y_adtest18 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_adconfidences1_-1.npy')
#
# y_adconfidenceall = np.stack((y_adtest01, y_adtest02, y_adtest03, y_adtest04, y_adtest05, y_adtest06, y_adtest07,
#                             y_adtest08, y_adtest09, y_adtest10,
#                             y_adtest11, y_adtest12, y_adtest13, y_adtest14, y_adtest15, y_adtest16, y_adtest17,
#                             y_adtest18), axis=1)
# print(y_adconfidenceall.shape)
# dim=18
# length = len(y_adtest01)
# y_adconfidenceall00 = np.zeros((length, dim, 10))
# # print(np.max(y_confidenceall-y_confidenceall00))
# for i in range(0, length):
#     for j in range(0, dim):
#         y_adconfidenceall00[i, j] = softmax(y_adconfidenceall[i, j])
#
# # y_confidenceall00=np.array(y_confidenceall00)
#
# print(y_adconfidenceall00.shape)
# print(y_adconfidenceall.shape)
#
# print(np.max(y_adconfidenceall))
# print(np.min(y_adconfidenceall))
#
# print(np.max(y_adconfidenceall00))
# print(np.min(y_adconfidenceall00))
#
# # print(y_test[0])
# print(np.argmax(y_adconfidenceall[0, 0]))
# print(y_adconfidenceall00[0, 0])
# print((y_adconfidenceall[0, 0]))
#
# # print(np.argmax(y_confidenceall00[0,0,0]))
# # print(y_confidenceall[0])
#
# np.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/adsensibility18',y_adconfidenceall00)
# *********************************************************************************************************************
# # co
# y_adtest01 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1.npy')
# y_adtest02 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_1.npy')
# y_adtest03 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_3.npy')
# y_adtest04 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_5.npy')
# y_adtest05 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_7.npy')
# y_adtest06 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_9.npy')
# y_adtest07 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_10.npy')
# y_adtest08 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_15.npy')
# y_adtest09 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_20.npy')
# y_adtest10 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_25.npy')
# y_adtest11 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_30.npy')
# y_adtest12 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_40.npy')
# y_adtest13 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_45.npy')
# y_adtest14 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_50.npy')
# y_adtest15 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_-50.npy')
# y_adtest16 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_-25.npy')
# y_adtest17 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_-10.npy')
# y_adtest18 = np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/y_confidences1_-1.npy')
#
# y_confidenceall = np.stack((y_adtest01, y_adtest02, y_adtest03, y_adtest04, y_adtest05, y_adtest06, y_adtest07,
#                             y_adtest08, y_adtest09, y_adtest10,
#                             y_adtest11, y_adtest12, y_adtest13, y_adtest14, y_adtest15, y_adtest16, y_adtest17,
#                             y_adtest18), axis=1)
# print(y_confidenceall.shape)
# print(y_confidenceall.shape)
# length = len(y_adtest01)
# dim=18
# y_confidenceall00 = np.zeros((length, dim, 10))
# # print(np.max(y_confidenceall-y_confidenceall00))
# for i in range(0, length):
#     for j in range(0, dim):
#         y_confidenceall00[i, j] = softmax(y_confidenceall[i, j])
#
# y_confidenceall00=np.array(y_confidenceall00)
#
# print(y_confidenceall00.shape)
# print(y_confidenceall.shape)
#
# print(np.max(y_confidenceall))
# print(np.min(y_confidenceall))
#
# print(np.max(y_confidenceall00))
# print(np.min(y_confidenceall00))
#
# # print(y_test[0])
# print(np.argmax(y_confidenceall[0, 0]))
# print(y_confidenceall00[0, 0])
# print((y_confidenceall[0, 0]))

# print(np.argmax(y_confidenceall00[0,0,0]))
# print(y_confidenceall[0])

# np.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/sensibility18',y_confidenceall00)
# *********************************************************************************************************************
sen=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/sensibility18.npy')
adsen=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/adsensibility18.npy')
x_co=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/5/x_confidence5.npy')
x_adco=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/5/x_adconfidence5.npy')
# x_co4=np.delete(x_co,2,axis=1)
# print(x_co4.shape)
# print(x_co[0])
# print(x_co4[0])
print(x_co[4,1])
x_co01=np.zeros((len(x_co), 5, 10))
x_adco01=np.zeros((len(x_co), 5, 10))
print(x_co01.shape)
print(x_adco01.shape)
for ii in range(0,len(x_co)):
    for jj in range(0,5):
        x_co01[ii,jj]=x_co[ii,jj]
        x_adco01[ii,jj]=x_adco[ii,jj]
print(x_co01.shape)
print(x_adco01.shape)
# x_co01[:,2]=sen[:,0]
for i in range(0,18):
    print('---i=',i)
    for j in range(0,len(x_co)):
        x_co01[j,2]=sen[j,i]
        x_adco01[j,2]=adsen[j,i]
    print('sen0', sen[4, i])
    print('***************************************************************')
    print(x_co[4, 2])
    print('***************************************************************')
    print(x_co01[4, 2])
    print(x_co01.shape)
    print('------------------------------------------------------------------------------')

    print('adsen0', adsen[3, i])
    print('***************************************************************')
    print(x_adco01[3, 2])
    print('***************************************************************')
    print(x_adco[3, 2])
    print(x_adco01.shape)
    X=np.concatenate((x_co01,x_adco01),axis=0)
    print(X.shape)
    y_co01=np.zeros((len(x_co01)))
    y_adco01=np.zeros((len(x_adco01)))+1
    print(y_co01)
    print(y_adco01)
    Y=np.concatenate((y_co01,y_adco01),axis=0)
    print(Y.shape)
    np.save('/tmp/sensity/X_angle{}'.format(str(i)),X)
    np.save('/tmp/sensity/Y_angle{}'.format(str(i)),Y)





