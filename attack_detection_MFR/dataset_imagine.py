import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)
# fgsm_renet
y_adtest01 = np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidence00.npy')
y_adtest02 = np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidencers303.npy')
y_adtest03 = np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidences1.npy')
y_adtest04 = np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidences1_50.npy')
y_adtest05 = np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidencegv001m0.npy')

y_adconfidenceall=np.stack((y_adtest01,y_adtest02,y_adtest03,y_adtest04,y_adtest05), axis=1)
dim=5
print(y_adconfidenceall.shape)
length = len(y_adtest01)
y_adconfidenceall00 = np.zeros((length, dim, 1001))
print(np.max(y_adconfidenceall - y_adconfidenceall00))
for i in range(0, length):
    for j in range(0, dim):
        y_adconfidenceall00[i, j] = softmax(y_adconfidenceall[i, j])

y_adconfidenceall00 = np.array(y_adconfidenceall00)

# y_confidenceall00=softmax(y_confidenceall)

print(y_adconfidenceall00.shape)
print(y_adconfidenceall.shape)

print(np.max(y_adconfidenceall))
print(np.min(y_adconfidenceall))

print(np.max(y_adconfidenceall00))
print(np.min(y_adconfidenceall00))

# print(y_test[0])
print(np.argmax(y_adconfidenceall[0, 0]))
print(y_adconfidenceall00[0, 0])
print((y_adconfidenceall[0, 0]))

np.save('/home/Bear/attack_detection/fgsm_resnet_imagenet/confidence_all/x_confidence{}'.format(str(5)),y_adconfidenceall00)
print(y_adconfidenceall[5,0,1])
print(y_adconfidenceall00[5,3,528])
print(np.argmax(y_adconfidenceall[5,2]))
print(np.argmax(y_adconfidenceall00[5,2]))
