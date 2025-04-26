import numpy as np
from sklearn.metrics import accuracy_score
y_adtest01 = np.argmax(np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_adconfidence00.npy'),axis=1)
y_adtest02 = np.argmax(np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_adconfidencers303.npy'),axis=1)
y_adtest03 = np.argmax(np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_adconfidences1.npy'),axis=1)
y_adtest04 = np.argmax(np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_adconfidences1_50.npy'),axis=1)
y_adtest05 = np.argmax(np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_adconfidencegv001m0.npy'),axis=1)

y_test01 =np.argmax( np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidence00.npy'),axis=1)
y_test02 = np.argmax(np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidencers303.npy'),axis=1)
y_test03 = np.argmax(np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidences1.npy'),axis=1)
y_test04 =np.argmax( np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidences1_50.npy'),axis=1)
y_test05 =np.argmax( np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidencegv001m0.npy'),axis=1)

print(y_adtest01.shape)
c1=accuracy_score(y_test01,y_test01)
a1=accuracy_score(y_adtest01,y_test01)

c2=accuracy_score(y_test02,y_test01)
a2=accuracy_score(y_adtest02,y_test01)

c3=accuracy_score(y_test03,y_test01)
a3=accuracy_score(y_adtest03,y_test01)

c4=accuracy_score(y_test04,y_test01)
a4=accuracy_score(y_adtest04,y_test01)

c5=accuracy_score(y_test05,y_test01)
a5=accuracy_score(y_adtest05,y_test01)
print(c1,a1)
print(c2,a2)
print(c3,a3)
print(c4,a4)
print(c5,a5)