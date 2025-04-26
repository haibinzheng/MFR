import numpy as np
import tensorflow as tf
import math
# def softmax(x):
#     return np.exp(x)/np.sum(np.exp(x),axis=0)
# import numpy as np
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    pass  # TODO: Compute and return softmax(x)
    x = np.array(x)
    x = np.exp(x)
    x.astype('float32')
    if x.ndim == 1:
        sumcol = sum(x)
        for i in range(x.size):
            x[i] = x[i]/float(sumcol)
    if x.ndim > 1:
        sumcol = x.sum(axis = 0)
        for row in x:
            for i in range(row.size):
                row[i] = row[i]/float(sumcol[i])
    return x

x_train0 = np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/18/x_confidence18.npy')
# y_train0 = np.load('/tmp/imagenet_fgsm/orin/y_test.npy')
# x_test0 = np.load('/home/Bear/attack_detection/WorB/Y56_ad.npy')
# y_test0 = np.load('/home/Bear/attack_detection/WorB/Y56_co.npy')
print(x_train0[0,0])
print(x_train0.shape)
# x_train0=x_train0/255
# x_test0=x_test0/255
# print(x_train0.shape)
# print(y_train0.shape)
# print(x_test0.shape)
# print(y_test0.shape)

print(np.max(x_train0))
print(np.min(x_train0))
# print(np.max(x_test0))
# print(np.min(x_test0))
# print(np.max(y_train0))
# print(np.min(y_train0))
# print(np.max(y_test0))
# print(np.min(y_test0))
# print(len(x_train0))
# print(x_test0-y_test0)
# print(x_test0)
# print(y_test0)
# np.save('/home/Bear/attack_detection/cnn_cifar_10/x_train_guiyi',x_train0)
# np.save('/home/Bear/attack_detection/cnn_cifar_10/x_test_guiyi',x_test0)
# x=np.load('')
x_train01=np.zeros((9836,10))
# for i in range(0, len(x_train0)):
#     for j in range(0, 10):
#         x_train01[i, j] = softmax(x_train0[i,j])
# x_train01=softmax(x_train0)
print(x_train01.shape)
print(np.max(x_train0))
print(np.min(x_train0))
print(np.max(x_train01))
print(np.min(x_train01))
print(x_train01[11])
print(x_train0[11])
m=0
# for i in range(0,len(x_train0)):
#     for j in range(0,18):
#         for k in range(0,10):
#             if math.isnan(x_train0[i,j,k]):
#                 # print(x_train0[i,j,k])
#                 print(i,j,k)
#                 m=m+1;
# print(m)
print(x_train0[9835,8])
y_confidence=np.load('/home/Bear/attack_detection/deepfool_mlp_mnist/mnist_all/y_confidencey2_-50.npy')
print(y_confidence[9835])
print(np.argmax(y_confidence[9835]))
x=softmax(y_confidence)
print(x[9835])
print(np.argmax(x[9835]))

A = [1.0,2.0,3.0,4.0,5.0,6.0]
with tf.Session() as session:
        session.run(tf.nn.softmax(A))
# print(x_tf[9835])