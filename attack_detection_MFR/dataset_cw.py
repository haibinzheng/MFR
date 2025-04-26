import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
#
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)
# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     pass  # TODO: Compute and return softmax(x)
#     x = np.array(x)
#     x = np.exp(x)
#     x.astype('float32')
#     if x.ndim == 1:
#         sumcol = sum(x)
#         for i in range(x.size):
#             x[i] = x[i]/float(sumcol)
#     if x.ndim > 1:
#         sumcol = x.sum(axis = 0)
#         for row in x:
#             for i in range(row.size):
#                 row[i] = row[i]/float(sumcol[i])
#     return x


# fgsm_renet
y_adtest01 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidence00.npy')
y_adtest02 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidence00_-25.npy')
y_adtest03 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidence00_-50.npy')
y_adtest04 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidence00_25.npy')
y_adtest05 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidence00_50.npy')
y_adtest06 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencegv001m0.npy')
y_adtest07 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencegv002m0.npy')
y_adtest08 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencegv003m0.npy')
y_adtest09 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencegv004m0.npy')
y_adtest10 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencegv005m0.npy')
y_adtest11 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencers20.npy')
y_adtest12 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencers24.npy')
y_adtest13 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencers32.npy')
y_adtest14 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencers36.npy')
y_adtest15 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencers40.npy')
y_adtest16 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencers44.npy')
y_adtest17 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidences1.npy')
y_adtest18 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidences1_-25.npy')
y_adtest19 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidences1_-50.npy')
y_adtest20 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidences1_25.npy')
y_adtest21 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidences1_50.npy')
y_adtest22 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidences2.npy')
y_adtest23 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidences2_-25.npy')
y_adtest24 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidences2_-50.npy')
y_adtest25 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidences2_25.npy')
y_adtest26 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidences2_50.npy')
y_adtest27 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencex1.npy')
y_adtest28 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencex1_-25.npy')
y_adtest29 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencex1_-50.npy')
y_adtest30 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencex1_25.npy')
y_adtest31 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencex1_50.npy')
y_adtest32 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencex2.npy')
y_adtest33 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencex2_-25.npy')
y_adtest34 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencex2_-50.npy')
y_adtest35 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencex2_25.npy')
y_adtest36 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencex2_50.npy')
y_adtest37 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencey1.npy')
y_adtest38 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencey1_-25.npy')
y_adtest39 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencey1_-50.npy')
y_adtest40 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencey1_25.npy')
y_adtest41 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencey1_50.npy')
y_adtest42 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencey2.npy')
y_adtest43 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencey2_-25.npy')
y_adtest44 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencey2_-50.npy')
y_adtest45 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencey2_25.npy')
y_adtest46 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencey2_50.npy')
y_adtest47 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencez1.npy')
y_adtest48 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencez1_-25.npy')
y_adtest49 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencez1_-50.npy')
y_adtest50 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencez1_25.npy')
y_adtest51 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencez1_50.npy')
y_adtest52 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencez2.npy')
y_adtest53 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencez2_-25.npy')
y_adtest54 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencez2_-50.npy')
y_adtest55 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencez2_25.npy')
y_adtest56 = np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_confidencez2_50.npy')


# 56
# y_test=np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/y_test00.npy')


for dim in [56,45,18,8,5]:
    if dim==56:
        y_confidenceall=np.stack((y_adtest01,y_adtest02,y_adtest03,y_adtest04,y_adtest05,y_adtest06,y_adtest07,y_adtest08,y_adtest09,y_adtest10,
                                        y_adtest11, y_adtest12, y_adtest13, y_adtest14, y_adtest15, y_adtest16, y_adtest17,
                                        y_adtest18, y_adtest19, y_adtest20,y_adtest21,y_adtest22,y_adtest23,y_adtest24,
                                        y_adtest25,y_adtest26,y_adtest27,y_adtest28,y_adtest29,y_adtest30,
                                        y_adtest31, y_adtest32, y_adtest33, y_adtest34, y_adtest35, y_adtest36, y_adtest37,
                                        y_adtest38, y_adtest39, y_adtest40,y_adtest41,y_adtest42,y_adtest43,y_adtest44,y_adtest45,
                                  y_adtest46,y_adtest47,y_adtest48,y_adtest49,y_adtest50,y_adtest51,y_adtest52,y_adtest53,y_adtest54,
                                  y_adtest55, y_adtest56),axis=1)

    # 45
    elif dim==45:
        y_confidenceall=np.stack((y_adtest01,y_adtest02,y_adtest03,y_adtest04,y_adtest05,
                                       y_adtest17,
                                        y_adtest18, y_adtest19, y_adtest20,y_adtest21,y_adtest22,y_adtest23,y_adtest24,
                                        y_adtest25,y_adtest26,y_adtest27,y_adtest28,y_adtest29,y_adtest30,
                                        y_adtest31, y_adtest32, y_adtest33, y_adtest34, y_adtest35, y_adtest36, y_adtest37,
                                        y_adtest38, y_adtest39, y_adtest40,y_adtest41,y_adtest42,y_adtest43,y_adtest44,y_adtest45,
                                  y_adtest46,y_adtest47,y_adtest48,y_adtest49,y_adtest50,y_adtest51,y_adtest52,y_adtest53,y_adtest54,
                                  y_adtest55, y_adtest56),axis=1)
    # 18
    elif dim==18:
        y_confidenceall=np.stack((y_adtest01,y_adtest17,y_adtest52,y_adtest42,y_adtest19,
                                       y_adtest21,
                                        y_adtest54, y_adtest56, y_adtest44,y_adtest46,y_adtest01,y_adtest13,y_adtest14,
                                        y_adtest11,y_adtest16,y_adtest01,y_adtest06,y_adtest10),axis=1)
    # 8
    elif dim==8:
        y_confidenceall=np.stack((y_adtest01,y_adtest17,y_adtest52,y_adtest42,y_adtest19,
                                       y_adtest21,
                                      y_adtest14,
                                   y_adtest10),axis=1)
    elif dim==5:
        y_confidenceall=np.stack((y_adtest01,y_adtest17,
                                   y_adtest21,
                                  y_adtest14,
                               y_adtest06),axis=1)

    print(y_confidenceall.shape)
    length = len(y_adtest01)
    y_confidenceall00 = np.zeros((length, dim, 10))
    print(np.max(y_confidenceall-y_confidenceall00))
    # for i in range(0, length):
    #     for j in range(0, dim):
    #         y_confidenceall00[i, j] = softmax(y_confidenceall[i, j])

    y_confidenceall00=np.array(y_confidenceall00)

    # y_confidenceall00=softmax(y_confidenceall)

    print(y_confidenceall00.shape)
    print(y_confidenceall.shape)

    print(np.max(y_confidenceall))
    print(np.min(y_confidenceall))

    print(np.max(y_confidenceall00))
    print(np.min(y_confidenceall00))

    # print(y_test[0])
    print(np.argmax(y_confidenceall[0, 0]))
    print(y_confidenceall00[0, 0])
    print((y_confidenceall[0, 0]))

    # print(np.argmax(y_confidenceall00[0,0,0]))
    # print(y_confidenceall[0])
    np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/{}'.format(str(dim))+'/x_confidence{}'.format(str(dim)), y_confidenceall00)



